"""The rclpy node. Subscribes to a camera, looks up the robot pose in TF, ingests in the
background, and answers questions published on `~/ask` with JSON on `~/answer`.

Run with `placecell-ros2` inside a sourced ROS 2 environment, or `ros2 run` once packaged.
Configuration is plain ROS parameters; the API key comes from the environment variable named
by `api_key_env`, never from a parameter, so it does not end up in launch files or logs.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from placecell.agent import Agent
from placecell.errors import PlacecellError
from placecell.lifecycle import Curator
from placecell.memory import Pose
from placecell.pipeline import Ingester, Observation, SegmentationPolicy, Segmenter
from placecell.providers import Captioner, EmbeddingProvider, HashingEmbedder
from placecell.retrieval import Recall
from placecell.ros2.bridge import KeyframeWriter, ObservationBuilder, pose_from_transform, stamp_to_seconds
from placecell.store import CollectionInfo, VectorStore


def build_embedder(base_url: str, model: str, api_key: str | None, dimension: int) -> EmbeddingProvider:
    if not model:
        return HashingEmbedder()
    from placecell.providers import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder(model, base_url, api_key, dimension=dimension or None)


def build_store(db_path: str, collection: str, embedder: EmbeddingProvider) -> VectorStore:
    info = CollectionInfo(collection, embedder.model_name, embedder.dimension)
    if not db_path:
        from placecell.store import InMemoryStore

        return InMemoryStore(info)
    from placecell.store.lancedb_store import LanceDBStore

    return LanceDBStore(Path(db_path).expanduser(), info)


def answer_payload(question: str, text: str, grounded: bool, evidence: Sequence[Any]) -> str:
    return json.dumps(
        {
            "question": question,
            "answer": text,
            "grounded": grounded,
            "evidence": [
                {
                    "id": r.memory.id,
                    "x": r.memory.pose.x,
                    "y": r.memory.pose.y,
                    "yaw": r.memory.pose.yaw,
                    "time": r.memory.timestamp,
                    "caption": r.memory.caption,
                    "confidence": r.confidence,
                }
                for r in evidence
            ],
        }
    )


class IngestWorker:
    """Batches observations from a queue into the ingester on its own thread."""

    def __init__(self, ingester: Ingester, lock: threading.Lock, batch_size: int, max_queue: int, log: Any) -> None:
        self._ingester = ingester
        self._lock = lock
        self._batch_size = batch_size
        self._queue: queue.Queue[Observation] = queue.Queue(maxsize=max_queue)
        self._log = log
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="placecell-ingest", daemon=True)
        self.dropped = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def submit(self, observation: Observation) -> None:
        try:
            self._queue.put_nowait(observation)
        except queue.Full:
            self.dropped += 1
            self._log.warning(f"ingest queue full, dropped {self.dropped} observations so far")

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._drain()
            if not batch:
                continue
            try:
                with self._lock:
                    report = self._ingester.ingest(batch)
            except PlacecellError as e:
                self._log.error(f"ingest failed: {e}")
                continue
            self._log.info(
                f"ingested {report.accepted}/{report.received}: {report.inserted} new, {report.merged} reinforced"
                + (f", {report.unsupported} unsupported" if report.unsupported else "")
            )

    def _drain(self) -> list[Observation]:
        batch: list[Observation] = []
        try:
            batch.append(self._queue.get(timeout=0.5))
        except queue.Empty:
            return batch
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch


def main(args: list[str] | None = None) -> None:  # pragma: no cover - needs a ROS 2 environment
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    class PlacecellNode(Node):
        def __init__(self) -> None:
            super().__init__("placecell")
            p = self._params()
            api_key = os.environ.get(p["api_key_env"]) or None
            embedder = build_embedder(p["embed_base_url"], p["embed_model"], api_key, p["embed_dimension"])
            store = build_store(p["db_path"], p["collection"], embedder)
            captioner: Captioner | None = None
            if p["caption_model"]:
                from placecell.providers import OpenAICompatibleCaptioner

                captioner = OpenAICompatibleCaptioner(p["caption_model"], p["caption_base_url"], api_key)
            if captioner is None and not embedder.capabilities.image:
                self.get_logger().warning(
                    "no caption_model and the embedder takes text only: frames cannot be stored. "
                    "Set caption_model, or use an embedding model that accepts images."
                )
            segmenter = Segmenter(SegmentationPolicy(p["min_interval_s"], p["min_travel_m"], p["min_turn_rad"]))
            ingester = Ingester(embedder, store, captioner, segmenter, batch_size=p["batch_size"])
            self._recall = Recall(store, embedder)
            self._agent: Agent | None = None
            if p["chat_model"]:
                from placecell.providers import OpenAICompatibleChat

                chat = OpenAICompatibleChat(p["chat_model"], p["chat_base_url"], api_key)
                self._agent = Agent(self._recall, chat, frame_id=p["map_frame"])
            self._lock = threading.Lock()
            self._worker = IngestWorker(ingester, self._lock, p["batch_size"], p["max_queue"], self.get_logger())
            self._curator = Curator(store)
            writer = KeyframeWriter(Path(p["keyframe_dir"]).expanduser())
            self._builder = ObservationBuilder(p["robot_id"], p["camera_id"], writer)
            self._map_frame, self._base_frame, self._map_id = p["map_frame"], p["base_frame"], p["map_id"]
            self._tf_timeout = p["tf_timeout_s"]
            self._tf = Buffer()
            self._tf_listener = TransformListener(self._tf, self)
            if p["compressed"]:
                self.create_subscription(
                    CompressedImage, p["image_topic"], self._on_compressed, qos_profile_sensor_data
                )
            else:
                self.create_subscription(Image, p["image_topic"], self._on_image, qos_profile_sensor_data)
            self.create_subscription(String, "~/ask", self._on_ask, 10)
            self._answers = self.create_publisher(String, "~/answer", 10)
            if p["curator_interval_s"] > 0:
                self.create_timer(p["curator_interval_s"], self._curate)
            self._worker.start()
            where = f"lancedb {p['db_path']}" if p["db_path"] else "memory"
            self.get_logger().info(
                f"placecell up: robot {p['robot_id']}, camera {p['camera_id']}, model {embedder.model_name}, "
                f"store {where}, agent {'on' if self._agent else 'off'}"
            )

        def _params(self) -> dict[str, Any]:
            defaults: dict[str, Any] = {
                "robot_id": "robot",
                "camera_id": "front",
                "image_topic": "/camera/color/image_raw",
                "compressed": False,
                "map_frame": "map",
                "base_frame": "base_footprint",
                "map_id": "",
                "db_path": "~/.placecell/db",
                "collection": "default",
                "keyframe_dir": "~/.placecell/keyframes",
                "embed_base_url": "https://api.openai.com/v1",
                "embed_model": "",
                "embed_dimension": 0,
                "caption_base_url": "https://api.openai.com/v1",
                "caption_model": "",
                "chat_base_url": "https://api.openai.com/v1",
                "chat_model": "",
                "api_key_env": "PLACECELL_API_KEY",
                "min_interval_s": 2.0,
                "min_travel_m": 0.3,
                "min_turn_rad": 0.35,
                "batch_size": 8,
                "max_queue": 64,
                "tf_timeout_s": 0.2,
                "curator_interval_s": 3600.0,
            }
            return {k: self.declare_parameter(k, v).value for k, v in defaults.items()}

        def _pose_at(self, sec: int, nanosec: int) -> Pose | None:
            from rclpy.duration import Duration
            from rclpy.time import Time

            try:
                tf = self._tf.lookup_transform(
                    self._map_frame,
                    self._base_frame,
                    Time(seconds=sec, nanoseconds=nanosec),
                    Duration(seconds=self._tf_timeout),
                )
            except TransformException as e:
                self.get_logger().warning(f"no pose for image: {e}", throttle_duration_sec=5.0)
                return None
            t, q = tf.transform.translation, tf.transform.rotation
            return pose_from_transform(t.x, t.y, q.x, q.y, q.z, q.w, self._map_frame, self._map_id)

        def _on_image(self, msg: Any) -> None:
            pose = self._pose_at(msg.header.stamp.sec, msg.header.stamp.nanosec)
            if pose is None:
                return
            stamp = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
            try:
                obs = self._builder.from_raw(
                    stamp, msg.height, msg.width, msg.encoding, msg.step, bytes(msg.data), pose
                )
            except PlacecellError as e:
                self.get_logger().warning(f"skipped image: {e}", throttle_duration_sec=5.0)
                return
            self._worker.submit(obs)

        def _on_compressed(self, msg: Any) -> None:
            pose = self._pose_at(msg.header.stamp.sec, msg.header.stamp.nanosec)
            if pose is None:
                return
            stamp = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
            try:
                obs = self._builder.from_compressed(stamp, msg.format, bytes(msg.data), pose)
            except PlacecellError as e:
                self.get_logger().warning(f"skipped image: {e}", throttle_duration_sec=5.0)
                return
            self._worker.submit(obs)

        def _on_ask(self, msg: Any) -> None:
            threading.Thread(target=self._answer, args=(msg.data,), daemon=True).start()

        def _answer(self, question: str) -> None:
            try:
                with self._lock:
                    if self._agent is not None:
                        result = self._agent.ask(question)
                        payload = answer_payload(question, result.text, result.grounded, result.evidence)
                    else:
                        hits = self._recall.similar(question, k=5)
                        text = hits[0].memory.caption if hits else "No matching memory."
                        payload = answer_payload(question, text, bool(hits), hits)
            except PlacecellError as e:
                payload = json.dumps({"question": question, "error": str(e)})
            self._answers.publish(String(data=payload))

        def _curate(self) -> None:
            with self._lock:
                report = self._curator.run()
            if report.removed:
                self.get_logger().info(f"curator removed {report.removed} memories")

        def destroy_node(self) -> bool:
            self._worker.stop()
            return bool(super().destroy_node())

    import signal

    from rclpy.signals import SignalHandlerOptions

    # Own the signals: rclpy's handler tears the context down from inside the signal handler,
    # which races with the executor's wait set. A flag lets the loop finish its iteration instead.
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = PlacecellNode()
    try:
        while not stop.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
