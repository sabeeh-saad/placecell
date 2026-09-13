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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from placecell.agent import Agent
from placecell.consolidation import ChatSummarizer, Consolidator
from placecell.corrections import JsonlCorrectionLog, correction_now
from placecell.errors import PlacecellError, ValidationError
from placecell.lifecycle import Curator, remove_local_file
from placecell.memory import Pose
from placecell.observer import Observer
from placecell.pipeline import Ingester, Observation, SegmentationPolicy, Segmenter
from placecell.providers import Captioner, EmbeddingProvider, HashingEmbedder
from placecell.refinement import REFINEMENT_PROMPT, MemoryRefiner, RefinementPolicy
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
                    "time": r.observed_at[0] if r.observed_at else r.memory.timestamp,
                    "last_seen": r.memory.last_seen,
                    "observed_at": list(r.observed_at or r.memory.sighting_times),
                    "caption": r.memory.caption,
                    "confidence": r.confidence,
                }
                for r in evidence
            ],
        }
    )


class IngestWorker:
    """One ordered writer consuming durable jobs. Provider calls never hold a shared lock."""

    def __init__(
        self,
        ingester: Ingester,
        lock: threading.Lock | None,
        batch_size: int,
        max_queue: int,
        log: Any,
        *,
        max_attempts: int = 5,
        retry_delay_s: float = 1,
    ) -> None:
        if min(batch_size, max_queue, max_attempts) < 1 or retry_delay_s < 0:
            raise ValidationError("invalid worker limits")
        self._ingester, self._batch_size, self._max_queue = ingester, batch_size, max_queue
        self._log, self._max_attempts, self._retry_delay_s = log, max_attempts, retry_delay_s
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="placecell-ingest", daemon=True)
        self._submit_lock = threading.Lock()
        self.dropped = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10) -> bool:
        with self._submit_lock:
            self._stop.set()
        self._wake.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def has_capacity(self) -> bool:
        return not self._stop.is_set() and self._ingester.jobs.stats()["queued"] < self._max_queue

    def submit(self, observation: Observation) -> bool:
        with self._submit_lock:
            if not self._stop.is_set() and self._ingester.jobs.enqueue(observation, self._max_queue):
                self._wake.set()
                return True
            self.dropped += 1
        # The journal pins all accepted evidence, including failed jobs and duplicate submissions.
        self._ingester.discard([observation])
        self._log.warning(f"ingest queue full or stopped, dropped {self.dropped} observations so far")
        return False

    def _run(self) -> None:
        try:
            self._work_loop()
        except Exception as e:
            self._log.error(f"ingest worker stopped; queued work retained: {e}")
        finally:
            self._stop.set()

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            jobs = self._ingester.jobs.pending(self._batch_size)
            if not jobs:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            try:
                report = self._ingester.ingest([job.observation for job in jobs], preselected=True)
            except Exception as e:
                completed = [job.id for job in jobs if self._ingester.persisted(job.observation)]
                self._ingester.jobs.complete(completed)
                self._ingester.jobs.fail(
                    (job.id for job in jobs if job.id not in completed),
                    str(e),
                    max_attempts=self._max_attempts,
                    retry_delay_s=self._retry_delay_s,
                )
                self._log.error(f"ingest failed; work retained for retry: {e}")
            else:
                self._ingester.jobs.complete(job.id for job in jobs)
                self._log.info(
                    f"ingested {report.accepted}/{report.received}: {report.inserted} new, {report.merged} reinforced"
                    + (f", {report.unsupported} unsupported" if report.unsupported else "")
                )
            try:
                self._ingester.discard([])  # drain cleanup intents after job ownership is released
            except OSError as e:
                self._log.error(f"evidence cleanup deferred: {e}")


class BoundedTasks:
    """Fixed daemon workers and a bounded waiting queue for questions or maintenance."""

    def __init__(self, workers: int, capacity: int, log: Any) -> None:
        if min(workers, capacity) < 1:
            raise ValidationError("task limits must be positive")
        self._queue: queue.Queue[tuple[Callable[..., None], tuple[Any, ...]]] = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._log = log
        self._threads = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for thread in self._threads:
            thread.start()

    def submit(self, function: Callable[..., None], *args: Any) -> bool:
        if self._stop.is_set():
            return False
        try:
            self._queue.put_nowait((function, args))
        except queue.Full:
            return False
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                function, args = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                function(*args)
            except Exception as e:
                self._log.error(f"background task failed: {e}")

    def stop(self, timeout: float = 10) -> bool:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout / len(self._threads))
        return all(not thread.is_alive() for thread in self._threads)


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
            policy = SegmentationPolicy(p["min_interval_s"], p["min_travel_m"], p["min_turn_rad"], p["max_interval_s"])
            segmenter = Segmenter(policy)
            self._admission = Segmenter(policy)
            self._robot_id, self._camera_id = p["robot_id"], p["camera_id"]
            self._store = store
            observer = Observer(store) if p["contradiction"] else None
            ingester = Ingester(embedder, store, captioner, segmenter, batch_size=p["batch_size"], observer=observer)
            self._corrections = JsonlCorrectionLog(Path(p["corrections_path"]).expanduser())
            self._recall = Recall(store, embedder, corrections=self._corrections)
            self._agent: Agent | None = None
            self._consolidator: Consolidator | None = None
            self._refiner: MemoryRefiner | None = None
            refinement_model = p["refine_model"] or p["caption_model"]
            if p["refine_interval_s"] > 0 and refinement_model:
                from placecell.providers import OpenAICompatibleCaptioner

                reviewer = OpenAICompatibleCaptioner(
                    refinement_model, p["caption_base_url"], api_key, prompt=REFINEMENT_PROMPT, detail="high"
                )
                self._refiner = MemoryRefiner(
                    store,
                    embedder,
                    reviewer,
                    RefinementPolicy(max_memories=p["refine_batch_size"]),
                    producer=refinement_model,
                )
            if p["chat_model"]:
                from placecell.providers import OpenAICompatibleChat

                chat = OpenAICompatibleChat(p["chat_model"], p["chat_base_url"], api_key)
                self._agent = Agent(self._recall, chat, frame_id=p["map_frame"], map_id=p["map_id"])
                if p["consolidate_interval_s"] > 0:
                    self._consolidator = Consolidator(store, embedder, ChatSummarizer(chat))
            self._worker = IngestWorker(
                ingester,
                None,
                p["batch_size"],
                p["max_queue"],
                self.get_logger(),
                max_attempts=p["ingest_attempts"],
                retry_delay_s=p["ingest_retry_delay_s"],
            )
            self._questions = BoundedTasks(p["question_workers"], p["question_queue"], self.get_logger())
            self._maintenance = BoundedTasks(1, 1, self.get_logger())
            self._curator = Curator(store, corrections=self._corrections, remover=remove_local_file)
            writer = KeyframeWriter(Path(p["keyframe_dir"]).expanduser())
            writer.recover_pending(store)
            store.drain_cleanup(remove_local_file)
            self._writer = writer
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
            self.create_subscription(String, "~/correct", self._on_correct, 10)
            self.create_subscription(String, "~/refine", self._on_refine, 10)
            self._answers = self.create_publisher(String, "~/answer", 10)
            if p["curator_interval_s"] > 0:
                self.create_timer(p["curator_interval_s"], self._curate)
            if self._consolidator is not None:
                self.create_timer(p["consolidate_interval_s"], self._consolidate)
            if self._refiner is not None:
                self.create_timer(p["refine_interval_s"], self._refine)
            self.create_timer(30.0, self._diagnostics)
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
                "max_interval_s": 60.0,
                "ingest_attempts": 5,
                "ingest_retry_delay_s": 1.0,
                "question_workers": 2,
                "question_queue": 8,
                "min_travel_m": 0.3,
                "min_turn_rad": 0.35,
                "batch_size": 8,
                "max_queue": 64,
                "tf_timeout_s": 0.2,
                "curator_interval_s": 3600.0,
                "contradiction": True,
                "corrections_path": "~/.placecell/corrections.jsonl",
                "consolidate_interval_s": 0.0,
                "refine_interval_s": 3600.0,
                "refine_batch_size": 8,
                "refine_model": "",
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
            if not self._admission.eligible(self._robot_id, self._camera_id, stamp, pose):
                return
            if not self._worker.has_capacity():
                self._worker.dropped += 1
                return
            try:
                obs = self._builder.from_raw(
                    stamp, msg.height, msg.width, msg.encoding, msg.step, bytes(msg.data), pose
                )
            except PlacecellError as e:
                self.get_logger().warning(f"skipped image: {e}", throttle_duration_sec=5.0)
                return
            if self._worker.submit(obs):
                self._admission.accept(obs)
            self._writer.confirm(obs.evidence)

        def _on_compressed(self, msg: Any) -> None:
            pose = self._pose_at(msg.header.stamp.sec, msg.header.stamp.nanosec)
            if pose is None:
                return
            stamp = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
            if not self._admission.eligible(self._robot_id, self._camera_id, stamp, pose):
                return
            if not self._worker.has_capacity():
                self._worker.dropped += 1
                return
            try:
                obs = self._builder.from_compressed(stamp, msg.format, bytes(msg.data), pose)
            except PlacecellError as e:
                self.get_logger().warning(f"skipped image: {e}", throttle_duration_sec=5.0)
                return
            if self._worker.submit(obs):
                self._admission.accept(obs)
            self._writer.confirm(obs.evidence)

        def _on_ask(self, msg: Any) -> None:
            if not self._questions.submit(self._answer, msg.data):
                self._answers.publish(String(data=json.dumps({"question": msg.data, "error": "question queue full"})))

        def _answer(self, question: str) -> None:
            try:
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

        def _on_correct(self, msg: Any) -> None:
            """JSON: {"memory_id": ..., "verdict": "right"|"wrong", "question": ..., "note": ...}."""
            try:
                data = json.loads(msg.data)
                correction = correction_now(
                    str(data["memory_id"]),
                    str(data["verdict"]),
                    str(data.get("question", "")),
                    str(data.get("note", "")),
                )
            except (ValueError, KeyError, TypeError, PlacecellError) as e:
                self.get_logger().warning(f"ignored correction: {e}")
                return
            self._corrections.record(correction)
            if correction.verdict == "wrong":
                self._store.refinements.request(correction.memory_id, "operator correction")

        def _on_refine(self, msg: Any) -> None:
            """JSON: {"memory_id": ..., "action": "recheck"|"rollback"}."""
            try:
                data = json.loads(msg.data)
                identity, action = str(data["memory_id"]), data.get("action", "recheck")
                if action == "recheck":
                    accepted = self._store.refinements.request(identity)
                elif action == "rollback" and self._refiner is not None:
                    accepted = self._refiner.rollback(identity)
                else:
                    raise ValidationError("unknown refinement action or refinement is disabled")
                self.get_logger().info(f"refinement {action} for {identity}: {'accepted' if accepted else 'skipped'}")
            except (ValueError, KeyError, TypeError, PlacecellError) as e:
                self.get_logger().warning(f"ignored refinement request: {e}")

        def _refine(self) -> None:
            self._maintenance.submit(self._run_refiner)

        def _run_refiner(self) -> None:
            if self._refiner is not None:
                report = self._refiner.run()
                if report.attempted:
                    self.get_logger().info(f"memory refinement: {report}")

        def _curate(self) -> None:
            self._maintenance.submit(self._run_curator)

        def _run_curator(self) -> None:
            report = self._curator.run()
            maintain = getattr(self._store, "maintain", None)
            if maintain is not None:
                maintain()
            if report.removed or report.discredited:
                self.get_logger().info(f"curator removed {report.removed} memories, discredited {report.discredited}")

        def _consolidate(self) -> None:
            self._maintenance.submit(self._run_consolidator)

        def _run_consolidator(self) -> None:
            if self._consolidator is None:  # pragma: no cover - timer only exists with a consolidator
                return
            try:
                report = self._consolidator.run()
            except PlacecellError as e:
                self.get_logger().error(f"consolidation failed: {e}")
                return
            if report.summaries:
                self.get_logger().info(f"consolidated {report.folded} memories into {report.summaries} summaries")

        def _diagnostics(self) -> None:
            stats = self._store.jobs.stats()
            self.get_logger().info(f"ingestion: {stats}, dropped={self._worker.dropped}")

        def destroy_node(self) -> bool:
            ingested = self._worker.stop()
            answered = self._questions.stop()
            maintained = self._maintenance.stop()
            if ingested and answered and maintained:
                self._store.close()
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
