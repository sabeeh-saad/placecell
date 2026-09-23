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
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from placecell.agent import Agent
from placecell.approach import ApproachPlanner, ApproachPolicy
from placecell.command_identity import CommandJournal, CommandScope
from placecell.consolidation import ChatSummarizer, Consolidator
from placecell.corrections import JsonlCorrectionLog, correction_now
from placecell.depth import DepthSnapshot
from placecell.errors import PlacecellError, ValidationError
from placecell.lifecycle import Curator, remove_local_file
from placecell.localization import LocalizationGate, LocalizationPolicy
from placecell.memory import Pose
from placecell.mission_context import MissionContext
from placecell.missions import MissionPlanner, PlanReviewAgent
from placecell.navigation import (
    DestinationResolver,
    NavigationCommands,
    NavigationPolicy,
    NavigationUpdate,
    load_named_places,
)
from placecell.object_arrival import ObjectArrivalPolicy, ObjectArrivalVerifier
from placecell.object_search import ObjectSearch, ObjectSearchPolicy
from placecell.objects import ObjectPolicy, ObjectRecall, ObjectTracker
from placecell.observer import Observer
from placecell.operator import navigation_payload as navigation_payload
from placecell.pipeline import Ingester, Observation, SegmentationPolicy, Segmenter
from placecell.providers import Captioner, EmbeddingProvider, HashingEmbedder
from placecell.recordings import RecordingWriter
from placecell.refinement import REFINEMENT_PROMPT, MemoryRefiner, RefinementPolicy
from placecell.retrieval import Recall
from placecell.ros2.bridge import (
    KeyframeWriter,
    ObservationBuilder,
    image_dimensions,
    pose_from_transform,
    stamp_to_seconds,
    update_localization,
)
from placecell.ros2.depth import PendingImages, aligned_snapshot
from placecell.ros2.navigation import Nav2Navigator, create_navigation_timers, create_navigator
from placecell.ros2.operator import OperatorInterface
from placecell.sensors import SensorHealth
from placecell.store import CollectionInfo, VectorStore
from placecell.tracing import TraceStore
from placecell.verification import VisionVerifier


def build_embedder(
    base_url: str,
    model: str,
    api_key: str | None,
    dimension: int,
    *,
    backend: str = "auto",
    device: str = "cpu",
    revision: str = "",
    local_files_only: bool = False,
    batch_size: int = 16,
    cache_folder: str = "",
) -> EmbeddingProvider:
    if backend == "openrouter":
        from placecell.providers.openrouter import OPENROUTER_BASE_URL, OpenRouterGeminiEmbedder

        return OpenRouterGeminiEmbedder(
            model or "google/gemini-embedding-2",
            api_key=api_key,
            dimension=dimension or 768,
            base_url=base_url or OPENROUTER_BASE_URL,
            batch_size=batch_size,
        )
    if backend == "gemini":
        from placecell.providers.gemini import DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, GeminiEmbedder

        return GeminiEmbedder(
            model or DEFAULT_GEMINI_MODEL,
            api_key=api_key,
            dimension=dimension or 768,
            base_url=base_url or GEMINI_BASE_URL,
            batch_size=batch_size,
        )
    if backend == "clip":
        from placecell.providers.clip import DEFAULT_CLIP_MODEL, ClipEmbedder

        embedder = ClipEmbedder(
            model or DEFAULT_CLIP_MODEL,
            device=device,
            revision=revision or None,
            local_files_only=local_files_only,
            batch_size=batch_size,
            cache_folder=cache_folder or None,
        )
        if dimension and dimension != embedder.dimension:
            raise ValidationError("embed_dimension does not match the CLIP checkpoint")
        return embedder
    if backend != "auto":
        raise ValidationError("embed_backend must be auto, gemini, openrouter or clip")
    if not model:
        return HashingEmbedder()
    from placecell.providers import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder(
        model, base_url or "https://api.openai.com/v1", api_key, dimension=dimension or None
    )


def build_store(db_path: str, collection: str, embedder: EmbeddingProvider) -> VectorStore:
    info = CollectionInfo(collection, embedder.model_name, embedder.dimension)
    if not db_path:
        from placecell.store import InMemoryStore

        return InMemoryStore(info)
    from placecell.store.lancedb_store import LanceDBStore

    return LanceDBStore(Path(db_path).expanduser(), info)


def build_mission_planner(parameters: dict[str, Any], api_key: str | None) -> MissionPlanner | None:
    if not parameters["mission_enabled"]:
        return None
    from placecell.providers import OpenAICompatibleChat
    from placecell.providers._http import RetryPolicy

    model = parameters["mission_model"]
    if not model:
        raise ValidationError("mission_enabled requires mission_model with tool calling")
    base_url = parameters["mission_base_url"] or parameters["chat_base_url"]
    options = {
        "api_key": api_key,
        "timeout_s": parameters["mission_request_timeout_s"],
        "max_tokens": 2048,
        "retry": RetryPolicy(attempts=1),
    }
    planner = OpenAICompatibleChat(model, base_url, **options)
    reviewer = OpenAICompatibleChat(
        parameters["mission_review_model"] or model,
        parameters["mission_review_base_url"] or base_url,
        **options,
    )
    return MissionPlanner(planner, PlanReviewAgent(reviewer), max_destinations=parameters["mission_max_destinations"])


def build_trace_store(parameters: dict[str, Any]) -> TraceStore | None:
    path = parameters["mission_trace_path"]
    if not path:
        return None
    return TraceStore(
        path,
        max_events=parameters["mission_trace_max_events"],
        max_bytes=parameters["mission_trace_max_bytes"],
        queue_size=parameters["mission_trace_queue_size"],
        secrets=[os.environ.get(value, "") for key, value in parameters.items() if key.endswith("api_key_env")],
    )


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
                    "similarity": r.similarity,
                    "image_similarity": r.image_similarity,
                    "caption_similarity": r.caption_similarity,
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


def create_node() -> Any:  # pragma: no cover - needs a ROS 2 environment
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.clock import Clock, ClockType, JumpThreshold
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    class PlacecellNode(Node):
        def __init__(self) -> None:
            super().__init__("placecell")
            p = self._params()
            if p["navigation_enabled"] and (not p["map_id"].strip() or not p["localization_required"]):
                raise ValidationError("Navigation requires a versioned map_id and localization_required:=true.")
            api_key = os.environ.get(p["api_key_env"]) or None
            embed_api_key = api_key
            if p["embed_api_key_env"]:
                embed_api_key = os.environ.get(p["embed_api_key_env"])
            elif p["embed_backend"] == "gemini":
                embed_api_key = os.environ.get("GEMINI_API_KEY") or api_key
            embedder = build_embedder(
                p["embed_base_url"],
                p["embed_model"],
                embed_api_key,
                p["embed_dimension"],
                backend=p["embed_backend"],
                device=p["embed_device"],
                revision=p["embed_revision"],
                local_files_only=p["embed_local_files_only"],
                batch_size=p["embed_batch_size"],
                cache_folder=p["embed_cache_folder"],
            )
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
            self._object_policy = ObjectPolicy(
                max_objects=p["object_max_records"],
                max_views=p["object_max_views"],
                retention_s=p["object_retention_s"],
                min_interval_s=p["object_min_interval_s"],
                require_position=bool(p["depth_topic"]),
            )
            self._object_recall: ObjectRecall | None = None
            tracker = None
            if p["object_backend"] not in {"gemini", "chat"}:
                raise ValidationError("object_backend must be gemini or chat")
            if p["objects_enabled"]:
                from placecell.providers.object_detection import ChatObjectDetector, GeminiObjectDetector

                detector_type = ChatObjectDetector if p["object_backend"] == "chat" else GeminiObjectDetector
                detector = detector_type(
                    p["object_model"],
                    api_key=os.environ.get(p["object_api_key_env"], ""),
                    base_url=p["object_base_url"],
                )
                tracker = ObjectTracker(store, embedder, detector, self._object_policy)
                self._object_recall = ObjectRecall(store, embedder, clock=self._memory_time)
            ingester = Ingester(
                embedder, store, captioner, segmenter, batch_size=p["batch_size"], observer=observer, objects=tracker
            )
            self._corrections = JsonlCorrectionLog(Path(p["corrections_path"]).expanduser())
            self._recall = Recall(store, embedder, corrections=self._corrections, clock=self._memory_time)
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
                self._agent = Agent(
                    self._recall, chat, frame_id=p["map_frame"], map_id=p["map_id"], clock=self._memory_time
                )
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
            self._curator = Curator(
                store, corrections=self._corrections, remover=remove_local_file, clock=self._memory_time
            )
            writer = KeyframeWriter(Path(p["keyframe_dir"]).expanduser())
            writer.recover_pending(store)
            store.drain_cleanup(remove_local_file)
            self._writer = writer
            self._builder = ObservationBuilder(p["robot_id"], p["camera_id"], writer)
            self._recording = RecordingWriter(p["recording_dir"]) if p["recording_dir"] else None
            self._map_frame, self._base_frame, self._map_id = p["map_frame"], p["base_frame"], p["map_id"]
            self._localization_required = p["localization_required"]
            self._sensors = SensorHealth(p["sensor_max_age_s"], clock=self._memory_time)
            self._clock_jump = self.get_clock().create_jump_callback(
                JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True),
                pre_callback=self._sensors.clock_changed.set,
            )
            self._localization = LocalizationGate(
                self._map_frame,
                self._map_id,
                LocalizationPolicy(
                    max_age_s=p["localization_max_age_s"],
                    max_position_std_m=p["localization_max_position_std_m"],
                    max_yaw_std_rad=p["localization_max_yaw_std_rad"],
                ),
                clock=self._memory_time,
            )
            self.create_subscription(
                PoseWithCovarianceStamped, p["localization_topic"], self._on_localization, qos_profile_sensor_data
            )
            self._tf_timeout = p["tf_timeout_s"]
            self._tf = Buffer()
            self._tf_listener = TransformListener(self._tf, self)
            self._depth_frames: deque[Any] = deque(maxlen=8)
            self._camera_infos: deque[Any] = deque(maxlen=8)
            self._depth_skew = p["object_depth_max_skew_s"]
            self._depth_error = p["object_position_error_m"]
            self._depth_angular_error = p["object_angular_error_rad"]
            self._pending_images = PendingImages(
                self._depth_skew, wait_s=p["rgbd_wait_s"], max_age_s=p["sensor_max_age_s"]
            )
            image_qos = (
                QoSProfile(depth=8, reliability=ReliabilityPolicy.RELIABLE)
                if p["rgbd_reliable"]
                else qos_profile_sensor_data
            )
            if p["objects_enabled"] and p["depth_topic"]:
                self._depth_frames = self._pending_images.depth
                self._camera_infos = self._pending_images.info
                self.create_subscription(
                    Image, p["depth_topic"], self._pending_images.add_depth, image_qos
                )
                self.create_subscription(
                    CameraInfo,
                    p["camera_info_topic"],
                    lambda msg: self._pending_images.add_depth(msg, calibration=True),
                    image_qos,
                )
            self._sync_images = p["objects_enabled"] and bool(p["depth_topic"])
            self._clock_fault_reported = False
            self.create_timer(0.04, self._drain_image, clock=Clock(clock_type=ClockType.STEADY_TIME))
            if p["compressed"]:
                self.create_subscription(
                    CompressedImage, p["image_topic"], self._receive_compressed, image_qos
                )
            else:
                self.create_subscription(Image, p["image_topic"], self._receive_image, image_qos)
            self.create_subscription(String, "~/ask", self._on_ask, 10)
            self.create_subscription(String, "~/correct", self._on_correct, 10)
            self.create_subscription(String, "~/refine", self._on_refine, 10)
            self._answers = self.create_publisher(String, "~/answer", 10)
            self._commands: NavigationCommands | None = None
            self._navigator: Nav2Navigator | None = None
            self._command_tasks: BoundedTasks | None = None
            self._mission_context: MissionContext | None = None
            self._mission_traces: TraceStore | None = None
            self._command_journal: CommandJournal | None = None
            if p["navigation_enabled"]:
                self._command_journal = CommandJournal(
                    p["command_journal_path"] or ":memory:",
                    CommandScope(p["robot_id"], p["map_id"], p["mission_conversation_id"]),
                    retry_window_s=p["command_retry_window_s"],
                    max_records=p["command_max_records"],
                )
                self._mission_traces = build_trace_store(p)
                if p["mission_enabled"]:
                    self._mission_context = MissionContext(
                        p["mission_context_path"] or ":memory:",
                        scope=json.dumps([p["robot_id"], p["map_id"], p["mission_conversation_id"]]),
                    )
                places = load_named_places(p["places_file"]) if p["places_file"] else {}
                verification_model = p["verification_model"] or p["caption_model"]
                verifier = (
                    VisionVerifier(
                        verification_model,
                        p["verification_base_url"] or p["caption_base_url"],
                        api_key,
                        timeout_s=p["verification_request_timeout_s"],
                    )
                    if verification_model
                    else None
                )
                object_arrival = None
                if tracker is not None:
                    from placecell.providers._http import RetryPolicy

                    arrival_detector = detector_type(
                        p["object_arrival_model"] or p["object_model"],
                        api_key=os.environ.get(p["object_api_key_env"], ""),
                        base_url=p["object_base_url"],
                        timeout_s=p["object_arrival_request_timeout_s"],
                        retry=RetryPolicy(attempts=1),
                    )
                    object_arrival = ObjectArrivalVerifier(
                        ObjectTracker(store, embedder, arrival_detector, self._object_policy),
                        arrival_detector,
                        ObjectArrivalPolicy(
                            max_observation_age_s=p["navigation_max_observation_age_s"],
                            min_similarity=p["object_arrival_min_similarity"],
                            moved_similarity=p["object_arrival_moved_similarity"],
                            similarity_margin=p["object_arrival_similarity_margin"],
                            max_uncertainty_m=p["object_arrival_max_uncertainty_m"],
                            max_position_age_s=p["object_arrival_max_position_age_s"],
                            max_move_m=p["object_arrival_max_move_m"],
                        ),
                        clock=self._memory_time,
                    )
                approach = None
                if p["approach_enabled"] or p["object_search_enabled"]:
                    from placecell.ros2.approach import create_planning_environment

                    if self._object_recall is None:
                        raise ValidationError("approach planning requires objects_enabled")

                    def current_pose() -> Pose | None:
                        stamp = self.get_clock().now().to_msg()
                        return self._pose_at(stamp.sec, stamp.nanosec)

                    environment = create_planning_environment(
                        self,
                        self._tf,
                        current_pose,
                        frame_id=self._map_frame,
                        map_id=self._map_id,
                        base_frame=self._base_frame,
                        costmap_topic=p["approach_costmap_topic"],
                        footprint_topic=p["approach_footprint_topic"],
                        action_name=p["approach_planner_action"],
                        planner_id=p["approach_planner_id"],
                        timeout_s=p["approach_request_timeout_s"],
                    )
                    approach = ApproachPlanner(
                        environment,
                        ApproachPolicy(
                            clearance_m=p["approach_clearance_m"],
                            max_uncertainty_m=p["approach_max_uncertainty_m"],
                            camera_yaw_offset_rad=p["approach_camera_yaw_offset_rad"],
                            max_sensor_age_s=p["approach_max_sensor_age_s"],
                            max_position_age_s=p["approach_max_position_age_s"],
                            planning_timeout_s=p["approach_planning_timeout_s"],
                        ),
                        clock=self._memory_time,
                    )
                resolver = DestinationResolver(
                    store,
                    self._recall,
                    robot_id=p["robot_id"],
                    camera_id=p["camera_id"],
                    frame_id=p["map_frame"],
                    map_id=p["map_id"],
                    clock=self._memory_time,
                    places=places,
                    verifier=verifier,
                    objects=self._object_recall,
                    approach=approach if p["approach_enabled"] else None,
                    object_arrival=object_arrival,
                    policy=NavigationPolicy(
                        min_similarity=p["navigation_min_similarity"],
                        min_confidence=p["navigation_min_confidence"],
                        max_age_s=p["navigation_max_memory_age_s"],
                    ),
                )
                self._navigator = create_navigator(
                    self, p["nav2_action"], p["navigation_response_timeout_s"], p["navigation_timeout_s"]
                )
                self._command_tasks = BoundedTasks(1, 1, self.get_logger())
                self._commands = NavigationCommands(
                    resolver,
                    self._navigator,
                    self._submit_command,
                    self._publish_navigation,
                    request_timeout_s=p["navigation_lookup_timeout_s"],
                    observation_clock=self._memory_time,
                    localization_ready=lambda: self._sensors.ready() and self._localization.ready(),
                    localization_generation=lambda: self._localization.generation,
                    sensor_ready=lambda d: self._sensors.ready(camera=d.source == "memory", depth=bool(d.object_id)),
                    sensor_generation=lambda d: (
                        self._sensors.generation(depth=bool(d.object_id)) if d.source == "memory" else 0
                    ),
                    arrival_timeout_s=p["navigation_arrival_timeout_s"],
                    max_observation_age_s=p["navigation_max_observation_age_s"],
                    arrival_max_attempts=p["navigation_arrival_max_attempts"],
                    mission_planner=build_mission_planner(p, api_key),
                    mission_context=self._mission_context,
                    trace_store=self._mission_traces,
                    search=ObjectSearch(
                        approach,
                        ObjectSearchPolicy(
                            max_viewpoints=p["object_search_max_viewpoints"],
                            timeout_s=p["object_search_timeout_s"],
                            radius_m=p["object_search_radius_m"],
                            max_path_m=p["object_search_max_path_m"],
                        ),
                    )
                    if p["object_search_enabled"] and approach is not None
                    else None,
                )
                create_navigation_timers(self, self._navigator, self._commands)
            self._operator = OperatorInterface(self, self._commands, journal=self._command_journal)
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
                "recording_dir": "",
                "compressed": False,
                "objects_enabled": False,
                "object_arrival_model": "",
                "object_arrival_request_timeout_s": 8.0,
                "object_arrival_min_similarity": 0.85,
                "object_arrival_moved_similarity": 0.95,
                "object_arrival_similarity_margin": 0.08,
                "object_arrival_max_uncertainty_m": 0.35,
                "object_arrival_max_position_age_s": 300.0,
                "object_arrival_max_move_m": 3.0,
                "object_search_enabled": False,
                "object_search_max_viewpoints": 3,
                "object_search_timeout_s": 60.0,
                "object_search_radius_m": 1.5,
                "object_search_max_path_m": 4.0,
                "approach_enabled": False,
                "approach_costmap_topic": "/global_costmap/costmap_raw",
                "approach_footprint_topic": "/local_costmap/published_footprint",
                "approach_planner_action": "/compute_path_to_pose",
                "approach_planner_id": "",
                "approach_request_timeout_s": 2.0,
                "approach_planning_timeout_s": 8.0,
                "approach_clearance_m": 0.5,
                "approach_max_uncertainty_m": 0.35,
                "approach_max_position_age_s": 300.0,
                "approach_max_sensor_age_s": 2.0,
                "approach_camera_yaw_offset_rad": 0.0,
                "object_model": "",
                "object_backend": "gemini",
                "object_base_url": "https://generativelanguage.googleapis.com/v1beta",
                "object_api_key_env": "GEMINI_API_KEY",
                "object_max_records": 1000,
                "object_max_views": 4,
                "object_retention_s": 2592000.0,
                "object_min_interval_s": 15.0,
                "depth_topic": "/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/camera/color/camera_info",
                "object_depth_max_skew_s": 0.08,
                "rgbd_wait_s": 0.3,
                "rgbd_reliable": False,
                "object_position_error_m": 0.1,
                "object_angular_error_rad": 0.05,
                "map_frame": "map",
                "base_frame": "base_footprint",
                "map_id": "",
                "localization_required": True,
                "localization_topic": "/amcl_pose",
                "localization_max_age_s": 5.0,
                "sensor_max_age_s": 5.0,
                "localization_max_position_std_m": 0.3,
                "localization_max_yaw_std_rad": 0.35,
                "db_path": "~/.placecell/db",
                "collection": "default",
                "keyframe_dir": "~/.placecell/keyframes",
                "embed_base_url": "",
                "embed_api_key_env": "",
                "embed_model": "",
                "embed_backend": "auto",
                "embed_device": "cpu",
                "embed_revision": "",
                "embed_local_files_only": False,
                "embed_batch_size": 16,
                "embed_cache_folder": "",
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
                "navigation_enabled": False,
                "mission_enabled": False,
                "mission_model": "",
                "mission_base_url": "",
                "mission_review_model": "",
                "mission_review_base_url": "",
                "mission_request_timeout_s": 8.0,
                "mission_max_destinations": 8,
                "mission_context_path": "~/.placecell/missions.sqlite3",
                "mission_trace_path": "",
                "mission_trace_max_events": 10000,
                "mission_trace_max_bytes": 16777216,
                "mission_trace_queue_size": 256,
                "mission_conversation_id": "default",
                "command_journal_path": "~/.placecell/commands.sqlite3",
                "command_retry_window_s": 86400.0,
                "command_max_records": 10000,
                "verification_model": "",
                "verification_base_url": "",
                "verification_request_timeout_s": 8.0,
                "navigation_arrival_timeout_s": 30.0,
                "navigation_max_observation_age_s": 5.0,
                "navigation_arrival_max_attempts": 3,
                "nav2_action": "navigate_to_pose",
                "places_file": "",
                "navigation_min_similarity": 0.5,
                "navigation_min_confidence": 0.2,
                "navigation_max_memory_age_s": 604800.0,
                "navigation_response_timeout_s": 10.0,
                "navigation_lookup_timeout_s": 30.0,
                "navigation_timeout_s": 600.0,
            }
            return {k: self.declare_parameter(k, v).value for k, v in defaults.items()}

        def _memory_time(self) -> float:
            return float(self.get_clock().now().nanoseconds) / 1e9

        def _on_localization(self, msg: Any) -> None:
            update_localization(self._localization, msg, self._map_id)

        def _pose_at(self, sec: int, nanosec: int) -> Pose | None:
            from rclpy.duration import Duration
            from rclpy.time import Time

            try:
                if stamp_to_seconds(sec, nanosec) <= 0 or not self._sensors.ready():
                    return None  # Time(0) asks TF for latest, not a capture-time transform.
                tf = self._tf.lookup_transform(
                    self._map_frame,
                    self._base_frame,
                    Time(seconds=sec, nanoseconds=nanosec),
                    Duration(seconds=self._tf_timeout),
                )
                t, q = tf.transform.translation, tf.transform.rotation
                pose = pose_from_transform(t.x, t.y, q.x, q.y, q.z, q.w, self._map_frame, self._map_id)
            except (TransformException, PlacecellError, ValueError) as e:
                self.get_logger().warning(f"no pose for image: {e}", throttle_duration_sec=5.0)
                return None
            if self._localization_required and not self._localization.accepts(pose, stamp_to_seconds(sec, nanosec)):
                self.get_logger().warning(
                    "skipping image: localization is missing, stale or uncertain", throttle_duration_sec=5.0
                )
                return None
            return pose

        def _depth_at(self, msg: Any, dimensions: tuple[int, int]) -> DepthSnapshot | None:
            from rclpy.duration import Duration
            from rclpy.time import Time

            if self._object_recall is None:
                return None
            stamp = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
            uncertainty = self._localization.uncertainty_at(stamp)
            if uncertainty is None:
                return None
            if not self._depth_frames or not self._camera_infos:
                self.get_logger().warning(
                    "object positions unavailable: waiting for aligned depth and CameraInfo", throttle_duration_sec=5.0
                )
                return None

            def skew(message: Any) -> float:
                return abs(stamp - stamp_to_seconds(message.header.stamp.sec, message.header.stamp.nanosec))

            try:
                depth = min((d for d in self._depth_frames if d.header.frame_id == msg.header.frame_id), key=skew)
                info = min(
                    (i for i in self._camera_infos if i.header.frame_id == msg.header.frame_id),
                    key=lambda i: 0 if stamp_to_seconds(i.header.stamp.sec, i.header.stamp.nanosec) == 0 else skew(i),
                )
                if dimensions != (info.width, info.height):
                    raise ValidationError("RGB and aligned depth dimensions differ")
                transform = self._tf.lookup_transform(
                    self._map_frame,
                    msg.header.frame_id,
                    Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec),
                    timeout=Duration(seconds=self._tf_timeout),
                )
                return aligned_snapshot(
                    depth,
                    info,
                    transform.transform,
                    rgb_stamp=stamp,
                    rgb_frame=msg.header.frame_id,
                    max_skew_s=self._depth_skew,
                    position_error_m=max(self._depth_error, 2 * uncertainty[0]),
                    angular_error_rad=max(self._depth_angular_error, 2 * uncertainty[1]),
                )
            except (TransformException, PlacecellError, ValueError) as e:
                self.get_logger().warning(f"object positions unavailable: {e}", throttle_duration_sec=5.0)
                return None

        def _capture(self, msg: Any, *, compressed: bool = False) -> tuple[Pose, float, DepthSnapshot | None] | None:
            stamp = 0.0
            try:
                stamp = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
                dimensions = image_dimensions(msg, compressed=compressed)
                pose = self._pose_at(msg.header.stamp.sec, msg.header.stamp.nanosec)
                if pose is None:
                    raise ValidationError("capture-time TF/localization is unavailable")
                depth = self._depth_at(msg, dimensions)
                if self._sensors.observe(stamp, camera=True, depth=depth is not None):
                    return pose, stamp, depth
            except (PlacecellError, ValueError, TypeError, AttributeError) as e:
                self._sensors.observe(stamp, camera=False, depth=False)
                self.get_logger().warning(f"skipping untrusted camera input: {e}", throttle_duration_sec=5.0)
            return None

        def _record(self, observation: Observation) -> None:
            if self._recording is not None:
                try:
                    self._recording.append(observation)
                except (OSError, PlacecellError) as e:
                    self._recording = None
                    self.get_logger().error(f"recording stopped after an export failure: {e}")

        def _on_image(self, msg: Any) -> None:
            capture = self._capture(msg)
            if capture is None:
                return
            pose, stamp, depth = capture
            force = self._commands is not None and self._commands.needs_observation
            if not force and not self._admission.eligible(self._robot_id, self._camera_id, stamp, pose):
                return
            if not force and not self._worker.has_capacity():
                self._worker.dropped += 1
                return
            try:
                obs = self._builder.from_raw(
                    stamp, msg.height, msg.width, msg.encoding, msg.step, bytes(msg.data), pose
                )
            except PlacecellError as e:
                self.get_logger().warning(f"skipped image: {e}", throttle_duration_sec=5.0)
                return
            obs = replace(
                obs, localization_checked=self._localization.accepts(pose, stamp), depth=depth, refresh_objects=force
            )
            if not self._sensors.ready():
                return
            self._record(obs)
            if self._commands is not None:
                self._commands.observe(obs)
            if self._worker.submit(obs):
                self._admission.accept(obs)
            self._writer.confirm(obs.evidence)

        def _receive_image(self, msg: Any) -> None:
            if self._sync_images:
                self._queue_image(msg, compressed=False)
            else:
                self._on_image(msg)

        def _receive_compressed(self, msg: Any) -> None:
            if self._sync_images:
                self._queue_image(msg, compressed=True)
            else:
                self._on_compressed(msg)

        def _queue_image(self, msg: Any, *, compressed: bool) -> None:
            if not self._sensors.ready():
                return
            if not self._pending_images.add(msg, compressed, time.monotonic(), source_now=self._memory_time()):
                try:
                    stamp = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
                except (AttributeError, TypeError, ValueError):
                    stamp = float("nan")
                self._sensors.observe(stamp, camera=False, depth=False)

        def _drain_image(self) -> None:
            if not self._sensors.ready():
                self._pending_images.clear()
                if not self._clock_fault_reported:
                    self._clock_fault_reported = True
                    self.get_logger().error(
                        "Clock changed or reset: navigation and new captures are blocked. "
                        "Confirm Nav2 is stopped, then restart with a fresh collection and keyframe directory."
                    )
                return
            ready = self._pending_images.pop(time.monotonic())
            if ready is not None:
                message, compressed = ready
                (self._on_compressed if compressed else self._on_image)(message)

        def _on_compressed(self, msg: Any) -> None:
            capture = self._capture(msg, compressed=True)
            if capture is None:
                return
            pose, stamp, depth = capture
            force = self._commands is not None and self._commands.needs_observation
            if not force and not self._admission.eligible(self._robot_id, self._camera_id, stamp, pose):
                return
            if not force and not self._worker.has_capacity():
                self._worker.dropped += 1
                return
            try:
                obs = self._builder.from_compressed(stamp, msg.format, bytes(msg.data), pose)
            except PlacecellError as e:
                self.get_logger().warning(f"skipped image: {e}", throttle_duration_sec=5.0)
                return
            obs = replace(
                obs, localization_checked=self._localization.accepts(pose, stamp), depth=depth, refresh_objects=force
            )
            if not self._sensors.ready():
                return
            self._record(obs)
            if self._commands is not None:
                self._commands.observe(obs)
            if self._worker.submit(obs):
                self._admission.accept(obs)
            self._writer.confirm(obs.evidence)

        def _on_ask(self, msg: Any) -> None:
            if not self._questions.submit(self._answer, msg.data):
                self._answers.publish(String(data=json.dumps({"question": msg.data, "error": "question queue full"})))

        def _submit_command(self, function: Callable[[], None]) -> bool:
            return self._command_tasks is not None and self._command_tasks.submit(function)

        def _publish_navigation(self, update: NavigationUpdate) -> None:
            self._operator.publish(update)

        def stop_navigation(self) -> None:
            if self._commands is not None:
                self._commands.close()

        def navigation_busy(self) -> bool:
            return self._commands is not None and self._commands.busy

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
            self._store.objects.prune(self._memory_time() - self._object_policy.retention_s)
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
            self.get_logger().info(
                f"ingestion: {stats}, dropped={self._worker.dropped}, objects={self._store.objects.count()}"
            )
            if self._mission_traces is not None:
                health = self._mission_traces.health()
                self.get_logger().info(f"mission traces: {health}")
                if health["dropped_events"] or health["write_errors"] or not health["writer_alive"]:
                    self.get_logger().warning("Mission trace capture is incomplete; inspect trace health counters.")

        def destroy_node(self) -> bool:
            self._clock_jump.unregister()
            self.stop_navigation()
            commands_done = self._command_tasks is None or self._command_tasks.stop()
            ingested = self._worker.stop()
            answered = self._questions.stop()
            maintained = self._maintenance.stop()
            if ingested and answered and maintained and commands_done:
                self._store.close()
                if self._mission_context is not None:
                    self._mission_context.close()
            if self._mission_traces is not None and not self._mission_traces.close():
                self.get_logger().warning("Mission trace writer did not finish before the shutdown deadline.")
            if self._command_journal is not None:
                self._command_journal.close()
            return bool(super().destroy_node())

    return PlacecellNode()


def main(args: list[str] | None = None) -> None:  # pragma: no cover - needs a ROS 2 environment
    import signal

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    # Own the signals: rclpy's handler tears the context down from inside the signal handler,
    # which races with the executor's wait set. A flag lets the loop finish its iteration instead.
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = create_node()
    # Leave room for commands, action results and steady deadlines while camera/TF work
    # is active. Commands and deadline timers have separate callback groups.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
    finally:
        node.stop_navigation()
        deadline = time.monotonic() + 2.0
        while node.navigation_busy() and rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if node.navigation_busy():
            node.get_logger().warning("Shutdown could not confirm navigation cancellation; check Nav2 status.")
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
