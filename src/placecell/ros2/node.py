"""The rclpy node. Subscribes to a camera, looks up the robot pose in TF, ingests in the
background, and answers questions published on `~/ask` with JSON on `~/answer`.

Run with `placecell-ros2` inside a sourced ROS 2 environment, or `ros2 run` once packaged.
Configuration is plain ROS parameters; API keys come from the environment variables named
by `api_key_env` and the per-endpoint `*_api_key_env` parameters, never from a parameter, so
they do not end up in launch files or logs. The shared `api_key_env` key goes only to the
scheme, host and port of `chat_base_url`; see `endpoint`.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable
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
from placecell.lifecycle import Curator, RetentionPolicy, remove_local_file
from placecell.localization import LocalizationGate, LocalizationPolicy
from placecell.maintenance import StorageLease
from placecell.memory import Pose
from placecell.mission_context import MissionContext
from placecell.navigation import (
    DestinationResolver,
    NavigationCommands,
    NavigationPolicy,
    NavigationUpdate,
    load_named_places,
)
from placecell.navigation_ownership import NavigationOwnership, NavigationScope
from placecell.object_arrival import ObjectArrivalPolicy, ObjectArrivalVerifier
from placecell.object_search import ObjectSearch, ObjectSearchPolicy
from placecell.objects import ObjectPolicy, ObjectRecall, ObjectTracker
from placecell.observer import Observer
from placecell.operator import navigation_payload as navigation_payload
from placecell.pipeline import Ingester, Observation, SegmentationPolicy, Segmenter
from placecell.providers import Captioner
from placecell.providers._http import RetryPolicy
from placecell.recordings import RecordingWriter
from placecell.refinement import REFINEMENT_PROMPT, MemoryRefiner, RefinementPolicy
from placecell.retrieval import Recall
from placecell.ros2.answers import ANSWER_SCHEMA_VERSION as ANSWER_SCHEMA_VERSION
from placecell.ros2.answers import NO_CONFIDENT_ANSWER as NO_CONFIDENT_ANSWER
from placecell.ros2.answers import answer_error as answer_error
from placecell.ros2.answers import answer_payload as answer_payload
from placecell.ros2.answers import answer_question as answer_question
from placecell.ros2.bridge import (
    KeyframeWriter,
    ObservationBuilder,
    image_dimensions,
    pose_from_transform,
    stamp_to_seconds,
    update_localization,
    update_odometry,
)
from placecell.ros2.components import ENDPOINTS as ENDPOINTS
from placecell.ros2.components import build_embedder as build_embedder
from placecell.ros2.components import build_mission_planner as build_mission_planner
from placecell.ros2.components import build_store as build_store
from placecell.ros2.components import build_trace_store as build_trace_store
from placecell.ros2.components import chat_options as chat_options
from placecell.ros2.components import embedding_api_key as embedding_api_key
from placecell.ros2.components import endpoint as endpoint
from placecell.ros2.components import shared_api_key as shared_api_key
from placecell.ros2.config import declare
from placecell.ros2.depth import PendingImages, aligned_snapshot
from placecell.ros2.navigation import Nav2Navigator, create_navigation_timers, create_navigator
from placecell.ros2.operator import OperatorInterface
from placecell.ros2.workers import BoundedTasks as BoundedTasks
from placecell.ros2.workers import IngestWorker as IngestWorker
from placecell.sensors import SensorHealth
from placecell.store.base import EVERYTHING
from placecell.store.limits import StoreLimits
from placecell.tracing import TraceStore
from placecell.verification import VisionVerifier

INDEX_SYNC_INTERVAL_S = 2.0
"""How often changed memories are copied into a vector index, so searches score few of them exactly."""


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
            config = declare(self)
            p = config.parameters()
            # Acquire before any store, keyframe, context or trace writer is opened.
            self._storage_lease = StorageLease.for_parameters(p)
            if config.navigation.enabled and (
                not config.localization.map_id.strip() or not config.localization.required
            ):
                raise ValidationError("Navigation requires a versioned map_id and localization_required:=true.")
            embedder = build_embedder(
                config.embedding.base_url,
                config.embedding.model,
                embedding_api_key(p),
                config.embedding.dimension,
                backend=config.embedding.backend,
                device=config.embedding.device,
                revision=config.embedding.revision,
                local_files_only=config.embedding.local_files_only,
                batch_size=config.embedding.batch_size,
                cache_folder=config.embedding.cache_folder,
            )
            store = build_store(
                config.storage.db_path,
                config.storage.collection,
                embedder,
                limits=StoreLimits(
                    config.storage.memory_max_records,
                    config.storage.memory_max_sightings,
                    config.storage.refine_max_pending,
                    config.storage.cleanup_max_pending,
                    config.storage.memory_evict_at_capacity,
                ),
            )
            captioner: Captioner | None = None
            caption_url, caption_key = endpoint(p, "caption")
            if config.caption.model:
                from placecell.providers import OpenAICompatibleCaptioner

                captioner = OpenAICompatibleCaptioner(
                    config.caption.model,
                    caption_url,
                    caption_key,
                    max_tokens=config.caption.max_tokens,
                    retry=RetryPolicy(attempts=1),
                )
            if captioner is None and not embedder.capabilities.image:
                self.get_logger().warning(
                    "no caption_model and the embedder takes text only: frames cannot be stored. "
                    "Set caption_model, or use an embedding model that accepts images."
                )
            policy = SegmentationPolicy(
                config.ingest.min_interval_s,
                config.ingest.min_travel_m,
                config.ingest.min_turn_rad,
                config.ingest.max_interval_s,
            )
            segmenter = Segmenter(policy)
            self._admission = Segmenter(policy)
            self._robot_id, self._camera_id = config.camera.robot_id, config.camera.camera_id
            self._store = store
            observer = Observer(store) if config.ingest.contradiction else None
            self._object_policy = ObjectPolicy(
                max_objects=config.objects.max_records,
                max_views=config.objects.max_views,
                retention_s=config.objects.retention_s,
                min_interval_s=config.objects.min_interval_s,
                require_position=bool(config.camera.depth_topic),
            )
            self._object_recall: ObjectRecall | None = None
            tracker = None
            if config.objects.backend not in {"gemini", "chat"}:
                raise ValidationError("object_backend must be gemini or chat")
            if config.objects.enabled:
                from placecell.providers.object_detection import ChatObjectDetector, GeminiObjectDetector

                detector_type = ChatObjectDetector if config.objects.backend == "chat" else GeminiObjectDetector
                detector = detector_type(
                    config.objects.model,
                    api_key=os.environ.get(config.objects.api_key_env, ""),
                    base_url=config.objects.base_url,
                    retry=RetryPolicy(attempts=1),
                )
                tracker = ObjectTracker(store, embedder, detector, self._object_policy)
                self._object_recall = ObjectRecall(store, embedder, clock=self._memory_time)
            ingester = Ingester(
                embedder,
                store,
                captioner,
                segmenter,
                batch_size=config.ingest.batch_size,
                observer=observer,
                objects=tracker,
            )
            self._corrections = JsonlCorrectionLog(
                Path(config.storage.corrections_path).expanduser(),
                max_records=config.storage.correction_max_records,
                max_bytes=config.storage.correction_max_bytes,
            )
            self._recall = Recall(store, embedder, corrections=self._corrections, clock=self._memory_time)
            self._agent: Agent | None = None
            self._answer_min_similarity = float(config.questions.answer_min_similarity)
            if not 0 < self._answer_min_similarity <= 1:
                raise ValidationError("answer_min_similarity must be within (0, 1]")
            self._consolidator: Consolidator | None = None
            self._refiner: MemoryRefiner | None = None
            refinement_model = config.maintenance.refine_model or config.caption.model
            if config.maintenance.refine_interval_s > 0 and refinement_model:
                from placecell.providers import OpenAICompatibleCaptioner

                reviewer = OpenAICompatibleCaptioner(
                    refinement_model,
                    caption_url,
                    caption_key,
                    prompt=REFINEMENT_PROMPT,
                    max_tokens=config.caption.max_tokens,
                    detail="high",
                )
                self._refiner = MemoryRefiner(
                    store,
                    embedder,
                    reviewer,
                    RefinementPolicy(max_memories=config.maintenance.refine_batch_size),
                    producer=refinement_model,
                )
            if config.chat.model:
                from placecell.providers import OpenAICompatibleChat

                chat = OpenAICompatibleChat(config.chat.model, *endpoint(p, "chat"), **chat_options(p, "chat"))
                self._agent = Agent(
                    self._recall,
                    chat,
                    frame_id=config.localization.map_frame,
                    map_id=config.localization.map_id,
                    clock=self._memory_time,
                    max_tool_calls=config.chat.max_tool_calls,
                    max_context_chars=config.chat.max_context_chars,
                )
                if config.maintenance.consolidate_interval_s > 0:
                    self._consolidator = Consolidator(store, embedder, ChatSummarizer(chat))
            self._worker = IngestWorker(
                ingester,
                None,
                config.ingest.batch_size,
                config.ingest.max_queue,
                self.get_logger(),
                max_attempts=config.ingest.attempts,
                retry_delay_s=config.ingest.retry_delay_s,
            )
            self._questions = BoundedTasks(config.questions.workers, config.questions.queue, self.get_logger())
            self._maintenance = BoundedTasks(1, 1, self.get_logger())
            # A separate worker, so frequent index syncs never crowd out hourly maintenance.
            self._indexing = BoundedTasks(1, 1, self.get_logger())
            self._curator = Curator(
                store,
                RetentionPolicy(
                    max_idle_s=config.storage.memory_max_idle_s, history_age_s=config.storage.memory_history_age_s
                ),
                corrections=self._corrections,
                remover=remove_local_file,
                clock=self._memory_time,
            )
            writer = KeyframeWriter(Path(config.storage.keyframe_dir).expanduser())
            writer.recover_pending(store)
            store.drain_cleanup(remove_local_file)
            self._writer = writer
            self._builder = ObservationBuilder(config.camera.robot_id, config.camera.camera_id, writer)
            self._recording = RecordingWriter(config.camera.recording_dir) if config.camera.recording_dir else None
            self._map_frame, self._base_frame, self._map_id = (
                config.localization.map_frame,
                config.localization.base_frame,
                config.localization.map_id,
            )
            self._localization_required = config.localization.required
            self._sensors = SensorHealth(
                config.sensors.max_age_s,
                max_future_s=config.sensors.max_future_s,
                max_failures=config.sensors.max_failures,
                clock=self._memory_time,
            )
            self._clock_jump = self.get_clock().create_jump_callback(
                JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True),
                pre_callback=self._sensors.clock_changed.set,
            )
            self._localization = LocalizationGate(
                self._map_frame,
                self._map_id,
                LocalizationPolicy(
                    max_age_s=config.localization.max_age_s,
                    max_position_std_m=config.localization.max_position_std_m,
                    max_yaw_std_rad=config.localization.max_yaw_std_rad,
                    max_capture_future_s=config.sensors.max_future_s,
                    stationary_translation_m=config.localization.stationary_translation_m,
                    stationary_rotation_rad=config.localization.stationary_rotation_rad,
                    max_stationary_age_s=config.localization.max_stationary_age_s or math.inf,
                ),
                clock=self._memory_time,
            )
            self.create_subscription(
                PoseWithCovarianceStamped, config.localization.topic, self._on_localization, qos_profile_sensor_data
            )
            self._tf_timeout = config.localization.tf_timeout_s
            self._tf = Buffer()
            self._tf_listener = TransformListener(self._tf, self)
            self._odom_frame = config.localization.odom_frame
            if self._odom_frame:
                self.create_timer(0.2, self._on_odometry, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self._depth_frames: deque[Any] = deque(maxlen=8)
            self._camera_infos: deque[Any] = deque(maxlen=8)
            self._depth_skew = config.objects.depth_max_skew_s
            self._depth_error = config.objects.position_error_m
            self._depth_angular_error = config.objects.angular_error_rad
            self._pending_images = PendingImages(
                self._depth_skew,
                wait_s=config.camera.rgbd_wait_s,
                max_age_s=config.sensors.max_age_s,
                max_message_bytes=config.camera.max_message_bytes,
                max_future_s=config.sensors.max_future_s,
            )
            image_qos = (
                QoSProfile(depth=8, reliability=ReliabilityPolicy.RELIABLE)
                if config.camera.rgbd_reliable
                else qos_profile_sensor_data
            )
            if config.objects.enabled and config.camera.depth_topic:
                self._depth_frames = self._pending_images.depth
                self._camera_infos = self._pending_images.info
                self.create_subscription(Image, config.camera.depth_topic, self._pending_images.add_depth, image_qos)
                self.create_subscription(
                    CameraInfo,
                    config.camera.info_topic,
                    lambda msg: self._pending_images.add_depth(msg, calibration=True),
                    image_qos,
                )
            self._sync_images = config.objects.enabled and bool(config.camera.depth_topic)
            self._clock_fault_reported = False
            self.create_timer(0.04, self._drain_image, clock=Clock(clock_type=ClockType.STEADY_TIME))
            if config.camera.compressed:
                self.create_subscription(
                    CompressedImage, config.camera.image_topic, self._receive_compressed, image_qos
                )
            else:
                self.create_subscription(Image, config.camera.image_topic, self._receive_image, image_qos)
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
            if config.navigation.enabled:
                self._command_journal = CommandJournal(
                    config.navigation.command_journal_path or ":memory:",
                    CommandScope(config.camera.robot_id, config.localization.map_id, config.mission.conversation_id),
                    retry_window_s=config.navigation.command_retry_window_s,
                    max_records=config.navigation.command_max_records,
                )
                self._mission_traces = build_trace_store(p)
                if config.mission.enabled:
                    self._mission_context = MissionContext(
                        config.mission.context_path or ":memory:",
                        scope=json.dumps(
                            [config.camera.robot_id, config.localization.map_id, config.mission.conversation_id]
                        ),
                        max_events=config.mission.context_max_events,
                        max_bytes=config.mission.context_max_bytes,
                        retention_s=config.mission.context_retention_s,
                        references_available=self._context_reference_available,
                    )
                places = load_named_places(config.navigation.places_file) if config.navigation.places_file else {}
                verification_model = config.verification.model or config.caption.model
                verifier = (
                    VisionVerifier(
                        verification_model,
                        *endpoint(p, "verification"),
                        timeout_s=config.verification.request_timeout_s,
                        max_tokens=config.verification.max_tokens,
                        structured_output=config.verification.structured_output,
                    )
                    if verification_model
                    else None
                )
                object_arrival = None
                if tracker is not None:
                    arrival_detector = detector_type(
                        config.object_arrival.model or config.objects.model,
                        api_key=os.environ.get(config.objects.api_key_env, ""),
                        base_url=config.objects.base_url,
                        timeout_s=config.object_arrival.request_timeout_s,
                        retry=RetryPolicy(attempts=1),
                    )
                    object_arrival = ObjectArrivalVerifier(
                        ObjectTracker(store, embedder, arrival_detector, self._object_policy),
                        arrival_detector,
                        ObjectArrivalPolicy(
                            max_observation_age_s=config.navigation.max_observation_age_s,
                            min_similarity=config.object_arrival.min_similarity,
                            moved_similarity=config.object_arrival.moved_similarity,
                            similarity_margin=config.object_arrival.similarity_margin,
                            max_uncertainty_m=config.object_arrival.max_uncertainty_m,
                            max_position_age_s=config.object_arrival.max_position_age_s,
                            max_move_m=config.object_arrival.max_move_m,
                        ),
                        clock=self._memory_time,
                    )
                approach = None
                if config.approach.enabled or config.object_search.enabled:
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
                        costmap_topic=config.approach.costmap_topic,
                        footprint_topic=config.approach.footprint_topic,
                        action_name=config.approach.planner_action,
                        planner_id=config.approach.planner_id,
                        timeout_s=config.approach.request_timeout_s,
                    )
                    approach = ApproachPlanner(
                        environment,
                        ApproachPolicy(
                            clearance_m=config.approach.clearance_m,
                            max_uncertainty_m=config.approach.max_uncertainty_m,
                            camera_yaw_offset_rad=config.approach.camera_yaw_offset_rad,
                            max_sensor_age_s=config.approach.max_sensor_age_s,
                            max_position_age_s=config.approach.max_position_age_s,
                            planning_timeout_s=config.approach.planning_timeout_s,
                        ),
                        clock=self._memory_time,
                    )
                resolver = DestinationResolver(
                    store,
                    self._recall,
                    robot_id=config.camera.robot_id,
                    camera_id=config.camera.camera_id,
                    frame_id=config.localization.map_frame,
                    map_id=config.localization.map_id,
                    clock=self._memory_time,
                    places=places,
                    verifier=verifier,
                    objects=self._object_recall,
                    approach=approach if config.approach.enabled else None,
                    object_arrival=object_arrival,
                    policy=NavigationPolicy(
                        min_similarity=config.navigation.min_similarity,
                        min_confidence=config.navigation.min_confidence,
                        max_age_s=config.navigation.max_memory_age_s,
                    ),
                )
                self._navigator = create_navigator(
                    self,
                    config.navigation.nav2_action,
                    config.navigation.response_timeout_s,
                    config.navigation.timeout_s,
                    ownership=NavigationOwnership(
                        config.navigation.ownership_path,
                        NavigationScope(
                            self._robot_id, self._map_id, self.resolve_topic_name(config.navigation.nav2_action)
                        ),
                    ),
                )
                self._command_tasks = BoundedTasks(1, 1, self.get_logger())
                navigator = self._navigator
                self._commands = NavigationCommands(
                    resolver,
                    self._navigator,
                    self._submit_command,
                    self._publish_navigation,
                    request_timeout_s=config.navigation.lookup_timeout_s,
                    observation_clock=self._memory_time,
                    localization_ready=lambda: self._sensors.ready() and self._localization.ready(),
                    localization_generation=lambda: self._localization.generation,
                    sensor_ready=lambda d: self._sensors.ready(camera=d.source == "memory", depth=bool(d.object_id)),
                    sensor_generation=lambda d: (
                        self._sensors.generation(depth=bool(d.object_id)) if d.source == "memory" else 0
                    ),
                    arrival_timeout_s=config.navigation.arrival_timeout_s,
                    max_observation_age_s=config.navigation.max_observation_age_s,
                    arrival_max_attempts=config.navigation.arrival_max_attempts,
                    mission_planner=build_mission_planner(p),
                    mission_context=self._mission_context,
                    trace_store=self._mission_traces,
                    startup_block_reason=lambda: navigator.startup_block_reason,
                    search=ObjectSearch(
                        approach,
                        ObjectSearchPolicy(
                            max_viewpoints=config.object_search.max_viewpoints,
                            timeout_s=config.object_search.timeout_s,
                            radius_m=config.object_search.radius_m,
                            max_path_m=config.object_search.max_path_m,
                        ),
                    )
                    if config.object_search.enabled and approach is not None
                    else None,
                )
                create_navigation_timers(self, self._navigator, self._commands)
            self._operator = OperatorInterface(self, self._commands, journal=self._command_journal)
            if config.maintenance.curator_interval_s > 0:
                self.create_timer(config.maintenance.curator_interval_s, self._curate)
            if self._consolidator is not None:
                self.create_timer(config.maintenance.consolidate_interval_s, self._consolidate)
            if self._refiner is not None:
                self.create_timer(config.maintenance.refine_interval_s, self._refine)
            sync_index = getattr(self._store, "sync_index", None)
            if sync_index is not None:
                self.create_timer(INDEX_SYNC_INTERVAL_S, lambda: self._indexing.submit(sync_index, key="sync"))
            self.create_timer(30.0, self._diagnostics)
            self._worker.start()
            where = f"lancedb {config.storage.db_path}" if config.storage.db_path else "memory"
            self.get_logger().info(
                f"placecell up: robot {config.camera.robot_id}, camera {config.camera.camera_id}, "
                f"model {embedder.model_name}, "
                f"store {where}, agent {'on' if self._agent else 'off'}"
            )

        def _memory_time(self) -> float:
            return float(self.get_clock().now().nanoseconds) / 1e9

        def _on_localization(self, msg: Any) -> None:
            update_localization(self._localization, msg, self._map_id)

        def _on_odometry(self) -> None:
            from rclpy.time import Time

            try:
                transform = self._tf.lookup_transform(self._odom_frame, self._base_frame, Time())
            except TransformException as e:
                self.get_logger().warning(
                    f"no odometry: idle localization expires after localization_max_age_s: {e}",
                    throttle_duration_sec=60.0,
                )
                return
            update_odometry(self._localization, transform)

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
            if not self._pending_images.accepts_size(msg):
                return
            capture = self._capture(msg)
            if capture is None:
                return
            pose, stamp, depth = capture
            force = self._commands is not None and self._commands.needs_observation
            if not force and not self._admission.eligible(self._robot_id, self._camera_id, stamp, pose):
                return
            if not force and not self._worker.has_capacity():
                self._worker.reject()
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
            if not self._pending_images.accepts_size(msg):
                return
            capture = self._capture(msg, compressed=True)
            if capture is None:
                return
            pose, stamp, depth = capture
            force = self._commands is not None and self._commands.needs_observation
            if not force and not self._admission.eligible(self._robot_id, self._camera_id, stamp, pose):
                return
            if not force and not self._worker.has_capacity():
                self._worker.reject()
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
            if len(msg.data) > 2000 or not msg.data.strip():
                self._answers.publish(
                    String(data=answer_error(msg.data[:128], "question must contain 1..2000 characters"))
                )
                return
            if not self._questions.submit(self._answer, msg.data):
                self._answers.publish(String(data=answer_error(msg.data, "question queue full")))

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
            payload = answer_question(question, self._agent, self._recall, self._answer_min_similarity)
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
                if self._store.get(correction.memory_id) is None:
                    raise ValidationError("correction refers to an unavailable scene memory")
                self._corrections.record(correction)
                if correction.verdict == "wrong":
                    accepted = self._store.refinements.request(correction.memory_id, "operator correction")
                    if not accepted:
                        self.get_logger().warning("Correction saved; recheck queue full or memory is ineligible.")
            except (ValueError, KeyError, TypeError, OSError, PlacecellError) as e:
                self.get_logger().warning(f"ignored correction: {e}")
                return

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
            self._maintenance.submit(self._run_refiner, key="refine")

        def _run_refiner(self) -> None:
            if self._refiner is not None:
                report = self._refiner.run()
                if report.attempted:
                    self.get_logger().info(f"memory refinement: {report}")

        def _curate(self) -> None:
            self._maintenance.submit(self._run_curator, key="curate")

        def _context_reference_available(self, data: dict[str, Any]) -> bool:
            if data.get("object_id"):
                record = self._store.objects.get(data["object_id"])
                return record is not None and record.status == "present"
            if data.get("memory_id"):
                memory = self._store.get(data["memory_id"])
                return memory is not None and not memory.superseded
            return True

        def _run_curator(self) -> None:
            self._store.drain_cleanup(remove_local_file)
            before = self._memory_time() - self._object_policy.retention_s
            for _ in range(128):
                if not self._store.objects.prune(before, limit=1):
                    break
                self._store.drain_cleanup(remove_local_file)
            report = self._curator.run()
            self._corrections.prune(m.id for batch in self._store.iter_query(EVERYTHING) for m in batch)
            if self._mission_context is not None:
                self._mission_context.prune()
            maintain = getattr(self._store, "maintain", None)
            if maintain is not None:
                maintain()
            if report.removed or report.discredited or report.history_pruned:
                self.get_logger().info(
                    f"curator removed {report.removed} memories, discredited {report.discredited}, "
                    f"pruned {report.history_pruned} sightings"
                )

        def _consolidate(self) -> None:
            self._maintenance.submit(self._run_consolidator, key="consolidate")

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
            stats = self._worker.health()
            self.get_logger().info(
                f"ingestion: {stats}, dropped={self._worker.dropped}, objects={self._store.objects.count()}"
            )
            self.get_logger().info(
                f"queues: questions={self._questions.health()}, maintenance={self._maintenance.health()}, "
                f"commands={self._command_tasks.health() if self._command_tasks is not None else None}, "
                f"images={self._pending_images.health()}, sensors={self._sensors.health()}"
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
            maintained = self._indexing.stop() and maintained
            if ingested and answered and maintained and commands_done:
                self._store.close()
                if self._mission_context is not None:
                    self._mission_context.close()
            traced = self._mission_traces is None or self._mission_traces.close()
            if not traced:
                self.get_logger().warning("Mission trace writer did not finish before the shutdown deadline.")
            if self._command_journal is not None:
                self._command_journal.close()
            if self._navigator is not None:
                self._navigator.close()
            if ingested and answered and maintained and commands_done and traced:
                self._storage_lease.close()
            # A stuck writer retains the lease until process exit.
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
