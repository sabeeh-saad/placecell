"""The rclpy node: camera, TF and localization in; answers, navigation status and snapshots out.

Importing this module needs ROS 2, so `placecell.ros2.node.create_node` imports it lazily.
The class only wires ROS topics, timers and TF to the parts that are built and tested
without ROS: `components` builds them, `capture` handles camera frames and `housekeeping`
runs corrections, maintenance and diagnostics.
"""

from __future__ import annotations

import functools
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.clock import Clock, ClockType, JumpThreshold
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from placecell.command_identity import CommandJournal
from placecell.errors import PlacecellError
from placecell.maintenance import StorageLease
from placecell.memory import Pose
from placecell.mission_context import MissionContext
from placecell.navigation import NavigationCommands, NavigationUpdate
from placecell.ros2.answers import answer_error, answer_question
from placecell.ros2.bridge import update_localization, update_odometry
from placecell.ros2.capture import FrameCapture
from placecell.ros2.components import (
    RosPorts,
    build_components,
    build_localization,
    build_navigation,
    build_pending_images,
)
from placecell.ros2.config import declare
from placecell.ros2.housekeeping import Housekeeping
from placecell.ros2.navigation import Nav2Navigator, create_navigation_timers, create_navigator
from placecell.ros2.operator import OperatorInterface
from placecell.ros2.workers import BoundedTasks
from placecell.tracing import TraceStore

INDEX_SYNC_INTERVAL_S = 2.0
"""How often changed memories are copied into a vector index, so searches score few of them exactly."""


class PlacecellNode(Node):
    def __init__(self) -> None:
        super().__init__("placecell")
        # Construction registers the release of whatever it acquires. After a failure everything is
        # released in reverse order, so a retry in this process can start; destroy_node does it otherwise.
        acquired = ExitStack()
        acquired.callback(super().destroy_node)
        try:
            self._construct(acquired)
        except BaseException:
            try:
                acquired.close()
            except Exception as e:
                self.get_logger().error(f"releasing a failed start did not finish: {e}")
            raise

    def _construct(self, acquired: ExitStack) -> None:
        config = declare(self)
        p = config.parameters()
        # Acquire before any store, keyframe, context or trace writer is opened.
        self._storage_lease = StorageLease.for_parameters(p)
        acquired.callback(self._storage_lease.close)
        parts = build_components(config, clock=self._memory_time, log=self.get_logger(), resources=acquired)
        self._store, self._object_policy = parts.store, parts.object_policy
        self._corrections, self._recall = parts.corrections, parts.recall
        self._agent, self._answer_min_similarity = parts.agent, parts.answer_min_similarity
        self._consolidator, self._refiner = parts.consolidator, parts.refiner
        self._worker, self._questions = parts.worker, parts.questions
        self._maintenance, self._indexing = parts.maintenance, parts.indexing
        self._base_frame, self._map_id = config.localization.base_frame, config.localization.map_id
        self._sensors = parts.sensors
        self._clock_jump = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True),
            pre_callback=self._sensors.clock_changed.set,
        )
        acquired.callback(self._clock_jump.unregister)
        self._localization = build_localization(config, clock=self._memory_time)
        self.create_subscription(
            PoseWithCovarianceStamped, config.localization.topic, self._on_localization, qos_profile_sensor_data
        )
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._odom_frame = config.localization.odom_frame
        if self._odom_frame:
            self.create_timer(0.2, self._on_odometry, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self._depth_frames: deque[Any] = deque(maxlen=8)
        self._camera_infos: deque[Any] = deque(maxlen=8)
        self._pending_images = build_pending_images(config)
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
        self.create_timer(0.04, self._drain_image, clock=Clock(clock_type=ClockType.STEADY_TIME))
        if config.camera.compressed:
            self.create_subscription(CompressedImage, config.camera.image_topic, self._receive_compressed, image_qos)
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
            navigation = build_navigation(
                config,
                parts,
                self._localization,
                clock=self._memory_time,
                log=self.get_logger(),
                ros=RosPorts(
                    create_navigator=functools.partial(create_navigator, self),
                    create_planning_environment=self._create_planning_environment,
                    resolve_topic_name=self.resolve_topic_name,
                ),
                submit=self._submit_command,
                publish=self._publish_navigation,
                references_available=self._context_reference_available,
                resources=acquired,
            )
            self._command_journal, self._mission_traces = navigation.journal, navigation.traces
            self._mission_context, self._navigator = navigation.context, navigation.navigator
            self._command_tasks, self._commands = navigation.tasks, navigation.commands
            create_navigation_timers(self, self._navigator, self._commands)
        self._frames = FrameCapture(
            config,
            parts,
            lookup=self._lookup_transform,
            localization=self._localization,
            pending=self._pending_images,
            depth_frames=self._depth_frames,
            camera_infos=self._camera_infos,
            commands=self._commands,
            clock=self._memory_time,
            log=self.get_logger(),
        )
        self._housekeeping = Housekeeping(
            parts,
            pending=self._pending_images,
            mission_context=self._mission_context,
            mission_traces=self._mission_traces,
            command_tasks=self._command_tasks,
            clock=self._memory_time,
            log=self.get_logger(),
        )
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
            f"model {parts.embedder.model_name}, "
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

    def _lookup_transform(self, target: str, source: str, sec: int, nanosec: int, timeout_s: float) -> Any:
        """FrameCapture's transform lookup; tf2 errors become PlacecellError."""
        from rclpy.duration import Duration
        from rclpy.time import Time

        try:
            return self._tf.lookup_transform(
                target, source, Time(seconds=sec, nanoseconds=nanosec), Duration(seconds=timeout_s)
            )
        except TransformException as e:
            raise PlacecellError(str(e)) from e

    def _create_planning_environment(self, **options: Any) -> Any:
        from placecell.ros2.approach import create_planning_environment

        def current_pose() -> Pose | None:
            stamp = self.get_clock().now().to_msg()
            return self._frames.pose_at(stamp.sec, stamp.nanosec)

        return create_planning_environment(self, self._tf, current_pose, **options)

    def _receive_image(self, msg: Any) -> None:
        self._frames.receive(msg, compressed=False)

    def _receive_compressed(self, msg: Any) -> None:
        self._frames.receive(msg, compressed=True)

    def _drain_image(self) -> None:
        self._frames.drain()

    def _on_ask(self, msg: Any) -> None:
        if len(msg.data) > 2000 or not msg.data.strip():
            self._answers.publish(String(data=answer_error(msg.data[:128], "question must contain 1..2000 characters")))
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
        self._housekeeping.on_correct(msg)

    def _on_refine(self, msg: Any) -> None:
        self._housekeeping.on_refine(msg)

    def _refine(self) -> None:
        self._maintenance.submit(self._run_refiner, key="refine")

    def _run_refiner(self) -> None:
        self._housekeeping.run_refiner()

    def _curate(self) -> None:
        self._maintenance.submit(self._run_curator, key="curate")

    def _context_reference_available(self, data: dict[str, Any]) -> bool:
        return self._housekeeping.reference_available(data)

    def _run_curator(self) -> None:
        self._housekeeping.run_curator()

    def _consolidate(self) -> None:
        self._maintenance.submit(self._run_consolidator, key="consolidate")

    def _run_consolidator(self) -> None:
        self._housekeeping.run_consolidator()

    def _diagnostics(self) -> None:
        self._housekeeping.diagnostics()

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
