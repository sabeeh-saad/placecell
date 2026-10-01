"""Camera messages to admitted observations: capture-time pose and depth, trust checks, admission.

Nothing here imports rclpy. Transforms come through a `TransformLookup` the node provides,
which reports a missing or late transform as a PlacecellError. Call every method from the
node's default callback group, like the camera and timer callbacks that drive it.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from placecell.depth import DepthSnapshot
from placecell.errors import PlacecellError, ValidationError
from placecell.localization import LocalizationGate
from placecell.memory import Pose
from placecell.navigation import NavigationCommands
from placecell.pipeline import Observation
from placecell.ros2.bridge import image_dimensions, pose_from_transform, stamp_to_seconds
from placecell.ros2.components import Components
from placecell.ros2.config import NodeConfig
from placecell.ros2.depth import PendingImages, aligned_snapshot

TransformLookup = Callable[[str, str, int, int, float], Any]
"""(target frame, source frame, stamp sec, stamp nanosec, timeout s) -> transform, or PlacecellError."""


class FrameCapture:
    """Raw and compressed frames take one path; `capture`, `pose_at` and `depth_at` are replaceable."""

    def __init__(
        self,
        config: NodeConfig,
        parts: Components,
        *,
        lookup: TransformLookup,
        localization: LocalizationGate,
        pending: PendingImages,
        depth_frames: deque[Any],
        camera_infos: deque[Any],
        commands: NavigationCommands | None,
        clock: Callable[[], float],
        log: Any,
    ) -> None:
        self._lookup, self._localization, self._pending_images = lookup, localization, pending
        self._depth_frames, self._camera_infos, self._commands = depth_frames, camera_infos, commands
        self._memory_time, self._log = clock, log
        self._sensors, self._object_recall, self._admission = parts.sensors, parts.object_recall, parts.admission
        self._worker, self._builder, self._writer = parts.worker, parts.builder, parts.writer
        self._recording = parts.recording
        self._robot_id, self._camera_id = config.camera.robot_id, config.camera.camera_id
        self._map_frame, self._base_frame = config.localization.map_frame, config.localization.base_frame
        self._map_id, self._localization_required = config.localization.map_id, config.localization.required
        self._tf_timeout = config.localization.tf_timeout_s
        self._depth_skew = config.objects.depth_max_skew_s
        self._depth_error = config.objects.position_error_m
        self._depth_angular_error = config.objects.angular_error_rad
        self._sync_images = config.objects.enabled and bool(config.camera.depth_topic)
        self._clock_fault_reported = False

    def pose_at(self, sec: int, nanosec: int) -> Pose | None:
        try:
            if stamp_to_seconds(sec, nanosec) <= 0 or not self._sensors.ready():
                return None  # Time(0) asks TF for latest, not a capture-time transform.
            tf = self._lookup(self._map_frame, self._base_frame, sec, nanosec, self._tf_timeout)
            t, q = tf.transform.translation, tf.transform.rotation
            pose = pose_from_transform(t.x, t.y, q.x, q.y, q.z, q.w, self._map_frame, self._map_id)
        except (PlacecellError, ValueError) as e:
            self._log.warning(f"no pose for image: {e}", throttle_duration_sec=5.0)
            return None
        if self._localization_required and not self._localization.accepts(pose, stamp_to_seconds(sec, nanosec)):
            self._log.warning("skipping image: localization is missing, stale or uncertain", throttle_duration_sec=5.0)
            return None
        return pose

    def depth_at(self, msg: Any, dimensions: tuple[int, int]) -> DepthSnapshot | None:
        if self._object_recall is None:
            return None
        stamp = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
        uncertainty = self._localization.uncertainty_at(stamp)
        if uncertainty is None:
            return None
        if not self._depth_frames or not self._camera_infos:
            self._log.warning(
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
            transform = self._lookup(
                self._map_frame,
                msg.header.frame_id,
                msg.header.stamp.sec,
                msg.header.stamp.nanosec,
                self._tf_timeout,
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
        except (PlacecellError, ValueError) as e:
            self._log.warning(f"object positions unavailable: {e}", throttle_duration_sec=5.0)
            return None

    def capture(self, msg: Any, *, compressed: bool = False) -> tuple[Pose, float, DepthSnapshot | None] | None:
        stamp = 0.0
        try:
            stamp = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
            dimensions = image_dimensions(msg, compressed=compressed)
            pose = self.pose_at(msg.header.stamp.sec, msg.header.stamp.nanosec)
            if pose is None:
                raise ValidationError("capture-time TF/localization is unavailable")
            depth = self.depth_at(msg, dimensions)
            if self._sensors.observe(stamp, camera=True, depth=depth is not None):
                return pose, stamp, depth
        except (PlacecellError, ValueError, TypeError, AttributeError) as e:
            self._sensors.observe(stamp, camera=False, depth=False)
            self._log.warning(f"skipping untrusted camera input: {e}", throttle_duration_sec=5.0)
        return None

    def _record(self, observation: Observation) -> None:
        if self._recording is not None:
            try:
                self._recording.append(observation)
            except (OSError, PlacecellError) as e:
                self._recording = None
                self._log.error(f"recording stopped after an export failure: {e}")

    def process(self, msg: Any, compressed: bool) -> None:
        if not self._pending_images.accepts_size(msg):
            return
        capture = self.capture(msg, compressed=compressed)
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
            if compressed:
                obs = self._builder.from_compressed(stamp, msg.format, bytes(msg.data), pose)
            else:
                obs = self._builder.from_raw(
                    stamp, msg.height, msg.width, msg.encoding, msg.step, bytes(msg.data), pose
                )
        except PlacecellError as e:
            self._log.warning(f"skipped image: {e}", throttle_duration_sec=5.0)
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

    def receive(self, msg: Any, *, compressed: bool) -> None:
        """A camera callback: wait for aligned depth when objects need it, otherwise process now."""
        if self._sync_images:
            self._queue_image(msg, compressed=compressed)
        else:
            self.process(msg, compressed)

    def _queue_image(self, msg: Any, *, compressed: bool) -> None:
        if not self._sensors.ready():
            return
        if not self._pending_images.add(msg, compressed, time.monotonic(), source_now=self._memory_time()):
            try:
                stamp = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
            except (AttributeError, TypeError, ValueError):
                stamp = float("nan")
            self._sensors.observe(stamp, camera=False, depth=False)

    def drain(self) -> None:
        """Process the next frame whose depth arrived or whose wait ran out."""
        if not self._sensors.ready():
            self._pending_images.clear()
            if not self._clock_fault_reported:
                self._clock_fault_reported = True
                self._log.error(
                    "Clock changed or reset: navigation and new captures are blocked. "
                    "Confirm Nav2 is stopped, then restart with a fresh collection and keyframe directory."
                )
            return
        ready = self._pending_images.pop(time.monotonic())
        if ready is not None:
            message, compressed = ready
            self.process(message, compressed)
