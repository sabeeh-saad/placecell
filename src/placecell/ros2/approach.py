"""Nav2 planning queries and fresh costmap/footprint inputs. No movement is issued here."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from placecell.approach import Costmap, PlanningSnapshot
from placecell.errors import ValidationError
from placecell.memory import Pose
from placecell.ros2.bridge import stamp_to_seconds, yaw_from_quaternion


def message_pose(message: Any, frame_id: str, map_id: str) -> Pose:
    q = message.orientation
    if (
        not math.isclose(sum(float(v) ** 2 for v in (q.x, q.y, q.z, q.w)), 1, abs_tol=0.001)
        or abs(q.x) > 0.001
        or abs(q.y) > 0.001
    ):
        raise ValidationError("planning requires a normalized planar pose")
    return Pose(
        float(message.position.x), float(message.position.y), yaw_from_quaternion(q.x, q.y, q.z, q.w), frame_id, map_id
    )


def costmap_from_message(message: Any, frame_id: str, map_id: str) -> Costmap:
    if message.header.frame_id != frame_id:
        raise ValidationError("costmap belongs to a different frame")
    m = message.metadata
    if min(m.size_x, m.size_y) < 1 or m.size_x * m.size_y > 16_000_000 or len(message.data) != m.size_x * m.size_y:
        raise ValidationError("invalid Nav2 costmap dimensions")
    return Costmap(
        message_pose(m.origin, frame_id, map_id),
        float(m.resolution),
        stamp_to_seconds(message.header.stamp.sec, message.header.stamp.nanosec),
        np.asarray(message.data).reshape(m.size_y, m.size_x),
    )


def footprint_radius(message: Any, base_x: float, base_y: float) -> float:
    """Published polygon and robot origin must be in the SAME frame at the SAME timestamp."""
    points = [(float(p.x) - base_x, float(p.y) - base_y) for p in message.polygon.points]
    if not 3 <= len(points) <= 128 or not all(math.isfinite(v) for point in points for v in point):
        raise ValidationError("invalid published footprint")
    area = abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1], strict=True))) / 2
    radius = max(math.hypot(x, y) for x, y in points)
    if area < 1e-5 or not 0.02 <= radius <= 3:
        raise ValidationError("published footprint is degenerate or too large")
    return radius


class Nav2PlanningEnvironment:
    def __init__(
        self,
        client: Any,
        make_goal: Callable[[Pose, Pose], Any],
        current_pose: Callable[[], Pose | None],
        frame_id: str,
        map_id: str,
        *,
        request_timeout_s: float = 2,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValidationError("planning request timeout must be positive")
        self.client, self.make_goal, self.current_pose = client, make_goal, current_pose
        self.frame_id, self.map_id = frame_id, map_id
        self.timeout, self.monotonic = request_timeout_s, monotonic
        self._lock = threading.Lock()
        self._costmap: Costmap | None = None
        self._footprint: tuple[float, float] | None = None

    def update_costmap(self, message: Any) -> None:
        try:
            grid = costmap_from_message(message, self.frame_id, self.map_id)
        except Exception:
            with self._lock:
                self._costmap = None
            raise
        with self._lock:
            self._costmap = grid

    def update_footprint(self, message: Any, base_x: float, base_y: float) -> None:
        try:
            radius = footprint_radius(message, base_x, base_y)
            stamp = stamp_to_seconds(message.header.stamp.sec, message.header.stamp.nanosec)
        except Exception:
            self.invalidate_footprint()
            raise
        with self._lock:
            self._footprint = radius, stamp

    def invalidate_footprint(self) -> None:
        with self._lock:
            self._footprint = None

    def snapshot(self, deadline: float, canceled: Callable[[], bool]) -> PlanningSnapshot:
        if canceled() or self.monotonic() >= deadline:
            raise ValidationError("planning canceled or timed out")
        with self._lock:
            grid, footprint = self._costmap, self._footprint
        pose = self.current_pose()
        if grid is None or footprint is None or pose is None:
            raise ValidationError("approach needs a published costmap, footprint and localized robot pose")
        return PlanningSnapshot(grid, pose, *footprint)

    def _wait(self, future: Any, deadline: float, canceled: Callable[[], bool]) -> Any:
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        while not ready.wait(0.02):
            if canceled() or self.monotonic() >= deadline:
                raise ValidationError("Nav2 planning canceled or timed out")
        if canceled() or self.monotonic() >= deadline:
            raise ValidationError("Nav2 planning canceled or timed out")
        return future.result()

    def path(self, start: Pose, goal: Pose, deadline: float, canceled: Callable[[], bool]) -> tuple[Pose, ...] | None:
        if canceled() or self.monotonic() >= deadline or not self.client.server_is_ready():
            raise ValidationError("Nav2 planner unavailable, canceled or timed out")
        deadline = min(deadline, self.monotonic() + self.timeout)
        future = self.client.send_goal_async(self.make_goal(start, goal))
        handle = None
        try:
            handle = self._wait(future, deadline, canceled)
            if not handle.accepted:
                return None
            response = self._wait(handle.get_result_async(), deadline, canceled)
        except Exception:

            def cancel_late(done: Any) -> None:
                try:
                    accepted = done.result()
                    if accepted.accepted:
                        accepted.cancel_goal_async()
                except Exception:
                    logging.getLogger(__name__).warning("Nav2 planning cancellation failed", exc_info=True)

            if handle is None:
                future.add_done_callback(cancel_late)
            elif handle.accepted:
                handle.cancel_goal_async()
            raise
        if response.status != 4 or getattr(response.result, "error_code", 0):
            return None
        path = response.result.path
        if path.header.frame_id != self.frame_id or not 1 <= len(path.poses) <= 4096:
            raise ValidationError("Nav2 returned a malformed or foreign path")
        if any(p.header.frame_id not in ("", self.frame_id) for p in path.poses):
            raise ValidationError("Nav2 path contains a foreign pose")
        return tuple(message_pose(p.pose, self.frame_id, self.map_id) for p in path.poses)


def create_planning_environment(
    node: Any,
    tf: Any,
    current_pose: Callable[[], Pose | None],
    *,
    frame_id: str,
    map_id: str,
    base_frame: str,
    costmap_topic: str,
    footprint_topic: str,
    action_name: str,
    planner_id: str,
    timeout_s: float,
) -> Nav2PlanningEnvironment:
    from geometry_msgs.msg import PolygonStamped, PoseStamped
    from nav2_msgs.action import ComputePathToPose
    from nav2_msgs.msg import Costmap as CostmapMessage
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time

    def stamped(pose: Pose) -> Any:
        message = PoseStamped()
        message.header.frame_id, message.header.stamp = pose.frame_id, node.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y = float(pose.x), float(pose.y)
        message.pose.orientation.z, message.pose.orientation.w = math.sin(pose.yaw / 2), math.cos(pose.yaw / 2)
        return message

    def make_goal(start: Pose, goal: Pose) -> Any:
        request = ComputePathToPose.Goal()
        request.start, request.goal, request.use_start, request.planner_id = (
            stamped(start),
            stamped(goal),
            True,
            planner_id,
        )
        return request

    environment = Nav2PlanningEnvironment(
        ActionClient(node, ComputePathToPose, action_name),
        make_goal,
        current_pose,
        frame_id,
        map_id,
        request_timeout_s=timeout_s,
    )

    def on_costmap(message: Any) -> None:
        try:
            environment.update_costmap(message)
        except Exception as e:
            node.get_logger().warning(f"approach costmap rejected: {e}", throttle_duration_sec=5.0)

    def on_footprint(message: Any) -> None:
        try:
            stamp = message.header.stamp
            transform = tf.lookup_transform(
                message.header.frame_id,
                base_frame,
                Time(seconds=stamp.sec, nanoseconds=stamp.nanosec),
                Duration(seconds=0),
            )
            t = transform.transform.translation
            environment.update_footprint(message, t.x, t.y)
        except Exception as e:
            environment.invalidate_footprint()
            node.get_logger().warning(f"approach footprint rejected: {e}", throttle_duration_sec=5.0)

    node.create_subscription(CostmapMessage, costmap_topic, on_costmap, qos_profile_sensor_data)
    node.create_subscription(PolygonStamped, footprint_topic, on_footprint, qos_profile_sensor_data)
    return environment
