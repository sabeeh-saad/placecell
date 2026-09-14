"""Bounded approach planning using a conservative circular robot footprint envelope."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from placecell.errors import ValidationError
from placecell.memory import Pose
from placecell.object_types import ObjectRecord, ObjectView


@dataclass(frozen=True)
class Costmap:
    origin: Pose
    resolution: float
    timestamp: float
    cells: NDArray[np.uint8] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        cells = np.asarray(self.cells)
        if (
            cells.ndim != 2
            or not cells.size
            or cells.size > 16_000_000
            or not np.issubdtype(cells.dtype, np.integer)
            or np.any(cells < 0)
            or np.any(cells > 255)
        ):
            raise ValidationError("costmap must be a bounded 2D byte grid")
        if not math.isfinite(self.resolution) or not 0.01 <= self.resolution <= 1:
            raise ValidationError("costmap resolution must be within 0.01..1 metres")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValidationError("costmap timestamp must be finite and nonnegative")
        copied = cells.astype(np.uint8, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "cells", copied)

    def free(self, pose: Pose, radius: float) -> bool:
        """Reject unknown, inscribed/lethal and outside-map cells intersecting the footprint envelope."""
        if not pose.same_frame(self.origin) or not math.isfinite(radius) or not 0 < radius <= 5:
            return False
        dx, dy = pose.x - self.origin.x, pose.y - self.origin.y
        c, s = math.cos(self.origin.yaw), math.sin(self.origin.yaw)
        x, y = c * dx + s * dy, -s * dx + c * dy
        height, width = self.cells.shape
        if (
            x - radius < 0
            or y - radius < 0
            or x + radius >= width * self.resolution
            or y + radius >= height * self.resolution
        ):
            return False
        lo_x, hi_x = int((x - radius) / self.resolution), int((x + radius) / self.resolution)
        lo_y, hi_y = int((y - radius) / self.resolution), int((y + radius) / self.resolution)
        # Distance to each cell's rectangle, not its centre; thin and diagonal collisions still count.
        xs = np.arange(lo_x, hi_x + 1) * self.resolution
        ys = np.arange(lo_y, hi_y + 1) * self.resolution
        gap_x = np.maximum(np.maximum(xs - x, x - xs - self.resolution), 0)
        gap_y = np.maximum(np.maximum(ys - y, y - ys - self.resolution), 0)
        mask = gap_y[:, None] ** 2 + gap_x[None, :] ** 2 <= radius**2
        return bool(np.all(self.cells[lo_y : hi_y + 1, lo_x : hi_x + 1][mask] < 253))

    def path_free(self, path: tuple[Pose, ...], radius: float) -> bool:
        if not path or len(path) > 4096 or not all(p.same_frame(self.origin) for p in path):
            return False
        if not self.free(path[0], radius + self.resolution / 4):
            return False
        steps_used = 0
        max_steps = min(20_000, int(2_000_000 / (2 * radius / self.resolution + 5) ** 2))
        for start, end in pairwise(path):
            steps = max(1, math.ceil(start.distance_to(end) / (self.resolution / 2)))
            steps_used += steps
            if steps_used > max_steps:
                return False
            # Enlarge by half a sampling step to cover space between samples as well.
            for step in range(1, steps + 1):
                t = step / steps
                pose = Pose(
                    start.x + (end.x - start.x) * t,
                    start.y + (end.y - start.y) * t,
                    frame_id=start.frame_id,
                    map_id=start.map_id,
                )
                if not self.free(pose, radius + self.resolution / 4):
                    return False
        return True


@dataclass(frozen=True)
class PlanningSnapshot:
    costmap: Costmap
    robot_pose: Pose
    footprint_radius_m: float
    footprint_timestamp: float

    def __post_init__(self) -> None:
        if not self.robot_pose.same_frame(self.costmap.origin):
            raise ValidationError("robot pose and costmap must share map and frame")
        if not math.isfinite(self.footprint_radius_m) or not 0.02 <= self.footprint_radius_m <= 3:
            raise ValidationError("a measured robot footprint is required")
        if not math.isfinite(self.footprint_timestamp) or self.footprint_timestamp < 0:
            raise ValidationError("invalid footprint timestamp")


class PlanningEnvironment(Protocol):
    def snapshot(self, deadline: float, canceled: Callable[[], bool]) -> PlanningSnapshot: ...

    def path(
        self, start: Pose, goal: Pose, deadline: float, canceled: Callable[[], bool]
    ) -> tuple[Pose, ...] | None: ...


@dataclass(frozen=True)
class ApproachPolicy:
    clearance_m: float = 0.5
    max_uncertainty_m: float = 0.35
    max_position_age_s: float = 300
    max_sensor_age_s: float = 2
    camera_yaw_offset_rad: float = 0
    max_view_change_rad: float = math.pi / 3
    candidates: int = 9
    max_path_requests: int = 3
    planning_timeout_s: float = 8
    max_path_m: float = 30
    start_tolerance_m: float = 0.25
    endpoint_tolerance_m: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            self.clearance_m,
            self.max_uncertainty_m,
            self.max_position_age_s,
            self.max_sensor_age_s,
            self.max_view_change_rad,
            self.planning_timeout_s,
            self.max_path_m,
            self.start_tolerance_m,
            self.endpoint_tolerance_m,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive) or not math.isfinite(self.camera_yaw_offset_rad):
            raise ValidationError("approach thresholds must be finite and positive")
        if not 1 <= self.max_path_requests <= self.candidates <= 32 or self.max_view_change_rad > math.pi / 2:
            raise ValidationError("approach candidate bounds are invalid")


@dataclass(frozen=True)
class ApproachPlan:
    object: ObjectRecord
    viewpoint: Pose
    pose: Pose
    path: tuple[Pose, ...]
    planned_at: float


class ApproachPlanner:
    def __init__(
        self,
        environment: PlanningEnvironment,
        policy: ApproachPolicy | None = None,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.environment, self.policy, self.clock, self.monotonic = (
            environment,
            policy or ApproachPolicy(),
            clock,
            monotonic,
        )

    def _snapshot(self, deadline: float, canceled: Callable[[], bool]) -> PlanningSnapshot:
        if canceled() or self.monotonic() >= deadline:
            raise ValidationError("approach planning canceled or timed out")
        snapshot = self.environment.snapshot(deadline, canceled)
        if any(
            not 0 <= self.clock() - stamp <= self.policy.max_sensor_age_s
            for stamp in (snapshot.costmap.timestamp, snapshot.footprint_timestamp)
        ):
            raise ValidationError("approach costmap or footprint is stale")
        return snapshot

    def plan(
        self, record: ObjectRecord, view: ObjectView, canceled: Callable[[], bool] = lambda: False
    ) -> ApproachPlan | None:
        p, position = self.policy, record.position
        if record.status != "present" or record.misses or not view.memory.localization_checked:
            raise ValidationError("object presence or localization is uncertain")
        if (
            position is None
            or record.position_timestamp is None
            or position.uncertainty_m > p.max_uncertainty_m
            or not 0 <= self.clock() - record.position_timestamp <= p.max_position_age_s
        ):
            return None  # The caller may use the already verified recorded viewpoint.
        deadline = self.monotonic() + p.planning_timeout_s
        snapshot = self._snapshot(deadline, canceled)
        if (
            view.object_id != record.id
            or not view.memory.pose.same_frame(snapshot.robot_pose)
            or record.map_id != snapshot.robot_pose.map_id
            or record.frame_id != snapshot.robot_pose.frame_id
        ):
            raise ValidationError("object, viewpoint and planning map differ")
        radius = snapshot.footprint_radius_m + p.clearance_m + position.radius_m + position.uncertainty_m
        direction = math.atan2(view.memory.pose.y - position.y, view.memory.pose.x - position.x)
        candidates = []
        angles = (
            [0.0] if p.candidates == 1 else np.linspace(-p.max_view_change_rad, p.max_view_change_rad, p.candidates)
        )
        for angle in angles:
            bearing = direction + float(angle)
            goal = Pose(
                position.x + radius * math.cos(bearing),
                position.y + radius * math.sin(bearing),
                bearing + math.pi - p.camera_yaw_offset_rad,
                record.frame_id,
                record.map_id,
            )
            if snapshot.costmap.free(goal, snapshot.footprint_radius_m):
                candidates.append(goal)
        candidates.sort(key=lambda goal: goal.distance_to(snapshot.robot_pose))
        valid: list[tuple[float, ApproachPlan]] = []
        for goal in candidates[: p.max_path_requests]:
            if canceled() or self.monotonic() >= deadline:
                raise ValidationError("approach planning canceled or timed out")
            path = self.environment.path(snapshot.robot_pose, goal, deadline, canceled)
            if path is None or not self._usable_path(path, snapshot, goal):
                continue
            length = sum(a.distance_to(b) for a, b in pairwise(path))
            valid.append((length, ApproachPlan(record, view.memory.pose, goal, path, self.clock())))
        if not valid:
            raise ValidationError("no collision-checked approach path is available")
        plan = min(valid, key=lambda item: item[0])[1]
        if not self.valid(plan, canceled):
            raise ValidationError("approach became unavailable while planning")
        return plan

    def _usable_path(self, path: tuple[Pose, ...], snapshot: PlanningSnapshot, goal: Pose) -> bool:
        p = self.policy
        if not path or len(path) > 4096 or any(not pose.same_frame(goal) for pose in path):
            return False
        if (
            path[0].distance_to(snapshot.robot_pose) > p.start_tolerance_m
            or path[-1].distance_to(goal) > p.endpoint_tolerance_m
        ):
            return False
        full_path = (snapshot.robot_pose, *path, goal)
        if sum(a.distance_to(b) for a, b in pairwise(full_path)) > p.max_path_m:
            return False
        return snapshot.costmap.path_free(full_path, snapshot.footprint_radius_m)

    def valid(self, plan: ApproachPlan, canceled: Callable[[], bool] = lambda: False) -> bool:
        snapshot = self._snapshot(self.monotonic() + self.policy.planning_timeout_s, canceled)
        position = plan.object.position
        if (
            position is None
            or not snapshot.robot_pose.same_frame(plan.pose)
            or plan.object.position_timestamp is None
            or not 0 <= self.clock() - plan.object.position_timestamp <= self.policy.max_position_age_s
        ):
            return False
        minimum = snapshot.footprint_radius_m + self.policy.clearance_m + position.radius_m + position.uncertainty_m
        if math.hypot(plan.pose.x - position.x, plan.pose.y - position.y) + 1e-6 < minimum:
            return False
        return self._usable_path(plan.path, snapshot, plan.pose)
