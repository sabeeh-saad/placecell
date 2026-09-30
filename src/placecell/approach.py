"""Bounded approach planning using a conservative circular robot footprint envelope."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import accumulate, pairwise
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from placecell.errors import ValidationError
from placecell.memory import Pose
from placecell.object_types import ObjectRecord, ObjectView

# Work bound for one path check: footprint samples, and the costmap cells their windows cover.
_PATH_SAMPLES = 20_000
_PATH_CELLS = 32_000_000


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
        """Apply Nav2's collision rule to a footprint whose circumscribed radius is ``radius``.

        The raw costmap is already inflated by the inscribed radius, so only the centre cell
        must stay below 253 (inscribed). No lethal (254) or unknown (255) cell may touch the
        enclosing circle, and the circle must lie inside the map.
        """
        if not pose.same_frame(self.origin) or not math.isfinite(radius) or not 0 < radius <= 5:
            return False
        dx, dy = pose.x - self.origin.x, pose.y - self.origin.y
        c, s = math.cos(self.origin.yaw), math.sin(self.origin.yaw)
        return self._clear(c * dx + s * dy, -s * dx + c * dy, radius, 0.0)

    def _clear(self, x: float, y: float, radius: float, slack: float) -> bool:
        """Check a circle at costmap-local ``x, y``; any cell within ``slack`` counts as a centre cell."""
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
        gap = gap_y[:, None] ** 2 + gap_x[None, :] ** 2
        window = self.cells[lo_y : hi_y + 1, lo_x : hi_x + 1]
        return bool(np.all(window[gap <= radius**2] < 254) and np.all(window[gap <= slack**2] < 253))

    def path_free(self, path: tuple[Pose, ...], radius: float) -> bool:
        """Apply ``free`` along the whole polyline, sampled by arc length within a fixed work bound."""
        if (
            not path
            or len(path) > 4096
            or not all(p.same_frame(self.origin) for p in path)
            or not math.isfinite(radius)
            or not 0 < radius <= 5
        ):
            return False
        arc = list(accumulate((a.distance_to(b) for a, b in pairwise(path)), initial=0.0))
        length = arc[-1]
        if not math.isfinite(length):
            return False
        # Half-cell spacing, doubled on long paths until the work fits; wider spacing only enlarges the circle.
        spacing = self.resolution / 2
        while True:
            samples = math.ceil(length / spacing) + 1
            window = (2 * (radius + spacing / 2) / self.resolution + 2) ** 2  # upper bound on cells per sample
            if samples <= _PATH_SAMPLES and samples * window <= _PATH_CELLS:
                break
            spacing *= 2
            if spacing > radius:
                return False
        at = np.linspace(0.0, length, samples)
        dx = np.interp(at, arc, [p.x for p in path]) - self.origin.x
        dy = np.interp(at, arc, [p.y for p in path]) - self.origin.y
        c, s = math.cos(self.origin.yaw), math.sin(self.origin.yaw)
        # Every path point lies within half a spacing of a sample; widen both checks by that much.
        slack = spacing / 2
        return all(
            self._clear(x, y, radius + slack, slack)
            for x, y in zip((c * dx + s * dy).tolist(), (-s * dx + c * dy).tolist(), strict=True)
        )


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
class ViewpointRegion:
    """Bound every leg of a local search to the original arrival area."""

    center: Pose
    radius_m: float = 1.5
    visited: tuple[Pose, ...] = ()
    separation_m: float = 0.35
    max_path_m: float = 4.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in (self.radius_m, self.separation_m, self.max_path_m)):
            raise ValidationError("search region limits must be finite and positive")

    def contains(self, pose: Pose) -> bool:
        return pose.same_frame(self.center) and pose.distance_to(self.center) <= self.radius_m

    def candidate(self, pose: Pose) -> bool:
        return self.contains(pose) and all(pose.distance_to(previous) >= self.separation_m for previous in self.visited)


@dataclass(frozen=True)
class ApproachPlan:
    object: ObjectRecord
    viewpoint: Pose
    pose: Pose
    path: tuple[Pose, ...]
    planned_at: float
    region: ViewpointRegion | None = None


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
        self,
        record: ObjectRecord,
        view: ObjectView,
        canceled: Callable[[], bool] = lambda: False,
        *,
        region: ViewpointRegion | None = None,
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
            if (region is None or region.candidate(goal)) and snapshot.costmap.free(goal, snapshot.footprint_radius_m):
                candidates.append(goal)
        # Preserve the learned side of the object before optimizing travel distance.
        # A shorter route to its back/side can hide the details used for identification.
        reference_heading = Pose(0, 0, direction + math.pi - p.camera_yaw_offset_rad)
        candidates.sort(
            key=lambda goal: (goal.heading_difference(reference_heading), goal.distance_to(snapshot.robot_pose))
        )
        valid: list[tuple[float, float, ApproachPlan]] = []
        for goal in candidates[: p.max_path_requests]:
            if canceled() or self.monotonic() >= deadline:
                raise ValidationError("approach planning canceled or timed out")
            path = self.environment.path(snapshot.robot_pose, goal, deadline, canceled)
            if path is None or not self._usable_path(path, snapshot, goal, region):
                continue
            length = sum(a.distance_to(b) for a, b in pairwise(path))
            valid.append(
                (
                    goal.heading_difference(reference_heading),
                    length,
                    ApproachPlan(record, view.memory.pose, goal, path, self.clock(), region),
                )
            )
        if not valid:
            raise ValidationError("no collision-checked approach path is available")
        plan = min(valid, key=lambda item: item[:2])[2]
        if not self.valid(plan, canceled):
            raise ValidationError("approach became unavailable while planning")
        return plan

    def _usable_path(
        self, path: tuple[Pose, ...], snapshot: PlanningSnapshot, goal: Pose, region: ViewpointRegion | None = None
    ) -> bool:
        p = self.policy
        if not path or len(path) > 4096 or any(not pose.same_frame(goal) for pose in path):
            return False
        if (
            path[0].distance_to(snapshot.robot_pose) > p.start_tolerance_m
            or path[-1].distance_to(goal) > p.endpoint_tolerance_m
        ):
            return False
        full_path = (snapshot.robot_pose, *path, goal)
        limit = min(p.max_path_m, region.max_path_m) if region else p.max_path_m
        if region and not all(region.contains(pose) for pose in full_path):
            return False
        if sum(a.distance_to(b) for a, b in pairwise(full_path)) > limit:
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
        return self._usable_path(plan.path, snapshot, plan.pose, plan.region)
