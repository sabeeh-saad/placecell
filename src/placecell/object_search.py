"""Opt-in, bounded viewpoint selection after an object arrival check fails."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from placecell.approach import ApproachPlan, ApproachPlanner, ViewpointRegion
from placecell.errors import ValidationError
from placecell.memory import Pose
from placecell.object_arrival import ObjectReference


@dataclass(frozen=True)
class ObjectSearchPolicy:
    max_viewpoints: int = 3
    timeout_s: float = 60
    radius_m: float = 1.5
    separation_m: float = 0.35
    max_path_m: float = 4

    def __post_init__(self) -> None:
        if type(self.max_viewpoints) is not int or not 1 <= self.max_viewpoints <= 8:
            raise ValidationError("local search needs a viewpoint limit within 1..8")
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.timeout_s, self.radius_m, self.separation_m, self.max_path_m)
        ):
            raise ValidationError("local search limits must be finite and positive")


class ObjectSearch:
    def __init__(self, planner: ApproachPlanner, policy: ObjectSearchPolicy | None = None) -> None:
        self.planner, self.policy = planner, policy or ObjectSearchPolicy()

    def next_view(
        self, reference: ObjectReference, anchor: Pose, visited: tuple[Pose, ...], canceled: Callable[[], bool]
    ) -> ApproachPlan:
        if len(visited) > self.policy.max_viewpoints:
            raise ValidationError("local search viewpoint limit reached")
        p = self.policy
        plan = self.planner.plan(
            reference.record,
            reference.views[0],
            canceled,
            region=ViewpointRegion(anchor, p.radius_m, visited, p.separation_m, p.max_path_m),
        )
        if plan is None:
            raise ValidationError("local search needs a fresh, reliable object position")
        return plan

    def valid(self, plan: ApproachPlan, canceled: Callable[[], bool]) -> bool:
        return plan.region is not None and self.planner.valid(plan, canceled)
