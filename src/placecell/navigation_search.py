"""Local search for an object around its remembered viewpoint when arrival finds it missing.

`NavigationCommands` plans the next viewpoint outside its lock and dispatches it under the
lock. `LocalSearch` holds one trip's search anchor, the viewpoints used, how many and until
when the search may continue, and plans the next checked goal; it never locks or sends one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from placecell.errors import TargetValidationError

if TYPE_CHECKING:
    from placecell.memory import Pose
    from placecell.navigation import Destination, DestinationResolver
    from placecell.object_search import ObjectSearch


@dataclass(slots=True)
class LocalSearch:
    """One trip's local search: its deadline, anchor, the viewpoints visited and their count."""

    deadline: float | None = None
    anchor: Pose | None = None
    visited: tuple[Pose, ...] = ()
    count: int = 0

    def start(self, pose: Pose) -> None:
        """Search around the dispatched destination, which counts as visited."""
        self.anchor = pose
        self.visited = (pose,)

    def next_goal(
        self,
        planner: ObjectSearch,
        resolver: DestinationResolver,
        destination: Destination,
        canceled: Callable[[], bool],
    ) -> Destination:
        """Plan and check the next viewpoint; raises when the target reference or the path changes."""
        if not resolver.arrival_available(destination):
            raise TargetValidationError("the selected object reference is unavailable", "retrieval")
        assert destination.object_reference is not None and self.anchor is not None
        plan = planner.next_view(destination.object_reference, self.anchor, self.visited, canceled)
        if not resolver.arrival_available(destination):
            raise TargetValidationError("the selected object changed", "retrieval")
        if not planner.valid(plan, canceled):
            raise TargetValidationError("the search path changed", "geometry")
        return replace(destination, pose=plan.pose, approach=plan)

    def visit(self, goal: Destination) -> int:
        """Count a dispatched viewpoint and return its attempt number."""
        self.count += 1
        self.visited += (goal.pose,)
        return self.count
