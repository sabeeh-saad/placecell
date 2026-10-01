"""Local search bookkeeping and goal checks, with a scripted planner and resolver."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from placecell.errors import TargetValidationError
from placecell.memory import Pose
from placecell.navigation import Destination
from placecell.navigation_search import LocalSearch

TARGET = Destination("cup", Pose(1, 2), "memory", target="cup", object_id="cup-1", object_reference=object())


class Planner:
    def __init__(self, valid: bool = True) -> None:
        self.calls: list[tuple[Pose, tuple[Pose, ...]]] = []
        self.path_valid = valid

    def next_view(self, reference, anchor, visited, canceled):
        self.calls.append((anchor, visited))
        return SimpleNamespace(pose=Pose(3, 4))

    def valid(self, plan, canceled):
        return self.path_valid


class Resolver:
    def __init__(self, *available: bool) -> None:
        self.available = list(available)

    def arrival_available(self, destination):
        return self.available.pop(0)


def test_search_starts_at_the_destination_and_counts_dispatched_viewpoints():
    search = LocalSearch()
    search.start(Pose(1, 2))
    planner = Planner()
    goal = search.next_goal(planner, Resolver(True, True), TARGET, lambda: False)
    assert goal.pose == Pose(3, 4) and goal.approach is not None and goal.object_id == "cup-1"
    assert planner.calls == [(Pose(1, 2), (Pose(1, 2),))]
    assert search.visit(goal) == 1 and search.visited == (Pose(1, 2), Pose(3, 4))


@pytest.mark.parametrize(
    ("available", "valid", "stage"),
    [((False,), True, "retrieval"), ((True, False), True, "retrieval"), ((True, True), False, "geometry")],
)
def test_a_changed_target_or_path_stops_the_search(available, valid, stage):
    search = LocalSearch()
    search.start(Pose(1, 2))
    with pytest.raises(TargetValidationError) as error:
        search.next_goal(Planner(valid), Resolver(*available), TARGET, lambda: False)
    assert error.value.failure_stage == stage
    assert search.count == 0
