from dataclasses import replace

import pytest

from placecell import ApproachPlanner, ObjectReference, ObjectSearch, ObjectSearchPolicy, ViewpointRegion
from placecell.errors import ValidationError
from tests.test_approach import Environment, pose, target


def setup_search():
    env = Environment()
    env.state = replace(env.state, robot_pose=pose(1.95))
    record, view = target()
    search = ObjectSearch(ApproachPlanner(env, clock=lambda: 1000))
    return env, search, ObjectReference(record, (view,))


def test_search_keeps_its_original_anchor_and_does_not_repeat_visited_viewpoints():
    env, search, reference = setup_search()
    anchor = env.state.robot_pose
    visited = (anchor,)
    for _ in range(3):
        plan = search.next_view(reference, anchor, visited, lambda: False)
        assert plan.region.center == anchor
        assert all(plan.pose.distance_to(previous) >= search.policy.separation_m for previous in visited)
        assert search.valid(plan, lambda: False)
        visited += (plan.pose,)
        env.state = replace(env.state, robot_pose=plan.pose)
    with pytest.raises(ValidationError, match="viewpoint limit"):
        search.next_view(reference, anchor, visited, lambda: False)


@pytest.mark.parametrize("problem", ["detour", "path_length", "outside_start", "all_visited", "stale_position"])
def test_search_rejects_paths_or_references_outside_its_limits(problem):
    env, search, reference = setup_search()
    anchor = env.state.robot_pose
    visited = (anchor,)
    if problem == "detour":
        env.result = lambda start, goal: (start, pose(1.95, 1.6), goal)
    elif problem == "path_length":
        search.policy = ObjectSearchPolicy(max_path_m=0.01)
    elif problem == "outside_start":
        env.state = replace(env.state, robot_pose=pose())
    elif problem == "all_visited":
        search.policy = ObjectSearchPolicy(separation_m=10)
    else:
        reference = replace(reference, record=replace(reference.record, position_timestamp=600))
    with pytest.raises(ValidationError):
        search.next_view(reference, anchor, visited, lambda: False)


def test_search_revalidates_the_current_robot_pose_and_rejects_ordinary_approach_plans():
    env, search, reference = setup_search()
    anchor = env.state.robot_pose
    plan = search.next_view(reference, anchor, (anchor,), lambda: False)
    assert not search.valid(replace(plan, region=None), lambda: False)
    env.state = replace(env.state, robot_pose=pose())
    assert not search.valid(plan, lambda: False)


@pytest.mark.parametrize("limit", [0, 9, True, 1.5])
def test_search_requires_a_bounded_integer_viewpoint_count(limit):
    with pytest.raises(ValidationError):
        ObjectSearchPolicy(max_viewpoints=limit)


def test_search_requires_finite_positive_distance_and_time_limits():
    with pytest.raises(ValidationError):
        ObjectSearchPolicy(timeout_s=float("nan"))
    with pytest.raises(ValidationError):
        ViewpointRegion(pose(), radius_m=0)
