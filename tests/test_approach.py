from dataclasses import replace

import numpy as np
import pytest

from placecell import (
    ApproachPlanner,
    ApproachPolicy,
    Costmap,
    Evidence,
    EvidenceKind,
    Memory,
    ObjectPosition,
    PlanningSnapshot,
    Pose,
)
from placecell.depth import Box
from placecell.errors import ValidationError
from placecell.object_types import ObjectRecord, ObjectView


def pose(x=0, y=0, yaw=0):
    return Pose(x, y, yaw, "map", "office-v1")


def grid():
    return Costmap(pose(-5, -5), 0.05, 1000, np.zeros((200, 200), dtype=np.uint8))


def target():
    record = ObjectRecord(
        "printer-1",
        "r1",
        "front",
        "map",
        "office-v1",
        "printer",
        1000,
        1000,
        ObjectPosition(3, 0, 0.8, 0.05, 0.2),
        position_timestamp=1000,
    )
    memory = Memory.create("r1", "front", 1000, pose(), Evidence(EvidenceKind.FRAME, "/frame.jpg"), "red printer")
    return record, ObjectView(record.id, replace(memory, localization_checked=True), Box(0.4, 0.4, 0.6, 0.6), b"crop")


class Environment:
    def __init__(self):
        self.state = PlanningSnapshot(grid(), pose(), 0.3, 1000)
        self.calls = []
        self.result = None
        self.after_path = lambda: None

    def snapshot(self, *_):
        return self.state

    def path(self, start, goal, *_):
        self.calls.append(goal)
        self.after_path()
        return self.result(start, goal) if self.result else (start, goal)


def planner(environment, **options):
    return ApproachPlanner(
        environment, ApproachPolicy(candidates=1, max_path_requests=1, **options), clock=lambda: 1000
    )


def test_stopping_distance_facing_direction_and_new_footprint_envelope():
    env = Environment()
    plan = planner(env).plan(*target())
    assert plan.pose.x == pytest.approx(1.95)
    assert plan.pose.y == pytest.approx(0)
    assert plan.pose.heading_difference(pose()) < 1e-6
    assert plan.viewpoint == pose()
    assert len(env.calls) == 1
    env.state = replace(env.state, footprint_radius_m=0.4)
    assert not planner(env).valid(plan)


def test_camera_mount_offset_adjusts_robot_heading():
    plan = planner(Environment(), camera_yaw_offset_rad=0.5).plan(*target())
    assert plan.pose.heading_difference(pose(yaw=-0.5)) < 1e-6


def test_candidate_search_tries_other_paths_within_request_budget():
    env = Environment()
    env.result = lambda start, goal: None if len(env.calls) == 1 else (start, goal)
    plan = ApproachPlanner(env, clock=lambda: 1000).plan(*target())
    assert len(env.calls) == 3 and plan.pose in env.calls[1:]
    facing = pose(yaw=float(np.arctan2(-plan.pose.y, 3 - plan.pose.x)))
    assert plan.pose.heading_difference(facing) < 1e-6


def test_path_sampling_bounds_work_and_includes_start_envelope():
    env = grid()
    assert not env.path_free((pose(), pose(500)), 0.3)
    cells = np.array(env.cells)
    cells[106, 100] = 254
    env = replace(env, cells=cells)
    assert env.free(pose(0.001, -0.01), 0.3)
    assert not env.path_free((pose(0.001, -0.01), pose(0.001, -0.02)), 0.3)


@pytest.mark.parametrize("change", ["no_position", "old_position", "unknown_timestamp", "uncertain_position"])
def test_uncertain_or_unmeasured_position_keeps_viewpoint_fallback(change):
    record, view = target()
    if change == "no_position":
        record = replace(record, position=None, position_timestamp=None)
    elif change == "old_position":
        record = replace(record, position_timestamp=600)
    elif change == "unknown_timestamp":
        record = replace(record, position_timestamp=None)
    else:
        record = replace(record, position=replace(record.position, uncertainty_m=1))
    env = Environment()
    assert planner(env).plan(record, view) is None
    assert env.calls == []


@pytest.mark.parametrize("cost", [253, 254, 255])
def test_path_collision_and_unknown_cells_reject_a_clear_endpoint(cost):
    env = Environment()
    cells = np.array(env.state.costmap.cells)
    cells[:, 120] = cost  # a wall at x=1 between start and the otherwise clear goal
    env.state = replace(env.state, costmap=replace(env.state.costmap, cells=cells))
    assert env.state.costmap.free(pose(1.95), 0.3)
    with pytest.raises(ValidationError, match="path"):
        planner(env).plan(*target())


def test_footprint_collision_interior_sampling_and_rotated_origin():
    original = grid()
    cells = np.array(original.cells)
    cells[100, 145] = 254
    updated = replace(original, cells=cells)
    assert updated.free(pose(1.98), 0.05)
    assert not updated.free(pose(1.98), 0.3)
    assert not original.free(pose(4.9), 0.3)
    assert not original.free(Pose(0, 0, map_id="foreign"), 0.3)
    rotated = Costmap(pose(5, -5, np.pi / 2), 0.05, 1000, np.zeros((200, 200), dtype=np.uint8))
    assert rotated.free(pose(), 0.3)
    assert original.path_free((pose(), pose(1)), 0.3)
    assert not updated.path_free((pose(), pose(3)), 0.3)


@pytest.mark.parametrize(
    "result",
    [
        lambda s, g: (),
        lambda s, g: None,
        lambda s, g: (pose(2), g),
        lambda s, g: (s, pose(1)),
        lambda s, g: (Pose(0, 0), g),
    ],
)
def test_malformed_foreign_and_partial_planner_paths_cannot_authorize_goal(result):
    env = Environment()
    env.result = result
    with pytest.raises(ValidationError):
        planner(env).plan(*target())


@pytest.mark.parametrize("what", ["costmap", "footprint", "missing", "unlocalized", "foreign"])
def test_stale_or_untrusted_inputs_stop_planning(what):
    env = Environment()
    record, view = target()
    if what == "costmap":
        env.state = replace(env.state, costmap=replace(env.state.costmap, timestamp=990))
    elif what == "footprint":
        env.state = replace(env.state, footprint_timestamp=990)
    elif what == "missing":
        record = replace(record, status="missing")
    elif what == "unlocalized":
        view = replace(view, memory=replace(view.memory, localization_checked=False))
    else:
        record = replace(record, map_id="another")
    with pytest.raises(ValidationError):
        planner(env).plan(record, view)
    assert env.calls == []


def test_costmap_change_during_planning_blocks_dispatch():
    env = Environment()

    def obstruct():
        cells = np.full((200, 200), 254, dtype=np.uint8)
        env.state = replace(env.state, costmap=replace(env.state.costmap, cells=cells))

    env.after_path = obstruct
    with pytest.raises(ValidationError, match="became unavailable"):
        planner(env).plan(*target())


def test_canceled_or_overlong_plan_is_not_used():
    env = Environment()
    with pytest.raises(ValidationError, match="canceled"):
        planner(env).plan(*target(), canceled=lambda: True)
    env.result = lambda start, goal: (start, pose(0, 4), goal)
    with pytest.raises(ValidationError):
        planner(env, max_path_m=3).plan(*target())


def test_invalid_maps_and_policies_are_rejected():
    with pytest.raises(ValidationError):
        Costmap(pose(), 0.05, 1000, np.full((5, 5), -1))
    with pytest.raises(ValidationError):
        replace(grid(), resolution=0)
    with pytest.raises(ValidationError):
        ApproachPolicy(clearance_m=-1)
    with pytest.raises(ValidationError):
        ApproachPolicy(candidates=100)
    with pytest.raises(ValidationError):
        PlanningSnapshot(grid(), pose(), 0, 1000)


@pytest.mark.parametrize("change", ["none", "object", "cancel", "localization"])
def test_object_command_checks_planning_changes_and_verifies_arrival_at_new_pose(tmp_path, change):
    import json

    from placecell import (
        CollectionInfo,
        DestinationResolver,
        InMemoryStore,
        NavigationCommands,
        NavigationEvent,
        Recall,
    )
    from placecell.objects import ObjectRecall, ObjectTracker
    from placecell.ros2.node import navigation_payload
    from tests.test_navigation import FakeNavigator
    from tests.test_objects import Detector, Matched, PixelEmbedder, ingest, observation

    embedder = PixelEmbedder()
    store = InMemoryStore(CollectionInfo("test", embedder.model_name, 3))
    record = ingest(ObjectTracker(store, embedder, Detector()), observation(tmp_path))[0]
    store.objects.save(replace(record, position=target()[0].position))
    env = Environment()
    now, ready = [1000], [True]
    lookup = DestinationResolver(
        store,
        Recall(store, embedder, clock=lambda: now[0]),
        robot_id="r1",
        camera_id="front",
        map_id="office-v1",
        verifier=Matched(),
        clock=lambda: now[0],
        objects=ObjectRecall(store, embedder, clock=lambda: now[0]),
        approach=planner(env),
    )
    nav, tasks, events = FakeNavigator(), [], []
    commands = NavigationCommands(
        lookup,
        nav,
        lambda f: tasks.append(f) is None,
        events.append,
        observation_clock=lambda: now[0],
        localization_ready=lambda: ready[0],
    )
    if change == "object":
        env.after_path = lambda: store.objects.delete(record.id)
    elif change == "cancel":
        env.after_path = commands.cancel
    elif change == "localization":
        env.after_path = lambda: ready.__setitem__(0, False)
    commands.handle("go to printer")
    tasks.pop()()
    if change != "none":
        assert not nav.sent and not commands.busy
        store.close()
        return
    _, destination, callback = nav.sent[0]
    assert destination.pose.x == pytest.approx(1.95)
    assert destination.memory.pose.x == 0 and lookup.current(destination)
    assert json.loads(navigation_payload(events[-1]))["destination"]["goal_kind"] == "object_approach"
    callback(NavigationEvent("succeeded"))
    assert commands.needs_observation
    now[0] = 1001
    obs = observation(tmp_path, 1001)
    commands.observe(obs)  # The old viewpoint cannot verify arrival at the new pose.
    assert commands.needs_observation and not tasks
    commands.observe(replace(obs, pose=destination.pose))
    tasks.pop()()
    assert events[-1].state == "destination_unverified" and not commands.busy
    assert events[-1].object_result == "unavailable"  # An object goal needs its own arrival verifier.
    store.close()
