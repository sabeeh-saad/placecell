import json
import math
from dataclasses import replace

import numpy as np
import pytest

from placecell import (
    ApproachPlanner,
    CollectionInfo,
    DepthSnapshot,
    DestinationResolver,
    InMemoryStore,
    NavigationCommands,
    NavigationEvent,
    ObjectArrivalVerifier,
    ObjectSearch,
    ObjectSearchPolicy,
    ObjectTracker,
    Recall,
)
from placecell.object_types import Detection
from placecell.objects import ObjectRecall
from placecell.ros2.node import navigation_payload
from placecell.verification import SceneVerdict
from tests.test_approach import Environment, pose
from tests.test_navigation import FakeNavigator
from tests.test_object_arrival import Comparator
from tests.test_objects import CENTER, Detector, Matched, PixelEmbedder, ingest, observation


@pytest.mark.parametrize("verdict,expected", [("matched", "resolved"), ("uncertain", "not_found")])
def test_selected_object_verification_keeps_its_original_scene(tmp_path, verdict, expected):
    from placecell import parse_movement

    calls = []

    class ContextVerifier:
        def verify(self, *_):
            raise AssertionError("Object verification should retain its scene context")

        def verify_object(self, target, crop, scene):
            calls.append((target, crop, scene))
            return SceneVerdict(verdict, "pixel evidence")

    harness = Harness(tmp_path)
    harness.resolver._verifier = ContextVerifier()
    assert harness.resolver.resolve(parse_movement("go to printer")).state == expected
    assert len(calls) == 1 and calls[0][0] == "printer"
    assert calls[0][1].startswith("data:image/png;base64,")
    assert calls[0][2].startswith("data:image/png;base64,") and calls[0][1] != calls[0][2]


def frame(tmp_path, timestamp, robot_pose, visible=True):
    obs = observation(tmp_path, timestamp, (CENTER,) if visible else (), ("red",) if visible else ())
    depth = np.full((100, 100), 5.0, dtype=np.float32)
    if visible:
        depth[45:55, 45:55] = math.hypot(3 - robot_pose.x, robot_pose.y)
    c, s = math.cos(robot_pose.yaw), math.sin(robot_pose.yaw)
    yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = yaw @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    transform[:3, 3] = [robot_pose.x, robot_pose.y, 0.8]
    snapshot = DepthSnapshot.capture(depth, (100, 100, 50, 50), transform, position_error_m=0.02, angular_error_rad=0)
    return replace(obs, pose=robot_pose, depth=snapshot)


class Harness:
    def __init__(self, tmp_path, *, search=True, limit=3):
        self.tmp_path, self.now, self.mono, self.ready = tmp_path, 1000, 0, True
        embedder, self.detector, self.comparator = PixelEmbedder(), Detector(), Comparator()
        self.store = InMemoryStore(CollectionInfo("navigation", embedder.model_name, 3))
        tracker = ObjectTracker(self.store, embedder, self.detector)
        self.record = ingest(tracker, frame(tmp_path, self.now, pose()))[0]
        self.arrival = ObjectArrivalVerifier(tracker, self.comparator, clock=lambda: self.now)
        self.env = Environment()
        original_snapshot = self.env.snapshot

        def snapshot(*args):
            state = original_snapshot(*args)
            return replace(state, costmap=replace(state.costmap, timestamp=self.now), footprint_timestamp=self.now)

        self.env.snapshot = snapshot
        planner = ApproachPlanner(self.env, clock=lambda: self.now)
        self.search = ObjectSearch(planner, ObjectSearchPolicy(max_viewpoints=limit, timeout_s=60))
        self.resolver = DestinationResolver(
            self.store,
            Recall(self.store, embedder, clock=lambda: self.now),
            robot_id="r1",
            camera_id="front",
            map_id="office-v1",
            verifier=Matched(),
            clock=lambda: self.now,
            objects=ObjectRecall(self.store, embedder, clock=lambda: self.now),
            approach=planner,
            object_arrival=self.arrival,
        )
        self.nav, self.tasks, self.events = FakeNavigator(), [], []
        self.commands = NavigationCommands(
            self.resolver,
            self.nav,
            lambda f: self.tasks.append(f) is None,
            self.events.append,
            clock=lambda: self.mono,
            observation_clock=lambda: self.now,
            localization_ready=lambda: self.ready,
            search=self.search if search else None,
        )

    def start(self):
        self.commands.handle("go to printer")
        self.tasks.pop()()
        assert len(self.nav.sent) == 1

    def arrive(self):
        _, destination, callback = self.nav.sent[-1]
        self.env.state = replace(self.env.state, robot_pose=destination.pose)
        callback(NavigationEvent("succeeded"))
        assert self.commands.needs_observation

    def observe(self, visible=True):
        self.now += 1
        self.detector.detections = [Detection("printer", "red printer", CENTER)] if visible else []
        obs = frame(self.tmp_path, self.now, self.nav.sent[-1][1].pose, visible)
        self.commands.observe(obs)
        assert len(self.tasks) == 1
        self.tasks.pop()()


def test_real_object_arrival_flow_uses_selected_reference_and_does_not_write(tmp_path):
    h = Harness(tmp_path)
    h.start()
    generation = h.store.objects.generation
    h.arrive()
    h.observe()
    assert h.events[-1].state == "succeeded" and h.events[-1].object_result == "matched"
    assert not h.commands.busy and len(h.nav.sent) == 1
    assert h.store.objects.generation == generation


def test_missing_object_searches_a_checked_view_and_ignores_previous_leg_feedback(tmp_path):
    h = Harness(tmp_path)
    h.start()
    h.arrive()
    h.observe(False)
    assert h.commands.busy and len(h.nav.sent) == 2
    old_id, initial, old_callback = h.nav.sent[0]
    new_id, searched, _ = h.nav.sent[1]
    assert old_id != new_id and initial.object_reference is searched.object_reference
    assert searched.pose.distance_to(initial.pose) <= h.search.policy.radius_m
    payload = json.loads(navigation_payload(h.events[-1]))
    assert payload["destination"]["goal_kind"] == "object_search" and payload["search_attempt"] == 1
    old_callback(NavigationEvent("succeeded"))
    assert not h.commands.needs_observation
    old_callback(NavigationEvent("failed"))
    assert h.commands.busy
    h.arrive()
    h.observe()
    assert h.events[-1].state == "succeeded" and h.events[-1].search_attempt == 1


def test_search_is_opt_in_and_ambiguity_does_not_trigger_more_motion(tmp_path):
    h = Harness(tmp_path, search=False)
    h.start()
    h.arrive()
    h.observe(False)
    assert h.events[-1].state == "destination_unverified" and len(h.nav.sent) == 1
    h = Harness(tmp_path)
    h.comparator.result = "uncertain"
    h.start()
    h.arrive()
    h.observe()
    assert h.events[-1].state == "destination_ambiguous" and len(h.nav.sent) == 1
    assert not h.commands.busy


def test_search_has_a_fixed_viewpoint_budget(tmp_path):
    h = Harness(tmp_path, limit=1)
    h.start()
    h.arrive()
    h.observe(False)
    h.arrive()
    h.observe(False)
    assert len(h.nav.sent) == 2 and not h.commands.busy
    assert "viewpoint limit" in h.events[-1].message


@pytest.mark.parametrize("change", ["cancel", "localization", "delete", "blocked", "deadline"])
def test_changes_during_search_planning_cannot_send_another_goal(tmp_path, change):
    h = Harness(tmp_path)
    h.start()
    h.arrive()

    def change_state():
        if change == "cancel":
            h.commands.cancel()
        elif change == "localization":
            h.ready = False
        elif change == "delete":
            h.store.objects.delete(h.record.id)
        elif change == "deadline":
            h.mono = 100
        else:
            h.env.state = replace(h.env.state, costmap=replace(h.env.state.costmap, cells=np.full((200, 200), 254)))

    h.env.after_path = change_state
    h.observe(False)
    assert len(h.nav.sent) == 1 and not h.commands.busy


def test_search_timeout_cancels_the_active_transport_and_waits_for_confirmation(tmp_path):
    h = Harness(tmp_path)
    h.start()
    h.arrive()
    h.observe(False)
    request_id, _, callback = h.nav.sent[-1]
    h.mono = 61
    h.commands.poll()
    assert h.nav.canceled == [request_id] and h.commands.busy
    callback(NavigationEvent("uncertain"))
    assert h.commands.busy
    callback(NavigationEvent("canceled"))
    assert not h.commands.busy


def test_cancel_during_provider_work_cannot_succeed(tmp_path):
    h = Harness(tmp_path)
    h.start()
    h.arrive()
    h.comparator.after = lambda: h.commands.cancel()
    h.observe()
    assert not h.commands.busy and len(h.nav.sent) == 1
    assert h.events[-1].state == "canceled"


@pytest.mark.parametrize("change", ["error", "delete", "deadline", "localization"])
def test_unavailable_verification_cannot_succeed_or_start_search(tmp_path, change):
    h = Harness(tmp_path)
    h.start()
    h.arrive()

    def change_state():
        if change == "error":
            raise RuntimeError("provider unavailable")
        if change == "delete":
            h.store.objects.delete(h.record.id)
        elif change == "deadline":
            h.mono = 31
        else:
            h.ready = False

    h.comparator.after = change_state
    h.observe()
    assert not h.commands.busy and len(h.nav.sent) == 1
    assert h.events[-1].state == "destination_unverified" and h.events[-1].object_result == "unavailable"


def test_late_request_comparison_cannot_publish_an_object_match(tmp_path):
    h = Harness(tmp_path)
    h.start()
    h.arrive()

    class LateVerifier:
        def verify(self, *_):
            h.mono = 31
            return Matched().verify("printer", "image")

    h.resolver._verifier = LateVerifier()
    h.observe()
    assert h.events[-1].state == "destination_unverified" and h.events[-1].object_result == "unavailable"


def test_search_requires_another_fresh_capture_and_stops_waiting_at_deadline(tmp_path):
    h = Harness(tmp_path)
    h.start()
    h.arrive()
    h.observe(False)
    h.arrive()
    h.commands.observe(frame(tmp_path, h.now, h.nav.sent[-1][1].pose))
    assert not h.tasks and h.commands.needs_observation
    h.mono = 61
    h.commands.poll()
    assert not h.commands.busy and h.events[-1].state == "destination_unverified"
    assert not h.nav.canceled  # The robot has already stopped at the second viewpoint.


def test_search_transport_error_retains_ownership_until_cancel_is_confirmed(tmp_path):
    h = Harness(tmp_path)
    h.start()
    h.arrive()
    original_send = h.nav.send

    def uncertain_send(*args):
        original_send(*args)
        raise RuntimeError("lost acknowledgement")

    h.nav.send = uncertain_send
    h.observe(False)
    assert h.commands.busy and h.events[-1].state == "uncertain"
    h.commands.cancel()
    assert h.nav.canceled == [h.nav.sent[-1][0]] and h.commands.busy
    h.nav.sent[-1][2](NavigationEvent("canceled"))
    assert not h.commands.busy


def test_wrong_fresh_request_verdict_blocks_object_success(tmp_path):
    from tests.test_verification import Verifier

    h = Harness(tmp_path, search=False)
    h.start()
    h.arrive()
    h.resolver._verifier = Verifier(["not_matched"])
    h.observe()
    assert h.events[-1].state == "destination_unverified"


def test_context_failure_cannot_start_another_object_search_viewpoint(tmp_path, monkeypatch):
    from placecell import MissionContext

    h = Harness(tmp_path)
    context = MissionContext()
    h.commands._context = context
    record = context.record

    def fail_search(request_id, kind, payload):
        if payload.get("state") == "searching":
            raise OSError("disk full")
        record(request_id, kind, payload)

    monkeypatch.setattr(context, "record", fail_search)
    try:
        h.start()
        h.arrive()
        h.observe(False)
        assert not h.commands.busy and len(h.nav.sent) == 1
        assert h.events[-1].state == "destination_unverified"
    finally:
        context.close()
