from __future__ import annotations

import math
import sys
from concurrent.futures import Future
from types import SimpleNamespace as Obj

import pytest

from placecell import Destination, Pose
from placecell.errors import ValidationError
from placecell.ros2.navigation import Nav2Navigator, create_navigator


class Handle:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result = Future()
        self.cancel_result = Future()
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancel_result


class Client:
    def __init__(self, ready=True):
        self.ready = ready
        self.response = Future()
        self.goals = []
        self.feedback = None

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal, feedback_callback):
        self.goals.append(goal)
        self.feedback = feedback_callback
        return self.response


def started(ready=True):
    client, events, clock = Client(ready), [], [0.0]
    navigator = Nav2Navigator(
        client, lambda pose: pose, clock=lambda: clock[0], response_timeout_s=5, trip_timeout_s=20
    )
    destination = Destination("printer", Pose(1, 2), "coordinates")
    navigator.send("request", destination, events.append)
    return navigator, client, events, clock


@pytest.mark.parametrize("status,expected", [(4, "succeeded"), (5, "canceled"), (6, "failed"), (0, "uncertain")])
def test_goal_feedback_and_terminal_status(status, expected):
    navigator, client, events, _clock = started()
    handle = Handle()
    client.response.set_result(handle)
    assert events[-1].state == "navigating" and client.goals == [Pose(1, 2)]
    client.feedback(Obj(feedback=Obj(distance_remaining=2.0)))
    assert events[-1].distance_remaining == 2
    handle.result.set_result(Obj(status=status, result=Obj()))  # Humble result has no error_code
    assert events[-1].state == expected
    before = len(events)
    if status in (4, 5, 6):
        client.feedback(Obj(feedback=Obj(distance_remaining=0)))
        navigator.cancel("request")
        navigator.poll()
        assert len(events) == before


def test_unavailable_rejected_and_result_error_code():
    _navigator, client, events, _clock = started(False)
    assert not client.goals and events[-1].state == "unavailable"
    _navigator, client, events, _clock = started()
    client.response.set_result(Handle(accepted=False))
    assert events[-1].state == "rejected"
    _navigator, client, events, _clock = started()
    handle = Handle()
    client.response.set_result(handle)
    handle.result.set_result(Obj(status=4, result=Obj(error_code=3, error_msg="path failed")))
    assert events[-1].state == "failed" and "path failed" in events[-1].message


def test_cancel_before_acceptance_and_cancel_acknowledgement_is_not_completion():
    navigator, client, events, _clock = started()
    navigator.cancel("other")
    navigator.cancel("request")
    handle = Handle()
    client.response.set_result(handle)
    assert handle.cancel_calls == 1 and events[-1].state == "canceling"
    navigator.cancel("request")
    assert handle.cancel_calls == 1
    handle.cancel_result.set_result(Obj(goals_canceling=["request"]))
    assert all(e.state != "canceled" for e in events)
    client.feedback(Obj(feedback=Obj(distance_remaining=1)))
    assert events[-1].state == "canceling"
    handle.result.set_result(Obj(status=5, result=Obj()))
    assert events[-1].state == "canceled"


def test_cancel_rejection_and_timeout_keep_the_trip_owned():
    navigator, client, events, clock = started()
    handle = Handle()
    client.response.set_result(handle)
    navigator.cancel("request")
    handle.cancel_result.set_result(Obj(goals_canceling=[]))
    assert events[-1].state == "cancel_failed"
    with pytest.raises(ValidationError):
        navigator.send("new", Destination("other", Pose(0, 0), "coordinates"), events.append)
    clock[0] = 6
    navigator.poll()
    assert any(e.state == "uncertain" for e in events)


def test_late_acceptance_after_response_timeout_is_canceled():
    navigator, client, events, clock = started()
    clock[0] = 6
    navigator.poll()
    assert events[-1].state == "uncertain"
    count = len(events)
    navigator.poll()
    assert len(events) == count
    handle = Handle()
    client.response.set_result(handle)
    assert handle.cancel_calls == 1 and events[-1].state == "canceling"


def test_trip_deadline_requests_cancellation_and_waits_for_result():
    navigator, client, events, clock = started()
    handle = Handle()
    client.response.set_result(handle)
    clock[0] = 21
    navigator.poll()
    assert handle.cancel_calls == 1
    assert events[-1].state == "canceling"
    handle.cancel_result.set_result(Obj(goals_canceling=["request"]))
    clock[0] = 27
    navigator.poll()
    assert events[-1].state == "uncertain"
    handle.result.set_result(Obj(status=5, result=Obj()))
    assert events[-1].state == "canceled"


@pytest.mark.parametrize("stage", ["submit", "accept", "result", "cancel", "cancel_response", "get_result"])
def test_transport_errors_never_report_success_or_release_uncertain_goals(stage, monkeypatch):
    client, events = Client(), []
    navigator = Nav2Navigator(client, lambda pose: pose)
    if stage == "submit":
        monkeypatch.setattr(client, "send_goal_async", lambda *a, **k: (_ for _ in ()).throw(OSError("transport")))
    navigator.send("r", Destination("printer", Pose(1, 2), "coordinates"), events.append)
    handle = Handle()
    if stage == "accept":
        client.response.set_exception(OSError("transport"))
    elif stage != "submit":
        if stage == "get_result":
            monkeypatch.setattr(handle, "get_result_async", lambda: (_ for _ in ()).throw(OSError("transport")))
        client.response.set_result(handle)
        if stage == "result":
            handle.result.set_exception(OSError("transport"))
        elif stage in ("cancel", "cancel_response"):
            if stage == "cancel":
                monkeypatch.setattr(handle, "cancel_goal_async", lambda: (_ for _ in ()).throw(OSError("transport")))
            navigator.cancel("r")
            if stage == "cancel_response":
                handle.cancel_result.set_exception(OSError("transport"))
    assert any(e.state == "uncertain" for e in events)
    assert not any(e.state in ("succeeded", "canceled", "failed") for e in events)
    with pytest.raises(ValidationError):
        navigator.send("new", Destination("printer", Pose(1, 2), "coordinates"), events.append)


def test_invalid_feedback_is_ignored_and_preparation_failure_sends_nothing():
    navigator, client, events, _clock = started()
    before = len(events)
    for message in (
        Obj(),
        Obj(feedback=Obj(distance_remaining="bad")),
        Obj(feedback=Obj(distance_remaining=float("nan"))),
        Obj(feedback=Obj(distance_remaining=-1)),
    ):
        client.feedback(message)
    assert len(events) == before
    other_events, other_client = [], Client()
    navigator = Nav2Navigator(other_client, lambda p: (_ for _ in ()).throw(ValueError("bad goal")))
    navigator.send("r", Destination("printer", Pose(1, 2), "coordinates"), other_events.append)
    assert not other_client.goals and other_events[-1].state == "unavailable"
    with pytest.raises(ValidationError):
        Nav2Navigator(client, lambda pose: pose, trip_timeout_s=float("nan"))


def test_nav2_factory_stamps_the_pose_and_preserves_heading(monkeypatch):
    client = Client()
    action_type = Obj(Goal=lambda: Obj(pose=Obj(header=Obj(), pose=Obj(position=Obj(), orientation=Obj()))))
    monkeypatch.setitem(sys.modules, "nav2_msgs.action", Obj(NavigateToPose=action_type))

    def action_client(node, action, name, **kwargs):
        assert action is action_type and name == "robot/navigate_to_pose"
        return client

    monkeypatch.setitem(sys.modules, "rclpy.action", Obj(ActionClient=action_client))
    monkeypatch.setitem(sys.modules, "rclpy.callback_groups", Obj(ReentrantCallbackGroup=object))
    node = Obj(get_clock=lambda: Obj(now=lambda: Obj(to_msg=lambda: "stamp")))
    navigator = create_navigator(node, "robot/navigate_to_pose", 10, 600)
    navigator.send(
        "r", Destination("printer", Pose(1, 2, math.pi / 2, frame_id="office_map"), "coordinates"), lambda e: None
    )
    goal = client.goals[0]
    assert goal.pose.header.frame_id == "office_map" and goal.pose.header.stamp == "stamp"
    assert goal.pose.pose.position.x == 1 and goal.pose.pose.position.y == 2
    assert goal.pose.pose.orientation.z == pytest.approx(2**-0.5)
    assert goal.pose.pose.orientation.w == pytest.approx(2**-0.5)
