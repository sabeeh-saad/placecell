"""Deterministic interleavings at the controller/transport ownership boundary."""

from concurrent.futures import Future
from threading import Event, Thread
from types import SimpleNamespace as Obj

import pytest

from placecell import Destination, NavigationEvent, Pose
from placecell.ros2.navigation import Nav2Navigator
from tests.test_missions import mission as mission
from tests.test_missions import start
from tests.test_nav2 import Client, Handle


@pytest.mark.parametrize("deadline", ["response", "trip"])
def test_terminal_success_carries_intent_before_timeout_callback_reaches_controller(mission, monkeypatch, deadline):
    m = mission
    client = Client()
    nav = Nav2Navigator(client, lambda p: p, clock=lambda: m.now[0], response_timeout_s=2, trip_timeout_s=10)
    m.commands._navigator = nav
    start(m)
    handle = Handle()
    if deadline == "trip":
        client.response.set_result(handle)
    entered, release = Event(), Event()
    emit = nav._emit

    def delayed(trip, event):
        if event.message.startswith(("Navigation time limit", "Nav2 has not acknowledged")):
            entered.set()
            assert release.wait(3)
        emit(trip, event)

    monkeypatch.setattr(nav, "_emit", delayed)
    m.now[0] += 11
    worker = Thread(target=nav.poll)
    worker.start()
    try:
        assert entered.wait(2)
        handle.result.set_result(Obj(status=4, result=Obj()))
        if deadline == "response":
            client.response.set_result(handle)
        assert m.events[-1].state == "canceled"
        assert not m.tasks and not m.commands.busy
        assert not any(e.state == "step_succeeded" for e in m.events)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()


def test_cancel_is_issued_before_slow_status_persistence(mission, monkeypatch):
    m = mission
    start(m)
    record = m.context.record
    issued = []

    def slow_status(request_id, kind, payload):
        if payload.get("state") == "canceling":
            issued.append(bool(m.nav.canceled))
        record(request_id, kind, payload)

    monkeypatch.setattr(m.context, "record", slow_status)
    m.commands.cancel()
    assert m.nav.canceled and m.commands.busy
    assert issued and all(issued), "transport cancel must precede status persistence"


def test_synchronous_cancel_result_is_not_overwritten_by_canceling_status(mission, monkeypatch):
    m = mission
    start(m)
    monkeypatch.setattr(m.nav, "cancel", lambda _: m.nav.sent[0][2](NavigationEvent("canceled")))
    m.commands.cancel()
    assert m.events[-1].state == "canceled" and not m.commands.busy


def test_stale_result_error_cannot_cancel_new_trip_with_reused_transport_id():
    client, events = Client(), []
    nav = Nav2Navigator(client, lambda p: p)
    destination = Destination("printer", Pose(1, 2), "named_place")
    nav.send("same", destination, events.append)
    old_trip = nav._trip
    first = Handle()
    client.response.set_result(first)
    first.result.set_result(Obj(status=4, result=Obj()))
    client.response = Future()
    nav.send("same", destination, events.append)
    second = Handle()
    client.response.set_result(second)
    failed = Future()
    failed.set_exception(OSError("delayed old callback"))
    nav._result(old_trip, failed)
    assert second.cancel_calls == 0
    assert events[-1].state == "navigating"


def test_transport_cancel_precedes_a_blocked_controller_callback():
    client = Client()
    entered, release = Event(), Event()

    def callback(event):
        if event.state == "canceling":
            entered.set()
            assert release.wait(3)

    nav = Nav2Navigator(client, lambda p: p)
    nav.send("r", Destination("printer", Pose(1, 2), "named_place"), callback)
    handle = Handle()
    client.response.set_result(handle)
    worker = Thread(target=lambda: nav.cancel("r"))
    worker.start()
    try:
        assert entered.wait(2)
        assert handle.cancel_calls == 1
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()


def test_duplicate_acceptance_cannot_reregister_or_cancel_a_finished_goal(monkeypatch):
    client, events = Client(), []
    nav = Nav2Navigator(client, lambda p: p)
    nav.send("r", Destination("printer", Pose(1, 2), "named_place"), events.append)
    trip, handle = nav._trip, Handle()
    calls = []
    monkeypatch.setattr(handle, "get_result_async", lambda: calls.append(1) or handle.result)
    client.response.set_result(handle)
    nav._accepted(trip, client.response)
    handle.result.set_result(Obj(status=4, result=Obj()))
    nav._accepted(trip, client.response)
    assert len(calls) == 1 and handle.cancel_calls == 0
    assert [e.state for e in events] == ["navigating", "succeeded"]
