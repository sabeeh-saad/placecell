"""Contracts reproduced by the full Gazebo checkpoint, with deterministic scheduling."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from placecell import CollectionInfo, InMemoryStore, ObjectTracker, Pose
from placecell.object_types import Detection
from placecell.objects import ObjectPolicy
from placecell.tracing import TraceStore, _Barrier, _QueuedEvent, _TraceQueue, read_trace
from tests.test_nav2 import Handle, started
from tests.test_object_navigation import Harness, frame
from tests.test_objects import LEFT, RIGHT, Detector, PixelEmbedder, ingest, observation


@pytest.mark.parametrize("fault", ["missing", "uncertain", "invalid_pixels", "unlocalized"])
def test_rgbd_identity_waits_for_usable_geometry_without_consuming_scan_interval(tmp_path, fault):
    embedder, detector = PixelEmbedder(), Detector()
    store = InMemoryStore(CollectionInfo("geometry", embedder.model_name, 3))
    tracker = ObjectTracker(store, embedder, detector, ObjectPolicy(require_position=True))
    first = ingest(tracker, observation(tmp_path))[0]
    obs = replace(observation(tmp_path, 2000), pose=Pose(1, 0, map_id="office-v1"))
    if fault == "missing":
        obs = replace(obs, depth=None)
    elif fault == "uncertain":
        obs = replace(obs, depth=replace(obs.depth, position_error_m=1))
    elif fault == "invalid_pixels":
        obs = replace(obs, depth=replace(obs.depth, data=observation(tmp_path, 1999, (), (), background=0).depth.data))
    else:
        obs = replace(obs, localization_checked=False)
    assert ingest(tracker, obs) == [first]
    # The complete capture immediately afterwards can reinforce the original identity.
    after = ingest(tracker, replace(observation(tmp_path, 2000.2), pose=obs.pose))
    assert len(after) == 1 and after[0].id == first.id and after[0].last_seen == 2000.2


def test_rgbd_geometry_requirement_preserves_two_real_lookalikes(tmp_path):
    embedder, detector = PixelEmbedder(), Detector()
    store = InMemoryStore(CollectionInfo("lookalikes", embedder.model_name, 3))
    tracker = ObjectTracker(store, embedder, detector, ObjectPolicy(require_position=True))
    detector.detections = [Detection("printer", "red printer", box) for box in (LEFT, RIGHT)]
    records = ingest(tracker, observation(tmp_path, boxes=(LEFT, RIGHT), colors=("red", "red")))
    assert len(records) == 2 and records[0].position.distance(records[1].position) > 0.5
    # No merge by category or identical appearance, including on subsequent complete views.
    assert {r.id for r in ingest(tracker, observation(tmp_path, 2000, (LEFT, RIGHT), ("red", "red")))} == {
        r.id for r in records
    }


def test_object_arrival_waits_for_complete_capture_then_verifies(tmp_path):
    h = Harness(tmp_path, search=False)
    h.start()
    h.arrive()
    h.now += 1
    obs = frame(tmp_path, h.now, h.nav.sent[-1][1].pose)
    h.commands.observe(replace(obs, depth=None))
    assert h.commands.needs_observation and not h.tasks
    h.observe()
    assert h.events[-1].state == "succeeded" and h.events[-1].object_result == "matched"


def test_incomplete_arrival_frames_do_not_extend_deadline(tmp_path):
    h = Harness(tmp_path, search=False)
    h.start()
    h.arrive()
    for _ in range(31):
        h.now += 1
        h.mono += 1
        h.commands.observe(replace(frame(tmp_path, h.now, h.nav.sent[-1][1].pose), depth=None))
        h.commands.poll()
    assert not h.tasks and not h.commands.busy
    assert h.events[-1].state == "destination_unverified"


def test_feedback_burst_is_bounded_without_delaying_cancel_or_terminal_result():
    navigator, client, events, clock = started()
    handle = Handle()
    client.response.set_result(handle)
    for index in range(1000):
        client.feedback(SimpleNamespace(feedback=SimpleNamespace(distance_remaining=index / 1000)))
    assert len([e for e in events if e.distance_remaining is not None]) == 1
    clock[0] = 0.21
    client.feedback(SimpleNamespace(feedback=SimpleNamespace(distance_remaining=0.5)))
    assert events[-1].distance_remaining == 0.5
    navigator.cancel("request")
    assert handle.cancel_calls == 1 and events[-1].state == "canceling"
    handle.result.set_result(SimpleNamespace(status=5, result=SimpleNamespace()))
    assert events[-1].state == "canceled"


def test_progress_flood_cannot_displace_dispatch_cancel_or_result(tmp_path, monkeypatch):
    traces = TraceStore(tmp_path / "traces.sqlite3", queue_size=4)
    entered, release = threading.Event(), threading.Event()
    original = traces._write

    def block(item):
        entered.set()
        assert release.wait(5)
        original(item)

    monkeypatch.setattr(traces, "_write", block)
    context = traces.context("mission", "request")
    try:
        context.emit("block")
        assert entered.wait(2)
        for index in range(1000):
            context.emit("nav2.event", state="navigating", message="", distance_remaining=index / 1000)
            context.emit("status", state="navigating", message="", distance_remaining=index / 1000)
        for stage in ("nav2.dispatch", "nav2.cancel_requested", "nav2.result", "status"):
            context.emit(stage, state="canceled", message="terminal evidence")
        assert traces.health()["pending"] <= 4
        assert traces.health()["dropped_critical_events"] == 0
    finally:
        release.set()
        assert traces.close(timeout=5)
    report = read_trace(traces.path)
    assert {e["stage"] for e in report["events"]} >= {"nav2.dispatch", "nav2.cancel_requested", "nav2.result", "status"}
    assert report["health"]["coalesced_events"] >= 1998
    assert report["health"]["dropped_critical_events"] == 0


def test_progress_coalescing_never_crosses_mission_step(tmp_path, monkeypatch):
    traces = TraceStore(tmp_path / "steps.sqlite3", queue_size=8)
    entered, release = threading.Event(), threading.Event()
    original = traces._write

    def block(item):
        entered.set()
        assert release.wait(5)
        original(item)

    monkeypatch.setattr(traces, "_write", block)
    try:
        traces.context("mission", "one", 1).emit("block")
        assert entered.wait(2)
        for request, step in (("one", 1), ("two", 2)):
            for distance in (2.0, 1.0):
                traces.context("mission", request, step).emit(
                    "status", state="navigating", message="", distance_remaining=distance
                )
    finally:
        release.set()
        assert traces.close(timeout=5)
    report = read_trace(traces.path)
    assert report["summary"]["loss_counters_nonzero"] == []
    assert report["health"]["coalesced_events"] == 2
    events = [e for e in report["events"] if e["stage"] == "status"]
    assert [(e["request_id"], e["step"], e["data"]["distance_remaining"]) for e in events] == [
        ("one", 1, 1.0),
        ("two", 2, 1.0),
    ]


def test_progress_coalescing_preserves_flush_boundary_and_queue_accounting():
    pending = _TraceQueue(3)
    key = ("mission", "request", 1, "status")
    first, latest = _QueuedEvent(("mission", "before"), key), _QueuedEvent(("mission", "after"), key)
    barrier = _Barrier()
    assert pending.offer(first) == "added"
    pending.put_nowait(barrier)
    assert pending.offer(latest) == "added"
    critical = _QueuedEvent(("mission", "cancel"))
    assert pending.offer(critical) == "evicted_progress"
    assert pending.offer(_QueuedEvent(("mission", "result"))) == "evicted_progress"
    # The two critical events retain their order and cannot evict the flush barrier.
    assert pending.offer(_QueuedEvent(("mission", "extra"))) == "dropped"
    assert pending.get_nowait() is barrier
    pending.task_done()
    assert pending.get_nowait() is critical
    pending.task_done()
    assert pending.get_nowait().row == ("mission", "result")
    pending.task_done()
    assert pending.unfinished_tasks == 0


def test_coalesced_progress_stays_after_intervening_critical_event():
    pending = _TraceQueue(3)
    key = ("mission", "request", 1, "status")
    pending.offer(_QueuedEvent(("mission", "navigating"), key))
    cancel = _QueuedEvent(("mission", "cancel_requested"))
    pending.offer(cancel)
    latest = _QueuedEvent(("mission", "canceling"), key)
    assert pending.offer(latest) == "coalesced"
    assert pending.get_nowait() is cancel
    pending.task_done()
    assert pending.get_nowait() is latest
    pending.task_done()
    assert pending.unfinished_tasks == 0


def test_critical_only_saturation_is_reported_without_blocking_callbacks(tmp_path, monkeypatch):
    traces = TraceStore(tmp_path / "full.sqlite3", queue_size=1)
    entered, release = threading.Event(), threading.Event()
    original = traces._write

    def block(item):
        entered.set()
        assert release.wait(5)
        original(item)

    monkeypatch.setattr(traces, "_write", block)
    context = traces.context("mission", "request")
    try:
        context.emit("block")
        assert entered.wait(2)
        context.emit("nav2.dispatch")
        context.emit("nav2.result")
        assert traces.health()["dropped_events"] == traces.health()["dropped_critical_events"] == 1
    finally:
        release.set()
        assert traces.close(timeout=5)
    assert read_trace(traces.path)["health"]["dropped_critical_events"] == 1
