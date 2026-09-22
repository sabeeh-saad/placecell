from __future__ import annotations

import json
import threading

import pytest

from placecell import NavigationCommands, NavigationEvent, NavigationUpdate, Observation, Pose
from placecell.errors import ValidationError
from placecell.operator import navigation_payload, parse_operator_command, snapshot_payload
from tests.conftest import embedded
from tests.test_missions import mission as mission  # pytest fixture
from tests.test_missions import start
from tests.test_navigation import make_commands


@pytest.mark.parametrize(
    ("envelope", "expected"),
    [
        (
            {"schema_version": 1, "command": "instruction", "text": "Visit printer then cupboard"},
            "Visit printer then cupboard",
        ),
        ({"schema_version": 1, "command": "stop"}, "stop"),
        ({"schema_version": 1, "command": "choose", "option": 2}, "option 2"),
    ],
)
def test_versioned_commands_route_through_existing_controller(envelope, expected):
    assert parse_operator_command(json.dumps(envelope)) == expected


@pytest.mark.parametrize(
    "payload",
    [
        "stop",
        "null",
        "[]",
        "{}",
        "x" * 16385,
        "[" * 1200 + "]" * 1200,
        '{"schema_version":1,"command":"stop","command":"instruction"}',
        '{"schema_version":true,"command":"stop"}',
        '{"schema_version":1.0,"command":"stop"}',
        '{"schema_version":2,"command":"stop"}',
        '{"schema_version":1,"command":"stop","text":"go to printer"}',
        '{"schema_version":1,"command":"choose","option":true}',
        '{"schema_version":1,"command":"choose","option":1.0}',
        '{"schema_version":1,"command":"choose","option":4}',
        '{"schema_version":1,"command":"choose","option":0}',
        '{"schema_version":1,"command":"instruction","text":null}',
        '{"schema_version":1,"command":"instruction","text":" "}',
        '{"schema_version":1,"command":"instruction"}',
        '{"schema_version":1,"command":{}}',
        json.dumps({"schema_version": 1, "command": "instruction", "text": "x" * 2001}),
    ],
)
def test_invalid_envelopes_are_rejected_without_text_fallback(payload):
    with pytest.raises(ValidationError):
        parse_operator_command(payload)


def test_snapshot_is_read_only_across_planning_and_goal_transitions(mission):
    m = mission
    initial = m.commands.snapshot()
    assert initial.status.state == "idle" and not initial.busy and initial.sequence == 0
    m.commands.handle("Visit printer then cupboard")
    planning = m.commands.snapshot()
    assert planning.busy and planning.status.state == "planning" and planning.status.mission_step == 0
    assert planning.active_request_id == planning.status.request_id
    for _ in range(3):
        assert m.commands.snapshot() == planning
    assert len(m.tasks) == 1 and not m.model.calls and not m.nav.sent
    m.tasks.pop(0)()
    first = m.commands.snapshot()
    assert first.status.mission_step == 1 and first.status.destination.label == "printer"
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    second = m.commands.snapshot()
    assert second.busy and second.status.state == "resolving" and second.status.mission_step == 2
    assert second.active_request_id != first.active_request_id
    assert second.status.destination is None
    m.tasks.pop(0)()
    m.nav.sent[1][2](NavigationEvent("succeeded"))
    final = m.commands.snapshot()
    assert not final.busy and final.active_request_id is None
    assert final.status.mission_destinations == ("printer", "cupboard")
    assert final.status.state == "succeeded" and final.status.mission_step == 2
    assert final.status.instance_id == initial.status.instance_id
    assert [event.sequence for event in m.events] == list(range(1, len(m.events) + 1))
    # A late result and an idle stop must not erase the retained terminal outcome.
    m.nav.sent[0][2](NavigationEvent("failed"))
    m.commands.handle("stop")
    assert m.commands.snapshot().status == final.status


def test_rejected_commands_do_not_replace_active_state(mission):
    m = mission
    start(m)
    m.nav.sent[0][2](NavigationEvent("navigating", distance_remaining=2.5))
    before = m.commands.snapshot()
    m.commands.handle("Visit the cupboard")
    m.commands.reject_command("Unknown version")
    after = m.commands.snapshot()
    assert after.status == before.status and after.busy
    assert after.sequence == before.sequence + 2
    assert m.events[-2].state == "busy" and m.events[-1].state == "invalid"
    payload = json.loads(snapshot_payload(after))
    assert payload["schema_version"] == 1 and payload["type"] == "mission_snapshot"
    assert payload["status"]["distance_remaining"] == 2.5
    assert payload["status"]["sequence"] < payload["sequence"]


def test_ambiguous_mission_survives_rejected_commands_then_expires(mission, hashing):
    m = mission
    del m.resolver._places["printer"]
    m.store.upsert(
        [embedded(hashing, "printer", t=t, pose=Pose(x, 2, map_id="office")) for t, x in ((1000, 1), (1001, 8))]
    )
    start(m)
    ambiguous = m.commands.snapshot()
    assert ambiguous.busy and ambiguous.active_request_id is None and ambiguous.choice_remaining_s == 30
    assert len(ambiguous.status.choices) == 2
    m.commands.handle("Visit another destination")
    m.commands.handle("option three")
    assert m.commands.snapshot().status == ambiguous.status
    m.now[0] += 30
    # At the deadline, reads and selections agree; the next poll records expiry.
    assert not json.loads(snapshot_payload(m.commands.snapshot()))["awaiting_choice"]
    m.commands.handle("option two")
    assert not m.tasks and m.events[-1].state == "invalid"
    m.commands.poll()
    expired = m.commands.snapshot()
    assert expired.status.state == "not_found" and not expired.busy
    assert not expired.status.choices and expired.choice_remaining_s is None


@pytest.mark.parametrize("action", ["stop", "expire"])
def test_legacy_single_goal_choices_are_cleared_in_the_snapshot(store, hashing, action, monkeypatch):
    monkeypatch.setattr("placecell.navigation.data_url", lambda uri: "data:image/jpeg;base64,YQ==")
    store.upsert([embedded(hashing, "printer", t=t, x=x) for t, x in ((1000, 1), (1001, 8))])
    commands, nav, tasks, _ = make_commands(store, hashing)
    now = [0.0]
    commands._clock = lambda: now[0]
    commands.handle("go to printer")
    tasks.pop()()
    assert commands.snapshot().status.state == "ambiguous"
    if action == "stop":
        commands.handle("stop")
    else:
        now[0] = 30
        commands.poll()
    snapshot = commands.snapshot()
    assert snapshot.status.state == ("canceled" if action == "stop" else "not_found")
    assert snapshot.choice_remaining_s is None and not snapshot.status.choices and not nav.sent


def test_cancel_snapshot_retains_uncertain_ownership_and_never_advances(mission):
    m = mission
    start(m)
    m.commands.handle(parse_operator_command('{"schema_version":1,"command":"stop"}'))
    assert m.commands.snapshot().status.state == "canceling" and m.commands.snapshot().busy
    m.nav.sent[0][2](NavigationEvent("uncertain", "No cancel acknowledgement"))
    assert m.commands.snapshot().status.state == "uncertain" and m.commands.snapshot().busy
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    final = m.commands.snapshot()
    assert final.status.state == "canceled" and not final.busy and not m.tasks
    m.commands.close()
    assert m.commands.snapshot().closed and m.commands.snapshot().status == final.status


def test_arrival_snapshot_exposes_verification_and_terminal_evidence(mission, hashing):
    m = mission
    del m.resolver._places["printer"]
    row = embedded(hashing, "printer", pose=Pose(1, 2, map_id="office"))
    m.store.upsert([row])
    start(m)
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    assert m.commands.snapshot().status.state == "awaiting_observation"
    m.now[0] += 1
    m.commands.observe(Observation("r1", "front", m.now[0], row.pose, row.evidence, localization_checked=True))
    pending = m.commands.snapshot()
    assert pending.status.state == "verifying_arrival" and pending.busy
    assert pending.status.destination.memory.id == row.id
    m.commands.cancel()
    canceled = m.commands.snapshot()
    m.tasks.pop(0)()
    assert canceled.status.state == "canceled" and m.commands.snapshot() == canceled


def test_snapshot_does_not_wait_for_planning_or_replay_it(mission):
    m = mission
    entered, release = threading.Event(), threading.Event()

    def wait_for_release():
        entered.set()
        assert release.wait(5)

    m.model.before_reply = wait_for_release
    m.commands.handle("Visit the printer")
    worker = threading.Thread(target=m.tasks.pop(0))
    worker.start()
    try:
        assert entered.wait(5)
        assert m.commands.snapshot().status.state == "planning"
        m.commands.cancel()
        assert m.commands.snapshot().status.state == "canceled"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not m.nav.sent


def test_restart_has_a_new_instance_and_does_not_replay_context(mission):
    m = mission
    start(m)
    before = m.commands.snapshot()
    restarted = NavigationCommands(
        m.resolver,
        m.nav,
        lambda f: m.tasks.append(f) is None,
        m.events.append,
        mission_planner=m.commands._mission_planner,
        mission_context=m.context,
    )
    after = restarted.snapshot()
    assert after.status.instance_id != before.status.instance_id
    assert after.status.state == "idle" and after.status.mission_id == ""
    assert after.sequence == 0 and not after.busy and not m.tasks and len(m.nav.sent) == 1


def test_context_failure_on_stop_does_not_hide_pending_cancellation(mission, monkeypatch):
    m = mission
    start(m)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(m.context, "record", broken)
    m.commands.handle("stop")
    assert m.commands.snapshot().status.state == "canceling" and m.commands.snapshot().busy
    m.nav.sent[0][2](NavigationEvent("canceled"))
    m.commands.handle("Visit printer")
    assert m.commands.snapshot().status.state == "unavailable" and not m.commands.snapshot().busy


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -float("inf")])
def test_status_uses_valid_json_for_unknown_distance(distance):
    value = json.loads(navigation_payload(NavigationUpdate("r", "navigating", "", distance_remaining=distance)))
    assert value["schema_version"] == 1 and value["distance_remaining"] is None
