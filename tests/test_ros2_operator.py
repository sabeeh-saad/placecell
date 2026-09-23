from __future__ import annotations

import json
import sqlite3
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from placecell import NavigationEvent, Pose
from placecell.command_identity import CommandJournal
from placecell.ros2.operator import OperatorInterface
from tests.conftest import embedded
from tests.test_command_identity import SCOPE, envelope
from tests.test_missions import mission as mission
from tests.test_missions import proposal, review


@pytest.fixture
def ros(monkeypatch):
    for name in ("rclpy.clock", "rclpy.qos", "rclpy.callback_groups", "std_msgs.msg", "std_srvs.srv"):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["rclpy.clock"].Clock = lambda **kw: kw
    sys.modules["rclpy.callback_groups"].MutuallyExclusiveCallbackGroup = object
    sys.modules["rclpy.clock"].ClockType = SimpleNamespace(STEADY_TIME="steady")
    sys.modules["rclpy.qos"].QoSProfile = lambda **kw: SimpleNamespace(**kw)
    sys.modules["rclpy.qos"].DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL="retained", VOLATILE="volatile")
    sys.modules["rclpy.qos"].ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable")
    sys.modules["std_msgs.msg"].String = lambda **kw: SimpleNamespace(**kw)
    sys.modules["std_srvs.srv"].Trigger = object
    publishers, subscriptions, services, timers = {}, {}, {}, []

    def create_publisher(cls, topic, qos):
        messages = []
        pub = SimpleNamespace(publish=lambda msg: messages.append(json.loads(msg.data)), messages=messages, qos=qos)
        publishers[topic] = pub
        return pub

    def create_subscription(cls, topic, callback, qos, **kwargs):
        subscriptions[topic] = SimpleNamespace(callback=callback, qos=qos, **kwargs)

    node = SimpleNamespace(
        create_publisher=create_publisher,
        create_subscription=create_subscription,
        create_service=lambda cls, name, cb: services.__setitem__(name, cb),
        create_timer=lambda period, cb, **kw: timers.append((period, cb, kw)),
    )
    return SimpleNamespace(
        node=node, publishers=publishers, subscriptions=subscriptions, services=services, timers=timers
    )


def read_service(ros):
    result = ros.services["~/get_mission_snapshot"](None, SimpleNamespace())
    assert result.success
    return json.loads(result.message)


def test_disabled_interface_reports_state_and_rejects_envelopes(ros):
    bridge = OperatorInterface(ros.node, None)
    initial = read_service(ros)
    assert initial["status"]["state"] == "disabled" and not initial["navigation_enabled"]
    ros.subscriptions["~/command_json"].callback(SimpleNamespace(data='{"schema_version":2,"command":"stop"}'))
    ros.subscriptions["~/command_json"].callback(SimpleNamespace(data='{"schema_version":1,"command":"stop"}'))
    ros.subscriptions["~/command"].callback(SimpleNamespace(data="go to printer"))
    events = ros.publishers["~/navigation_status"].messages
    assert [event["state"] for event in events] == ["invalid", "disabled", "disabled"]
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert all(event["instance_id"] == initial["instance_id"] for event in events)
    bridge.publish_snapshot()
    latest = read_service(ros)
    assert latest["status"] == initial["status"] and latest["sequence"] == 3
    assert ros.publishers["~/mission_snapshot"].messages[-1]["sequence"] == 3


def test_interface_preserves_volatile_commands_and_retains_only_snapshots(ros, mission):
    m = mission
    bridge = OperatorInterface(ros.node, m.commands)
    m.commands._publish_callback = bridge.publish
    for topic in ("~/command", "~/command_json"):
        assert ros.subscriptions[topic].qos.depth == 1
        assert ros.subscriptions[topic].qos.durability == "volatile"
    assert ros.publishers["~/navigation_status"].qos == 10
    snapshot_qos = ros.publishers["~/mission_snapshot"].qos
    assert snapshot_qos.depth == 1 and snapshot_qos.durability == "retained"
    assert ros.timers[0][2]["clock"]["clock_type"] == "steady"
    ros.subscriptions["~/command_json"].callback(
        SimpleNamespace(data='{"schema_version":1,"command":"instruction","text":"Visit printer then cupboard"}')
    )
    assert read_service(ros)["status"]["state"] == "planning"
    assert len(m.tasks) == 1 and not m.nav.sent
    m.tasks.pop(0)()
    ros.subscriptions["~/command_json"].callback(SimpleNamespace(data="go to cupboard"))
    assert ros.publishers["~/navigation_status"].messages[-1]["state"] == "invalid"
    assert read_service(ros)["status"]["state"] == "submitting" and len(m.nav.sent) == 1
    ros.subscriptions["~/command_json"].callback(SimpleNamespace(data='{"schema_version":1,"command":"stop"}'))
    assert read_service(ros)["status"]["state"] == "canceling" and m.nav.canceled
    m.nav.sent[0][2](NavigationEvent("canceled"))
    ros.timers[0][1]()
    terminal = ros.publishers["~/mission_snapshot"].messages[-1]
    assert terminal["status"]["state"] == "canceled" and not terminal["busy"]
    assert not m.tasks


def test_service_does_not_add_status_events_or_invoke_navigation(ros, mission):
    m = mission
    bridge = OperatorInterface(ros.node, m.commands)
    m.commands._publish_callback = bridge.publish
    m.commands.handle("Visit printer then cupboard")
    m.tasks.pop(0)()
    count = len(ros.publishers["~/navigation_status"].messages)
    for _ in range(3):
        assert read_service(ros)["busy"]
        bridge.publish_snapshot()
    assert len(ros.publishers["~/navigation_status"].messages) == count
    assert len(m.nav.sent) == 1 and not m.nav.canceled and not m.tasks


@pytest.fixture
def identified(ros, mission, tmp_path):
    journal = CommandJournal(tmp_path / "commands.db", SCOPE, clock=lambda: 1000)
    bridge = OperatorInterface(ros.node, mission.commands, journal=journal)
    mission.commands._publish_callback = bridge.publish
    yield SimpleNamespace(bridge=bridge, journal=journal, receipts=ros.publishers["~/command_receipt"].messages)
    journal.close()


def send(identified, **changes):
    value = envelope(**changes)
    if value["command"] != "instruction":
        value.pop("text")
    identified.bridge._on_json(SimpleNamespace(data=json.dumps(value)))
    return identified.receipts[-1]


def finish(m):
    m.nav.sent[-1][2](NavigationEvent("succeeded"))
    m.tasks.pop(0)()
    m.nav.sent[-1][2](NavigationEvent("succeeded"))


def test_retries_before_during_and_after_terminal_do_not_repeat_mission(ros, mission, identified):
    m, i = mission, identified
    first = send(i)
    assert first["disposition"] == "recorded"
    assert first["request_id"] == read_service(ros)["status"]["request_id"]
    assert send(i)["disposition"] == "duplicate" and len(m.tasks) == 1
    m.tasks.pop(0)()
    assert send(i)["disposition"] == "duplicate" and len(m.nav.sent) == 1
    assert send(i, text="different")["disposition"] == "conflict"
    finish(m)
    terminal = m.commands.snapshot()
    assert send(i)["disposition"] == "duplicate" and m.commands.snapshot() == terminal
    assert len(m.nav.sent) == 2 and len(m.model.calls) == len(m.critic.calls) == 1
    m.model.replies.append(proposal())
    m.critic.replies.append(review())
    assert send(i, command_id="deliberate-repeat")["disposition"] == "recorded"
    m.tasks.pop(0)()
    finish(m)
    assert len(m.nav.sent) == 4 and len(m.model.calls) == 2
    assert read_service(ros)["command_identity"]["durable"]


def test_busy_refusal_is_consumed_and_cannot_become_later_work(ros, mission, identified):
    first = send(identified)
    refused = send(identified, command_id="busy-request")
    assert refused["disposition"] == "recorded"
    assert ros.publishers["~/navigation_status"].messages[-1]["state"] == "busy"
    assert mission.commands.snapshot().status.request_id == first["request_id"]
    mission.tasks.pop(0)()
    finish(mission)
    assert send(identified, command_id="busy-request")["disposition"] == "duplicate"
    assert not mission.tasks and len(mission.nav.sent) == 2


def test_stop_retry_and_stale_target_cannot_stop_later_mission(ros, mission, identified):
    m, i = mission, identified
    first = send(i)
    m.tasks.pop(0)()
    stop = {"command_id": "stop-one", "command": "stop", "target_request_id": first["request_id"]}
    assert send(i, **stop)["disposition"] == "recorded" and len(m.nav.canceled) == 1
    assert send(i, **stop)["disposition"] == "duplicate" and len(m.nav.canceled) == 1
    m.nav.sent[-1][2](NavigationEvent("succeeded"))
    assert m.commands.snapshot().status.state == "canceled" and not m.tasks
    m.model.replies.append(proposal())
    m.critic.replies.append(review())
    send(i, command_id="next-mission")
    m.tasks.pop(0)()
    before = m.commands.snapshot().status
    assert send(i, **stop)["disposition"] == "duplicate"
    assert send(i, **{**stop, "command_id": "late-first-delivery"})["disposition"] == "recorded"
    assert ros.publishers["~/navigation_status"].messages[-1]["state"] == "stale_command"
    assert m.commands.snapshot().status == before and len(m.nav.canceled) == 1


def test_failed_journal_blocks_new_work_and_leaves_legacy_stop_available(ros, mission, identified, monkeypatch):
    m, i = mission, identified
    send(i)
    m.tasks.pop(0)()

    def fail(command):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(i.journal, "claim", fail)
    assert send(i, command_id="other")["disposition"] == "unavailable"
    assert len(m.nav.sent) == 1 and not m.tasks
    i.bridge._on_text(SimpleNamespace(data="stop"))
    assert m.nav.canceled and m.commands.snapshot().status.state == "canceling"
    assert ros.subscriptions["~/command"].callback_group is not ros.subscriptions["~/command_json"].callback_group


def test_stop_during_journal_write_invalidates_pending_instruction(ros, mission, identified, monkeypatch):
    m, i = mission, identified
    entered, release = threading.Event(), threading.Event()
    original = i.journal.claim

    def slow(command):
        entered.set()
        assert release.wait(timeout=5)
        return original(command)

    monkeypatch.setattr(i.journal, "claim", slow)
    thread = threading.Thread(target=lambda: send(i))
    thread.start()
    try:
        assert entered.wait(timeout=5)
        i.bridge._on_text(SimpleNamespace(data="stop"))
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and i.receipts[-1]["disposition"] == "recorded"
    assert ros.publishers["~/navigation_status"].messages[-1]["state"] == "stale_command"
    assert not m.tasks and not m.nav.sent
    assert send(i)["disposition"] == "duplicate" and not m.tasks


@pytest.mark.parametrize("enabled", [False, True])
def test_no_journal_never_routes_version_two(ros, mission, enabled):
    bridge = OperatorInterface(ros.node, mission.commands if enabled else None)
    bridge._on_json(SimpleNamespace(data=json.dumps(envelope())))
    assert ros.publishers["~/command_receipt"].messages[-1]["disposition"] == ("unavailable" if enabled else "disabled")
    assert not mission.tasks and read_service(ros)["command_identity"] is None


def test_identified_choice_is_used_once_and_bound_to_its_prompt(ros, mission, identified, hashing):
    m, i = mission, identified
    del m.resolver._places["printer"]
    m.store.upsert(
        [embedded(hashing, "printer", t=t, pose=Pose(x, 2, map_id="office")) for t, x in ((1000, 1), (1001, 8))]
    )
    first = send(i)
    m.tasks.pop(0)()
    assert m.commands.snapshot().status.state == "ambiguous"
    choice = {"command_id": "choice", "command": "choose", "option": 2, "target_request_id": first["request_id"]}
    assert (
        send(i, **{**choice, "command_id": "stale-choice", "target_request_id": "a" * 32})["disposition"] == "recorded"
    )
    assert not m.tasks and m.commands.snapshot().status.state == "ambiguous"
    accepted = send(i, **choice)
    assert accepted["disposition"] == "recorded" and len(m.tasks) == 1
    assert send(i, **choice)["disposition"] == "duplicate" and len(m.tasks) == 1
    m.tasks.pop(0)()
    assert len(m.nav.sent) == 1 and m.nav.sent[0][1].pose.x == 1  # option 2, not the top hit
    assert send(i, **{**choice, "command_id": "late-choice"})["disposition"] == "recorded"
    assert ros.publishers["~/navigation_status"].messages[-1]["state"] == "stale_command"
    assert len(m.nav.sent) == 1 and not m.tasks


def test_controller_exception_after_reservation_is_not_retried(ros, mission, identified, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("crash after admission")

    monkeypatch.setattr(mission.commands, "handle", fail)
    with pytest.raises(RuntimeError, match="crash"):
        send(identified)
    assert send(identified)["disposition"] == "duplicate"
    assert len(calls) == 1 and not mission.tasks and not mission.nav.sent
