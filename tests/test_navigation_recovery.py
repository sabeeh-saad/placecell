import sys
from concurrent.futures import Future
from types import SimpleNamespace as Obj

import pytest

from placecell import Destination, MissionPlanner, PlanReviewAgent, Pose
from placecell.errors import ValidationError
from placecell.navigation_ownership import NavigationOwnership, NavigationScope, main
from placecell.ros2.navigation import Nav2Navigator
from placecell.ros2.recovery import GoalRecovery, create_recovery
from tests.test_missions import mission as mission

SCOPE = NavigationScope("robot", "map-v1", "/navigate_to_pose")


class Service:
    def __init__(self):
        self.ready = True
        self.calls, self.removed = [], []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        future = Future()
        self.calls.append((request, future))
        return future

    def remove_pending_request(self, future):
        self.removed.append(future)


def recovery(owner):
    result, cancel, clock = Service(), Service(), [0.0]
    transport = GoalRecovery(owner, result, cancel, str, str, timeout_s=1, clock=lambda: clock[0])
    return transport, result, cancel, clock


def test_missing_scope_lock_and_persistent_identity(tmp_path):
    path = tmp_path / "nav.sqlite3"
    owner = NavigationOwnership(path, SCOPE)
    assert owner.snapshot()["state"] == "unknown"
    with pytest.raises(ValidationError, match="unresolved"):
        owner.reserve("first")
    with pytest.raises(ValidationError, match="another process"):
        NavigationOwnership(path, SCOPE)
    owner.attest_clean("Test server is freshly created with no clients")
    identity = owner.reserve("first")
    with pytest.raises(ValidationError, match="unresolved"):
        owner.reserve("second")
    with pytest.raises(ValidationError, match="identity"):
        owner.terminal("different", "succeeded")
    with pytest.raises(ValidationError, match="terminal"):
        owner.terminal(identity, "canceling")
    owner.close()
    with pytest.raises(ValidationError, match="scope"):
        NavigationOwnership(path, NavigationScope("other", "map-v1", "/navigate_to_pose"))
    owner = NavigationOwnership(path, SCOPE)
    assert owner.snapshot()["goal_id"] == identity and owner.snapshot()["request_id"] == "first"
    owner.terminal(identity, "canceled")
    owner.close()
    owner = NavigationOwnership(path, SCOPE)
    assert owner.snapshot()["state"] == "clean"
    owner.close()
    owner.close()


@pytest.mark.parametrize("path", ["", ":memory:"])
def test_requires_persistent_path(path):
    with pytest.raises(ValidationError, match="persistent"):
        NavigationOwnership(path, SCOPE)


@pytest.mark.parametrize("scope", [("", "map", "/nav"), ("r", "", "/nav"), ("r", "map", "nav")])
def test_scope_validation(scope):
    with pytest.raises(ValidationError):
        NavigationScope(*scope)


def test_missing_journal_never_queries_or_assumes_ready(tmp_path):
    owner = NavigationOwnership(tmp_path / "nav", SCOPE)
    transport, result, cancel, clock = recovery(owner)
    for i in range(100):
        clock[0] = i
        transport.poll()
    assert "unknown" in transport.block_reason and not result.calls and not cancel.calls
    owner.close()


def test_corrupt_nil_goal_is_refused_before_ros_can_interpret_it_as_cancel_all(tmp_path):
    path = tmp_path / "nav"
    owner = NavigationOwnership(path, SCOPE)
    owner.attest_clean("Fresh server")
    owner.reserve("first")
    with owner._db:
        owner._db.execute("UPDATE ownership SET goal_id=? WHERE id=1", ("0" * 32,))
    owner.close()
    with pytest.raises(ValidationError, match="invalid goal identity"):
        NavigationOwnership(path, SCOPE)


@pytest.mark.parametrize("status", [4, 5, 6])
def test_only_matching_terminal_result_releases_recovery(tmp_path, status):
    owner = NavigationOwnership(tmp_path / "nav", SCOPE)
    owner.attest_clean("Fresh controlled server")
    goal = owner.reserve("first")
    transport, result, cancel, _clock = recovery(owner)
    transport.poll()
    assert result.calls[0][0] == cancel.calls[0][0] == goal
    cancel.calls[0][1].set_result(Obj(goals_canceling=[goal]))
    transport.poll()
    assert transport.block_reason and owner.snapshot()["state"] == "pending"
    result.calls[0][1].set_result(Obj(status=status))
    transport.poll()
    assert not transport.block_reason and owner.snapshot()["state"] == "clean"
    transport.poll()
    assert len(result.calls) == len(cancel.calls) == 1
    owner.close()


def test_unknown_result_absent_service_timeouts_and_stale_response_cannot_release(tmp_path):
    owner = NavigationOwnership(tmp_path / "nav", SCOPE)
    owner.attest_clean("Fresh server")
    owner.reserve("first")
    transport, result, cancel, clock = recovery(owner)
    result.ready = cancel.ready = False
    transport.poll()
    assert not result.calls and transport.block_reason
    result.ready = cancel.ready = True
    transport.poll()
    result.calls[0][1].set_result(Obj(status=0))
    cancel.calls[0][1].set_result(Obj(goals_canceling=[]))
    transport.poll()
    assert transport.block_reason
    clock[0] = 1
    transport.poll()
    assert len(result.calls) == 2
    clock[0] = 2
    transport.poll()
    assert len(result.removed) == len(cancel.removed) == 1
    result.calls[1][1].set_result(Obj(status=5))
    transport.poll()
    assert transport.block_reason and owner.snapshot()["state"] == "pending"
    clock[0] = 3
    transport.poll()
    transport.close()
    assert len(result.removed) == len(cancel.removed) == 2
    owner.close()


def test_recovery_errors_retain_uncertainty(tmp_path, monkeypatch):
    owner = NavigationOwnership(tmp_path / "nav", SCOPE)
    owner.attest_clean("Fresh server")
    owner.reserve("first")
    transport, result, _cancel, clock = recovery(owner)
    transport.poll()
    result.calls[0][1].set_exception(RuntimeError("transport error"))
    transport.poll()
    assert "failed" in transport.block_reason
    clock[0] = 2
    transport.poll()
    result.calls[1][1].set_result(Obj(status=5))

    def fail(*_args):
        raise OSError("journal unavailable")

    monkeypatch.setattr(owner, "terminal", fail)
    transport.poll()
    assert transport.block_reason and owner.snapshot()["state"] == "pending"
    clock[0] = 4
    monkeypatch.setattr(result, "call_async", fail)
    transport.poll()
    assert "transport failed" in transport.block_reason
    transport.close()
    owner.close()


def test_adapter_commits_wire_identity_before_submission_and_keeps_failed_terminal(tmp_path, monkeypatch):
    from tests.test_nav2 import Client, Handle

    owner = NavigationOwnership(tmp_path / "nav", SCOPE)
    owner.attest_clean("Fresh server")
    transport, _, _, _ = recovery(owner)
    client = Client()
    original = client.send_goal_async

    def send(goal, feedback_callback, goal_uuid):
        assert owner.snapshot() == {"state": "pending", "goal_id": goal_uuid, "request_id": "first", "outcome": ""}
        return original(goal, feedback_callback)

    client.send_goal_async = send
    nav = Nav2Navigator(client, lambda pose: pose, ownership=owner, recovery=transport)
    events = []
    nav.send("first", Destination("x", Pose(1, 2), "coordinates"), events.append)
    handle = Handle()
    client.response.set_result(handle)

    def fail(*_args):
        raise OSError("journal unavailable")

    monkeypatch.setattr(owner, "terminal", fail)
    handle.result.set_result(Obj(status=4, result=Obj()))
    assert events[-1].state == "uncertain" and owner.snapshot()["state"] == "pending"
    with pytest.raises(ValidationError, match="already pending"):
        nav.send("second", Destination("y", Pose(2, 3), "coordinates"), events.append)
    nav.close()
    with pytest.raises(ValidationError, match="closed"):
        nav.send("third", Destination("z", Pose(3, 4), "coordinates"), events.append)


def test_controller_blocks_planning_stop_and_snapshot_until_reconciled(mission):
    from placecell.navigation import NavigationCommands

    m = mission
    blocked = ["Previous Nav2 ownership is unknown"]
    commands = NavigationCommands(
        m.resolver,
        m.nav,
        lambda work: m.tasks.append(work) is None,
        m.events.append,
        mission_planner=MissionPlanner(m.model, PlanReviewAgent(m.critic)),
        startup_block_reason=lambda: blocked[0],
    )
    assert commands.snapshot().status.state == "uncertain" and commands.busy
    commands.handle("Visit printer then cupboard")
    commands.handle("stop")
    commands.poll()
    assert commands.snapshot().status.state == "uncertain" and not m.tasks
    blocked[0] = ""
    commands.poll()
    assert not commands.busy and commands.snapshot().status.state == "idle" and not m.tasks
    commands.handle("Visit printer then cupboard")
    assert len(m.tasks) == 1


def test_cli_requires_explicit_attestation_and_keeps_reason(tmp_path, capsys):
    args = ["--journal", str(tmp_path / "nav"), "--robot-id", "r", "--map-id", "v1", "--action-name", "/nav"]
    with pytest.raises(SystemExit):
        main(["attest-clean", *args])
    main(["inspect", *args])
    assert '"state": "unknown"' in capsys.readouterr().out
    main(["attest-clean", *args, "--confirm-nav2-stopped", "--reason", "Reset supervised test server"])
    assert "Reset supervised test server" in capsys.readouterr().out


def test_action_alias_is_resolved_before_creating_recovery_services(tmp_path, monkeypatch):
    owner = NavigationOwnership(tmp_path / "nav", SCOPE)
    owner.attest_clean("Fresh test server")
    goal_id = owner.reserve("first")
    service_type = Obj(Request=lambda: Obj(goal_info=Obj(goal_id=None)))
    monkeypatch.setitem(sys.modules, "action_msgs.srv", Obj(CancelGoal=service_type))
    monkeypatch.setitem(
        sys.modules, "nav2_msgs.action", Obj(NavigateToPose=Obj(Impl=Obj(GetResultService=service_type)))
    )
    monkeypatch.setitem(sys.modules, "rclpy.callback_groups", Obj(ReentrantCallbackGroup=object))
    monkeypatch.setitem(sys.modules, "unique_identifier_msgs.msg", Obj(UUID=lambda **kw: Obj(**kw)))
    clients = {}

    def create_client(_type, name, **_kw):
        clients[name] = Service()
        return clients[name]

    node = Obj(resolve_topic_name=lambda _: "/resolved/navigate", create_client=create_client)
    transport = create_recovery(node, "alias", owner, 1)
    assert set(clients) == {"/resolved/navigate/_action/get_result", "/resolved/navigate/_action/cancel_goal"}
    transport.poll()
    assert bytes(clients["/resolved/navigate/_action/get_result"].calls[0][0].goal_id.uuid).hex() == goal_id
    assert bytes(clients["/resolved/navigate/_action/cancel_goal"].calls[0][0].goal_info.goal_id.uuid).hex() == goal_id
    transport.close()
    owner.close()
