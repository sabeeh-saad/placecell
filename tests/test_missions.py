from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from placecell import (
    ChatReply,
    CollectionInfo,
    DestinationResolver,
    InMemoryStore,
    MissionContext,
    MissionPlan,
    MissionPlanner,
    NavigationCommands,
    NavigationEvent,
    Observation,
    PlanReviewAgent,
    Pose,
    Recall,
    ToolCall,
)
from placecell.errors import ProviderError, ValidationError
from placecell.ros2.node import build_mission_planner, navigation_payload
from placecell.verification import SceneVerdict
from tests.conftest import embedded
from tests.test_navigation import FakeNavigator


def proposal(destinations=("printer", "cupboard"), *, decision="ready", **extra):
    return ChatReply(
        None,
        (
            ToolCall(
                "plan",
                "propose_navigation_plan",
                {"decision": decision, "destinations": list(destinations), "message": "Requested visits.", **extra},
            ),
        ),
    )


def review(decision="approve"):
    return ChatReply(
        None, (ToolCall("review", "review_navigation_plan", {"decision": decision, "message": "Checked intent."}),)
    )


class Model:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.before_reply = lambda: None

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        self.before_reply()
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_two_agents_preserve_plan_and_share_context_without_sharing_a_conversation():
    model, critic = Model(proposal()), Model(review())
    history = [{"kind": "instruction", "data": {"text": "the printer beside the window"}}]
    plan = MissionPlanner(model, PlanReviewAgent(critic)).plan(
        "Visit that printer, followed by the cupboard", context=history
    )
    assert plan.destinations == ("printer", "cupboard")
    for calls in (model.calls, critic.calls):
        messages, tools = calls[0]
        assert [m.role for m in messages] == ["system", "user"]
        assert json.loads(messages[1].content)["recent_context"] == history
        assert len(tools) == 1
    assert model.calls[0][0][0].content != critic.calls[0][0][0].content
    assert "destinations" in json.loads(critic.calls[0][0][1].content)


@pytest.mark.parametrize("decision", ["clarify", "reject"])
def test_reviewer_can_block_an_otherwise_valid_plan(decision):
    planner = MissionPlanner(Model(proposal()), PlanReviewAgent(Model(review(decision))))
    plan = planner.plan("Do not visit the printer or cupboard")
    assert plan.decision == decision and not plan.destinations


@pytest.mark.parametrize(
    "reply",
    [
        ChatReply("Go to the printer"),
        ChatReply(None, (ToolCall("1", "drive", {"x": 1}),)),
        ChatReply(None, proposal().tool_calls * 2),
        proposal(x=3),
        proposal(tuple(str(i) for i in range(9))),
        proposal(("",)),
        proposal((1,)),
        proposal(decision="clarify"),
        proposal((), decision="ready"),
        proposal(decision="drive"),
        ChatReply(None, (ToolCall("1", "propose_navigation_plan", {}),)),
        ChatReply(
            None,
            (
                ToolCall(
                    "1", "propose_navigation_plan", {"decision": "ready", "destinations": "printer", "message": "x"}
                ),
            ),
        ),
    ],
)
def test_malformed_or_over_budget_plans_never_reach_the_reviewer(reply):
    critic = Model(review())
    with pytest.raises((ProviderError, ValidationError)):
        MissionPlanner(Model(reply), PlanReviewAgent(critic)).plan("visit places")
    assert not critic.calls


def test_clarification_needs_no_review_and_canceled_work_needs_no_model():
    model, critic = Model(proposal((), decision="clarify")), Model()
    planner = MissionPlanner(model, PlanReviewAgent(critic))
    assert planner.plan("go there").decision == "clarify" and not critic.calls
    with pytest.raises(ValidationError):
        planner.plan("go there", lambda: True)
    assert len(model.calls) == 1
    for instruction in ("", "x" * 2001):
        with pytest.raises(ValidationError):
            planner.plan(instruction)
    with pytest.raises(ValidationError):
        MissionPlanner(model, PlanReviewAgent(critic), max_destinations=0)


@pytest.fixture
def mission(hashing, monkeypatch):
    monkeypatch.setattr("placecell.navigation.data_url", lambda uri: "data:image/jpeg;base64,YQ==")
    store = InMemoryStore(CollectionInfo("missions", hashing.model_name, hashing.dimension))
    model, critic, now, ready = Model(proposal()), Model(review()), [3000.0], [True]
    verifier = SimpleNamespace(verify=lambda *_: SceneVerdict("matched", "Target visible."))
    resolver = DestinationResolver(
        store,
        Recall(store, hashing, clock=lambda: now[0]),
        robot_id="r1",
        map_id="office",
        places={"printer": Pose(1, 2, map_id="office"), "cupboard": Pose(8, 2, map_id="office")},
        verifier=verifier,
        clock=lambda: now[0],
    )
    nav, tasks, events = FakeNavigator(), [], []
    context = MissionContext()
    commands = NavigationCommands(
        resolver,
        nav,
        lambda f: tasks.append(f) is None,
        events.append,
        mission_planner=MissionPlanner(model, PlanReviewAgent(critic)),
        mission_context=context,
        clock=lambda: now[0],
        observation_clock=lambda: now[0],
        localization_ready=lambda: ready[0],
    )
    yield SimpleNamespace(
        commands=commands,
        resolver=resolver,
        store=store,
        model=model,
        critic=critic,
        now=now,
        ready=ready,
        nav=nav,
        tasks=tasks,
        events=events,
        context=context,
        verifier=verifier,
    )
    context.close()
    store.close()


def start(m):
    m.commands.handle("First visit the printer; afterwards take me to the cupboard")
    assert m.events[-1].state == "planning"
    m.tasks.pop(0)()


def test_ordered_mission_reports_each_goal_and_ignores_duplicate_old_results(mission):
    m = mission
    start(m)
    assert len(m.nav.sent) == 1 and m.nav.sent[0][1].label == "printer"
    planned = next(e for e in m.events if e.state == "planned")
    assert planned.mission_destinations == ("printer", "cupboard")
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    assert len(m.nav.sent) == 1 and m.events[-1].mission_step == 2
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    assert len(m.tasks) == 1
    m.tasks.pop(0)()
    assert m.nav.sent[1][1].label == "cupboard"
    assert m.nav.sent[0][0] != m.nav.sent[1][0]
    m.nav.sent[1][2](NavigationEvent("succeeded"))
    assert not m.commands.busy
    final = json.loads(navigation_payload(m.events[-1]))
    assert final["state"] == "succeeded" and final["mission_step"] == 2
    assert final["mission_id"] == planned.mission_id
    assert final["mission_destinations"] == ["printer", "cupboard"]
    assert len(m.model.calls) == len(m.critic.calls) == 1


def test_single_goal_and_repeated_visits_are_not_deduplicated(mission):
    m = mission
    m.model.replies = [proposal(("printer", "printer"))]
    start(m)
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    m.tasks.pop(0)()
    assert len(m.nav.sent) == 2
    m.nav.sent[1][2](NavigationEvent("succeeded"))
    m.model.replies, m.critic.replies = [proposal(("cupboard",))], [review()]
    start(m)
    m.nav.sent[-1][2](NavigationEvent("succeeded"))
    assert m.events[-1].mission_step == 1 and not m.commands.busy


@pytest.mark.parametrize("phase", ["planner", "reviewer", "queued", "between_steps"])
def test_stop_discards_late_agent_answers_and_queued_steps(mission, phase):
    m = mission
    if phase in {"planner", "reviewer"}:
        (m.model if phase == "planner" else m.critic).before_reply = lambda: m.commands.handle("stop")
        start(m)
    elif phase == "queued":
        m.commands.handle("visit both places")
        m.commands.handle("stop")
        m.tasks.pop(0)()
    else:
        start(m)
        m.nav.sent[0][2](NavigationEvent("succeeded"))
        m.commands.handle("stop")
        m.tasks.pop(0)()
    assert len(m.nav.sent) == (1 if phase == "between_steps" else 0)
    assert not m.commands.busy and m.events[-1].state == "canceled"


@pytest.mark.parametrize("result", ["succeeded", "canceled", "failed"])
def test_cancel_during_motion_never_advances_even_if_success_arrives_late(mission, result):
    m = mission
    start(m)
    m.commands.handle("stop")
    assert m.commands.busy and len(m.nav.canceled) == 1
    m.nav.sent[0][2](NavigationEvent(result))
    assert not m.commands.busy and not m.tasks and len(m.nav.sent) == 1


@pytest.mark.parametrize("state", ["canceling", "uncertain", "cancel_failed"])
def test_transport_cancellation_intent_survives_feedback_and_late_success(mission, state):
    m = mission
    start(m)
    callback = m.nav.sent[0][2]
    callback(NavigationEvent(state))
    callback(NavigationEvent("navigating", distance_remaining=1.0))
    assert m.events[-1].state == "canceling" and m.commands.busy
    callback(NavigationEvent("succeeded"))
    assert m.events[-1].state == "canceled"
    assert not m.commands.busy and not m.tasks and len(m.nav.sent) == 1


@pytest.mark.parametrize("outcome", ["failed", "rejected", "unavailable", "uncertain"])
def test_failure_or_uncertain_transport_never_skips_to_the_next_goal(mission, outcome):
    m = mission
    start(m)
    m.nav.sent[0][2](NavigationEvent(outcome))
    assert not m.tasks and len(m.nav.sent) == 1
    assert m.commands.busy == (outcome == "uncertain")


def test_next_goal_is_resolved_against_current_state_and_not_precomputed(mission):
    m = mission
    start(m)
    del m.resolver._places["cupboard"]
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    m.tasks.pop(0)()
    assert len(m.nav.sent) == 1 and m.events[-1].state == "not_found"
    assert not m.commands.busy


@pytest.mark.parametrize("arrival", ["matched", "uncertain", "not_matched"])
def test_memory_goal_requires_fresh_verified_arrival_before_next_step(mission, hashing, arrival):
    m = mission
    del m.resolver._places["printer"]
    memory = embedded(hashing, "printer", pose=Pose(1, 2, map_id="office"))
    m.store.upsert([memory])
    start(m)
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    assert m.events[-1].state == "awaiting_observation" and not m.tasks
    m.now[0] += 1
    m.verifier.verify = lambda *_: SceneVerdict(arrival, "Fresh view.")
    m.commands.observe(Observation("r1", "front", m.now[0], memory.pose, memory.evidence, True))
    m.tasks.pop(0)()
    assert len(m.nav.sent) == 1
    if arrival == "matched":
        m.tasks.pop(0)()
        assert len(m.nav.sent) == 2
    else:
        assert not m.tasks and not m.commands.busy and m.events[-1].state == "destination_unverified"


@pytest.mark.parametrize("action", ["choose", "expire", "stop"])
def test_ambiguity_retains_remaining_mission_and_never_picks_the_top_match(mission, hashing, action):
    m = mission
    del m.resolver._places["printer"]
    m.store.upsert(
        [embedded(hashing, "printer", t=t, pose=Pose(x, 2, map_id="office")) for t, x in ((1000, 1), (1001, 8))]
    )
    start(m)
    assert not m.nav.sent and m.events[-1].state == "ambiguous" and m.commands.busy
    m.commands.handle("go somewhere else")
    assert m.events[-1].state == "busy" and len(m.model.calls) == 1
    if action == "choose":
        m.commands.handle("option two")
        m.tasks.pop(0)()
        assert len(m.nav.sent) == 1 and m.events[-1].mission_destinations == ("printer", "cupboard")
        assert m.events[-1].mission_step == 1
    elif action == "expire":
        m.now[0] += 31
        m.commands.poll()
        assert not m.commands.busy and not m.nav.sent
    else:
        m.commands.handle("stop")
        assert not m.commands.busy and m.events[-1].state == "canceled"


@pytest.mark.parametrize("failure", ["provider", "review", "timeout", "localization", "malformed"])
def test_planning_failures_do_not_move(mission, failure):
    m = mission
    if failure == "provider":
        m.model.replies = [OSError("offline")]
    elif failure == "review":
        m.critic.replies = [review("reject")]
    elif failure == "timeout":
        m.model.before_reply = lambda: m.now.__setitem__(0, m.now[0] + 31)
    elif failure == "localization":
        m.critic.before_reply = lambda: m.ready.__setitem__(0, False)
    else:
        m.model.replies = [proposal(coordinates=[2, 3])]
    start(m)
    assert not m.nav.sent and not m.commands.busy


def test_poll_expires_planning_before_the_worker_runs(mission):
    m = mission
    m.commands.handle("visit the printer and then cupboard")
    m.now[0] += 31
    m.commands.poll()
    m.tasks.pop()()
    assert not m.model.calls and not m.nav.sent and not m.commands.busy


def test_full_queue_at_next_step_stops_mission(mission, monkeypatch):
    m = mission
    start(m)
    monkeypatch.setattr(m.commands, "_submit", lambda _: False)
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    assert m.events[-1].state == "failed" and not m.commands.busy


def test_follow_up_receives_saved_prompt_plan_and_actual_outcome(mission):
    m = mission
    m.model.replies = [proposal(("printer",)), proposal(("printer",))]
    m.critic.replies = [review(), review()]
    start(m)
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    m.commands.handle("take me there again")
    m.tasks.pop(0)()
    context = json.loads(m.model.calls[1][0][1].content)["recent_context"]
    assert any(c["kind"] == "instruction" and "printer" in c["data"]["text"] for c in context)
    assert any(c["kind"] == "status" and c["data"]["state"] == "succeeded" for c in context)
    assert not any(c["kind"] == "instruction" and c["data"]["text"] == "take me there again" for c in context)


def test_context_survives_restart_is_scoped_and_never_dispatches_old_commands(tmp_path):
    path = tmp_path / "context.sqlite3"
    context = MissionContext(path, scope="r1:office:alice")
    context.record("1", "instruction", {"text": "visit printer then cupboard"})
    context.record("1", "status", {"state": "navigating", "step": 1})
    context.close()
    reopened = MissionContext(path, scope="r1:office:alice")
    assert len(reopened.recent()) == 2 and reopened.recent(limit=1)[0]["data"]["state"] == "navigating"
    assert not reopened.recent(exclude_request_id="1")
    for scope in ("r2:office:alice", "r1:warehouse:alice", "r1:office:bob"):
        other = MissionContext(path, scope=scope)
        assert other.recent() == []
        other.close()
    reopened.close()


def test_context_failure_blocks_new_motion_but_never_blocks_stop(mission, monkeypatch):
    m = mission
    start(m)

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(m.context, "record", broken)
    m.commands.handle("stop")
    assert m.nav.canceled == [m.nav.sent[0][0]]
    m.nav.sent[0][2](NavigationEvent("canceled"))
    m.commands.handle("go to cupboard")
    assert not m.tasks and len(m.nav.sent) == 1
    assert m.events[-1].state == "unavailable"


def test_context_failure_during_dispatch_prevents_motion(mission, monkeypatch):
    m = mission
    record = m.context.record

    def fail_dispatch(request_id, kind, payload):
        if payload.get("state") == "submitting":
            raise OSError("disk full")
        record(request_id, kind, payload)

    monkeypatch.setattr(m.context, "record", fail_dispatch)
    start(m)
    assert not m.nav.sent and not m.commands.busy and m.events[-1].state == "unavailable"


def test_ros_builder_requires_explicit_model_and_creates_separate_roles(monkeypatch):
    from placecell import providers

    calls = []

    def model(*args, **kwargs):
        calls.append((args, kwargs))
        return Model(proposal() if len(calls) == 1 else review())

    monkeypatch.setattr(providers, "OpenAICompatibleChat", model)
    assert build_mission_planner({"mission_enabled": False}, None) is None
    with pytest.raises(ValidationError):
        build_mission_planner({"mission_enabled": True, "mission_model": ""}, None)
    planner = build_mission_planner(
        {
            "mission_enabled": True,
            "mission_model": "planner",
            "mission_base_url": "https://models.test/v1",
            "mission_review_model": "reviewer",
            "mission_review_base_url": "",
            "mission_max_destinations": 8,
            "mission_request_timeout_s": 8.0,
        },
        "fake-key",
    )
    assert planner.plan("visit printer then cupboard").decision == "ready"
    assert [c[0][0] for c in calls] == ["planner", "reviewer"]
    assert all(c[1]["retry"].attempts == 1 for c in calls)


def test_stop_does_not_wait_for_a_running_agent_thread(mission):
    m = mission
    entered, release = threading.Event(), threading.Event()

    def slow():
        entered.set()
        assert release.wait(3)

    m.model.before_reply = slow
    m.commands.handle("go to printer then cupboard")
    worker = threading.Thread(target=m.tasks.pop(0))
    stopper = threading.Thread(target=m.commands.cancel)
    worker.start()
    try:
        assert entered.wait(2)
        stopper.start()
        stopper.join(1)
        assert not stopper.is_alive() and not m.commands.busy
    finally:
        release.set()
        worker.join(2)
        if stopper.ident is not None:
            stopper.join(2)
    assert not worker.is_alive() and not m.nav.sent and not m.critic.calls


def test_restarted_controller_reads_history_only_when_a_new_request_arrives(mission, tmp_path):
    m = mission
    path = tmp_path / "session.sqlite3"
    old = MissionContext(path, scope="robot:map:operator")
    old.record("old", "instruction", {"text": "Visit the printer and cupboard"})
    old.record("old", "status", {"state": "navigating", "step": 1, "destinations": ["printer", "cupboard"]})
    old.close()
    restored = MissionContext(path, scope="robot:map:operator")
    try:
        commands = NavigationCommands(
            m.resolver,
            m.nav,
            lambda f: m.tasks.append(f) is None,
            m.events.append,
            mission_planner=MissionPlanner(m.model, PlanReviewAgent(m.critic)),
            mission_context=restored,
        )
        commands.poll()
        assert not commands.busy and not m.tasks and not m.nav.sent and not m.model.calls
        commands.handle("Visit the cupboard now")
        m.tasks.pop()()
        history = json.loads(m.model.calls[0][0][1].content)["recent_context"]
        assert history[0]["data"]["text"] == "Visit the printer and cupboard"
        assert history[1]["data"]["state"] == "navigating"
    finally:
        restored.close()


@pytest.mark.parametrize("kind", ["missing", "unknown", "empty_reason"])
def test_malformed_review_cannot_approve_a_plan(kind):
    if kind == "missing":
        reply = ChatReply("approved")
    elif kind == "unknown":
        reply = review("maybe")
    else:
        reply = ChatReply(None, (ToolCall("1", "review_navigation_plan", {"decision": "approve", "message": ""}),))
    with pytest.raises((ProviderError, ValidationError)):
        MissionPlanner(Model(proposal()), PlanReviewAgent(Model(reply))).plan("visit printer then cupboard")


def test_invalid_context_or_plan_payloads_are_rejected():
    with pytest.raises(ValidationError):
        MissionContext(scope=" ")
    context = MissionContext()
    try:
        for limit in (0, 101, True):
            with pytest.raises(ValidationError):
                context.recent(limit=limit)
        with pytest.raises(ValidationError):
            context.recent(max_chars=0)
        with pytest.raises(ValidationError):
            context.record("1", "execute", {"text": "move"})
        with pytest.raises(ValidationError):
            context.record("1", "instruction", {"text": "x" * 32001})
        for destinations in (["printer"], tuple("printer" for _ in range(21))):
            with pytest.raises(ValidationError):
                MissionPlan("ready", destinations, "x")
    finally:
        context.close()


def test_model_context_has_a_character_budget_without_deleting_persistent_history():
    context = MissionContext()
    try:
        context.record("1", "instruction", {"text": "x" * 1000})
        context.record("2", "instruction", {"text": "y" * 1000})
        recent = context.recent(max_chars=1500)
        assert len(recent) == 1 and recent[0]["request_id"] == "2"
        assert len(json.dumps(recent)) <= 1500
        assert len(context.recent()) == 2
    finally:
        context.close()


def test_context_failure_after_a_completed_step_does_not_launch_the_next(mission, monkeypatch):
    m = mission
    start(m)

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(m.context, "record", fail)
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    assert not m.commands.busy and not m.tasks and len(m.nav.sent) == 1
    assert m.events[-1].state == "unavailable"


def test_context_failure_during_motion_requests_cancellation_on_poll(mission, monkeypatch):
    m = mission
    start(m)

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(m.context, "record", fail)
    m.nav.sent[0][2](NavigationEvent("navigating"))
    m.commands.poll()
    assert m.nav.canceled == [m.nav.sent[0][0]] and m.commands.busy
    m.nav.sent[0][2](NavigationEvent("canceled"))
    assert not m.commands.busy
