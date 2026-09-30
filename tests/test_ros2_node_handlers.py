"""Question, correction and refinement topics, housekeeping timers and the navigation hooks."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from placecell.consolidation import Consolidator
from placecell.errors import ProviderError
from placecell.lifecycle import Curator, CuratorReport
from placecell.navigation_ownership import NavigationOwnership, NavigationScope
from placecell.providers import HashingEmbedder
from placecell.refinement import RefinementReport
from placecell.ros2.node import NO_CONFIDENT_ANSWER
from placecell.tracing import TraceStore
from tests.conftest import embedded
from tests.ros_fakes import localization, spy
from tests.ros_fakes import make_node as make_node
from tests.ros_fakes import ros as ros

T0 = 1000


def quiet(node):
    """Forget construction-time log records."""
    node.logger.records.clear()
    return node


def answers(node):
    return [json.loads(m.data) for m in node.publisher("~/answer").messages]


def remember(node, caption="printer", **fields):
    memory = embedded(HashingEmbedder(), caption, t=T0, **fields)
    node._store.upsert([memory])
    return memory


def ask(node, text):
    node.subscription("~/ask").callback(SimpleNamespace(data=text))


@pytest.mark.parametrize("text", ["", "   ", "x" * 2001])
def test_question_outside_the_length_bound_is_answered_with_an_error(make_node, text):
    node = make_node()
    ask(node, text)
    assert answers(node) == [
        {
            "schema_version": 1,
            "type": "answer",
            "question": text[:128],
            "error": "question must contain 1..2000 characters",
            "error_type": "",
        }
    ]


def test_question_is_answered_from_retrieval_on_a_worker(make_node):
    node = make_node()
    ask(node, "where is the kettle?")
    assert node.publisher("~/answer").wait_for(1)
    (answer,) = answers(node)
    assert (answer["source"], answer["answer"], answer["evidence"]) == ("retrieval", NO_CONFIDENT_ANSWER, [])
    memory = remember(node)
    ask(node, "printer")
    assert node.publisher("~/answer").wait_for(2)
    answer = answers(node)[-1]
    assert answer["answer"] == "printer" and answer["citations_valid"]
    assert [e["id"] for e in answer["evidence"]] == [memory.id]


def test_full_question_queue_answers_busy(make_node):
    node = make_node(question_workers=1, question_queue=1)
    entered, release, done = threading.Event(), threading.Event(), threading.Semaphore(0)

    def blocked(question):
        entered.set()
        release.wait(10)
        done.release()

    node._answer = blocked
    ask(node, "first")
    assert entered.wait(10)
    ask(node, "second")
    ask(node, "third")
    assert [a["error"] for a in answers(node)] == ["question queue full"]
    assert answers(node)[0]["question"] == "third"
    release.set()
    assert done.acquire(timeout=10) and done.acquire(timeout=10)


def correct(node, **payload):
    node.subscription("~/correct").callback(SimpleNamespace(data=json.dumps(payload)))


def test_correction_is_recorded_and_a_wrong_verdict_requests_a_recheck(make_node):
    node = quiet(make_node())
    memory = remember(node)
    correct(node, memory_id=memory.id, verdict="right", question="where is the printer?")
    correct(node, memory_id=memory.id, verdict="wrong", note="it moved")
    assert len(node._corrections) == 2
    assert [job["memory_id"] for job in node._store.refinements.pending()] == [memory.id]
    assert node._store.refinements.pending()[0]["reason"] == "operator correction"
    assert not node.logger.messages("warning")


def test_correction_without_a_recheck_is_still_saved(make_node):
    node = quiet(make_node())
    memory = remember(node, "chair")
    node._store.upsert([replace(memory, id="no-image", evidence=None, view_timestamp=None, sightings=())])
    correct(node, memory_id="no-image", verdict="wrong")
    assert len(node._corrections) == 1
    assert node.logger.messages("warning") == ["Correction saved; recheck queue full or memory is ineligible."]


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ('{"memory_id": "missing", "verdict": "right"}', "correction refers to an unavailable scene memory"),
        ("not json", "Expecting value"),
        ('{"verdict": "right"}', "'memory_id'"),
        ("[]", "list indices must be integers"),
    ],
)
def test_invalid_correction_is_ignored(make_node, data, reason):
    node = quiet(make_node())
    node.subscription("~/correct").callback(SimpleNamespace(data=data))
    assert len(node._corrections) == 0
    (warning,) = node.logger.messages("warning")
    assert warning.startswith("ignored correction: ") and reason in warning


def refine(node, data):
    node.subscription("~/refine").callback(SimpleNamespace(data=data if isinstance(data, str) else json.dumps(data)))


def test_refinement_requests(make_node):
    node = quiet(make_node())
    memory = remember(node)
    refine(node, {"memory_id": memory.id})
    refine(node, {"memory_id": "missing", "action": "recheck"})
    refine(node, {"memory_id": memory.id, "action": "rollback"})
    refine(node, {"memory_id": memory.id, "action": "erase"})
    refine(node, "{")
    assert node.logger.messages("info")[-2:] == [
        f"refinement recheck for {memory.id}: accepted",
        "refinement recheck for missing: skipped",
    ]
    disabled = "ignored refinement request: unknown refinement action or refinement is disabled"
    assert node.logger.messages("warning")[:2] == [disabled, disabled]
    assert node.logger.messages("warning")[2].startswith("ignored refinement request: ")


def test_rollback_goes_to_the_refiner(make_node):
    node = make_node(caption_model="scripted")
    memory = remember(node)
    refine(node, {"memory_id": memory.id, "action": "rollback"})
    assert node.logger.messages("info")[-1] == f"refinement rollback for {memory.id}: skipped"


def inline_maintenance(node):
    """Run maintenance tasks on the test thread, recording their coalescing keys."""
    keys = []

    def submit(function, *args, key=""):
        keys.append(key)
        function(*args)
        return True

    node._maintenance.submit = submit
    return keys


def test_refinement_timer_runs_the_refiner_under_one_key(make_node, monkeypatch):
    node = make_node(caption_model="scripted", curator_interval_s=0.0)
    keys = inline_maintenance(node)
    reports = iter([RefinementReport(), RefinementReport(attempted=1, updated=1)])
    monkeypatch.setattr("placecell.refinement.MemoryRefiner.run", lambda self: next(reports))
    before = len(node.logger.messages("info"))
    node.timer(3600.0).callback()
    node.timer(3600.0).callback()
    assert keys == ["refine", "refine"]
    assert node.logger.messages("info")[before:] == [f"memory refinement: {RefinementReport(attempted=1, updated=1)}"]


def test_refinement_timer_coalesces_while_a_pass_runs(make_node):
    node = make_node(caption_model="scripted", curator_interval_s=0.0)
    entered, release = threading.Event(), threading.Event()
    node._run_refiner = lambda: entered.set() or release.wait(10)
    node.timer(3600.0).callback()
    assert entered.wait(10)
    node.timer(3600.0).callback()
    assert node._maintenance.health()["coalesced"] == 1
    release.set()


def test_curator_drains_cleanup_around_each_object_prune_then_curates(make_node, monkeypatch, ros, tmp_path):
    node = make_node(navigation_enabled=True, mission_enabled=True, mission_model="scripted")
    keys = inline_maintenance(node)
    removals = iter([1, 1, 0])
    befores = []
    store = node._store
    store.drain_cleanup = spy(ros.calls, "drain_cleanup", store.drain_cleanup)

    def prune(before, limit):
        ros.calls.append("objects.prune")
        befores.append((before, limit))
        return next(removals)

    store.objects.prune = prune
    monkeypatch.setattr(Curator, "run", lambda self: ros.calls.append("curator.run") or CuratorReport())
    node._corrections.prune = spy(ros.calls, "corrections.prune", node._corrections.prune)
    node._mission_context.prune = spy(ros.calls, "mission_context.prune", node._mission_context.prune)
    store.maintain = lambda: ros.calls.append("maintain")
    node.timer(3600.0).callback()
    assert keys == ["curate"]
    assert ros.calls == [
        "drain_cleanup",
        *["objects.prune", "drain_cleanup"] * 2,
        "objects.prune",
        "curator.run",
        "corrections.prune",
        "mission_context.prune",
        "maintain",
    ]
    assert befores == [(T0 - 2592000.0, 1)] * 3
    assert not node.logger.messages("info")[-1].startswith("curator removed")


def test_object_pruning_is_bounded_per_pass_and_reports_removals(make_node, monkeypatch, ros):
    node = make_node()
    inline_maintenance(node)
    node._store.objects.prune = lambda before, limit: ros.calls.append("prune") or 1
    report = CuratorReport(expired=1, discredited=2, history_pruned=3)
    monkeypatch.setattr(Curator, "run", lambda self: report)
    node.timer(3600.0).callback()
    assert ros.calls.count("prune") == 128
    assert node.logger.messages("info")[-1] == (
        f"curator removed {report.removed} memories, discredited 2, pruned 3 sightings"
    )


@pytest.mark.parametrize("outcome", ["summaries", "nothing", "error"])
def test_consolidation_timer(make_node, monkeypatch, outcome):
    def run(self):
        if outcome == "error":
            raise ProviderError("model offline")
        return SimpleNamespace(summaries=2 if outcome == "summaries" else 0, folded=5)

    monkeypatch.setattr(Consolidator, "run", run)
    node = make_node(chat_model="scripted", consolidate_interval_s=600.0)
    keys = inline_maintenance(node)
    before = list(node.logger.records)
    node.timer(600.0).callback()
    assert keys == ["consolidate"]
    logged = [(level, message) for level, message, _ in node.logger.records[len(before) :]]
    assert (
        logged
        == {
            "summaries": [("info", "consolidated 5 memories into 2 summaries")],
            "nothing": [],
            "error": [("error", "consolidation failed: model offline")],
        }[outcome]
    )


def test_vector_index_sync_runs_on_its_own_worker(make_node, ros, tmp_path):
    node = make_node(db_path=str(tmp_path / "db"))
    submitted = []
    node._indexing.submit = lambda function, key="": submitted.append((function, key)) or True
    node.timer(2.0).callback()
    assert submitted == [(node._store.sync_index, "sync")]


def test_diagnostics_report_queues_and_trace_health(make_node, monkeypatch, tmp_path):
    node = quiet(
        make_node(
            navigation_enabled=True,
            mission_trace_path=str(tmp_path / "traces.sqlite3"),
        )
    )
    diagnostics = node.timer(30.0)
    diagnostics.callback()
    ingestion, queues, traces = node.logger.messages("info")[-3:]
    assert ingestion.startswith("ingestion: {'queued': 0,") and ingestion.endswith("dropped=0, objects=0")
    assert queues.startswith("queues: questions={") and "commands={'accepted': 0," in queues
    assert ", images={'oversized': 0," in queues and ", sensors={'future_dropped': 0," in queues
    assert traces.startswith("mission traces: {")
    assert not node.logger.messages("warning")
    monkeypatch.setattr(TraceStore, "health", lambda self: {"dropped_events": 1, "write_errors": 0})
    diagnostics.callback()
    assert node.logger.messages("warning") == ["Mission trace capture is incomplete; inspect trace health counters."]


def test_diagnostics_without_navigation(make_node):
    node = make_node()
    node.timer(30.0).callback()
    assert "commands=None" in node.logger.messages("info")[-1]


def attest(tmp_path):
    ownership = NavigationOwnership(
        str(tmp_path / "navigation.sqlite3"), NavigationScope("robot", "test-v1", "/navigate_to_pose")
    )
    ownership.attest_clean("offline test without a Nav2 server")
    ownership.close()


def statuses(node):
    return [json.loads(m.data) for m in node.publisher("~/navigation_status").messages]


def test_command_runs_on_the_command_worker_and_publishes_navigation_status(make_node, tmp_path):
    attest(tmp_path)
    places = tmp_path / "places.json"
    places.write_text(json.dumps({"printer": {"x": 1.0, "y": 2.0, "yaw": 0.0, "map_id": "test-v1"}}))
    node = make_node(navigation_enabled=True, places_file=str(places))
    node.subscription("/amcl_pose").callback(localization(T0))
    assert not node.navigation_busy()
    node.subscription("~/command").callback(SimpleNamespace(data="go to printer"))
    status = node.publisher("~/navigation_status")
    assert status.wait_until(lambda messages: any(json.loads(m.data)["state"] == "unavailable" for m in messages))
    assert node._command_tasks.health()["accepted"] == 1
    (client,) = node.action_clients
    assert client.goals == []  # the server was not ready, so no goal was sent
    client.ready = True
    node.subscription("~/command").callback(SimpleNamespace(data="go to printer"))
    assert client.sent.wait(10)
    (goal,) = client.goals
    assert goal.pose.header.frame_id == "map" and goal.pose.header.stamp.sec == T0
    assert (goal.pose.pose.position.x, goal.pose.pose.position.y) == (1.0, 2.0)
    assert node.navigation_busy()
    node.stop_navigation()
    assert node._commands.snapshot().closed


def test_unreconciled_navigation_ownership_blocks_until_attested(make_node):
    node = make_node(navigation_enabled=True)
    assert node.navigation_busy()
    assert node._commands.snapshot().status.state == "uncertain"


def test_navigation_hooks_without_navigation(make_node):
    node = make_node()
    node.stop_navigation()
    assert not node.navigation_busy() and not node._submit_command(lambda: None)


def test_mission_context_drops_history_whose_references_are_gone(make_node):
    node = make_node(navigation_enabled=True, mission_enabled=True, mission_model="scripted")
    memory = remember(node)
    context = node._mission_context
    context.record("a", "status", {"state": "succeeded", "memory_id": memory.id})
    context.record("b", "status", {"state": "succeeded"})
    assert [event["request_id"] for event in context.recent()] == ["a", "b"]
    node._store.upsert([replace(memory, superseded=True)])
    assert [event.get("kind") for event in context.recent()] == ["status"]
    context.record("c", "status", {"state": "succeeded", "object_id": "gone"})
    assert [event.get("kind") for event in context.recent()] == ["retention_boundary"]
