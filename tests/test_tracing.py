from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from dataclasses import replace

import pytest

from placecell.errors import ProviderError, ValidationError
from placecell.fault_injection import _control, _Rig, run_faults
from placecell.memory import Memory, Pose
from placecell.providers._http import Endpoint, RetryPolicy
from placecell.ros2.node import build_trace_store
from placecell.trace_export import main
from placecell.tracing import (
    TraceStore,
    bind_trace,
    current_trace,
    provider_usage,
    read_trace,
    trace_event,
    trace_scope,
    trace_span,
)
from tests.conftest import FakeTransport


@pytest.fixture
def store(tmp_path):
    traces = TraceStore(tmp_path / "traces.sqlite3", queue_size=1024, secrets=["my-configured-credential"])
    yield traces
    assert traces.close()


def exported(store, mission=None):
    assert store.flush()
    return read_trace(store.path, mission)


def test_persistent_nested_spans_errors_and_bound_worker_context(store):
    with trace_scope(store.context("mission", "first", 1)):
        with trace_span("planning"):
            trace_event("plan.proposed", destinations=["printer"])
            with pytest.raises(TimeoutError), trace_span("model_call"):
                raise TimeoutError("never persist raw exceptions or my-configured-credential")
        worker = bind_trace(current_trace(), lambda: trace_event("delayed", done=True))
    with trace_scope(store.context("other", "second", 2)):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
    assert current_trace() is None
    trace_event("disabled")
    report = exported(store, "mission")
    assert len(report["events"]) == 6
    assert {event["request_id"] for event in report["events"]} == {"first"}
    spans = [e for e in report["events"] if e["kind"] == "end"]
    assert spans[0]["data"]["outcome"] == "error" and spans[0]["data"]["error_type"] == "TimeoutError"
    assert all(e["data"]["duration_ms"] >= 0 for e in spans)
    assert spans[0]["data"]["parent_span_id"] == spans[1]["data"]["span_id"]
    assert not report["summary"]["unfinished_spans"]
    assert "never persist raw exceptions" not in json.dumps(report)
    assert store.close()
    reopened = TraceStore(store.path)
    try:
        assert exported(reopened, "mission")["events"] == report["events"]
        assert reopened.health()["unclean_shutdowns"] == 0
    finally:
        reopened.close()


def test_credential_image_redaction_and_explicit_truncation(store):
    context = store.context("mission", "request")
    context.emit(
        "input",
        text="go to my-configured-credential; password=hidden Bearer abc sk-test-123",
        api_key="hidden-key",
        headers={"authorization": "hidden-header"},
        image="data:image/png;base64,YQ==",
        endpoint="https://example.test?key=hidden-query",
        unknown=float("nan"),
        unsupported=object(),
    )
    context.emit("oversized", text="z" * 3000, values=list(range(70)))
    context.emit("large", values=["y" * 1900] * 64)
    report = exported(store)
    encoded = json.dumps(report)
    for secret in (
        "my-configured-credential",
        "hidden-key",
        "hidden-header",
        "hidden-query",
        "YQ==",
        "sk-test-123",
        "abc",
    ):
        assert secret not in encoded
    assert report["events"][0]["data"]["unknown"] is None
    assert report["events"][1]["truncated"]
    assert report["events"][2]["data"]["omitted"]
    assert report["health"]["truncated_events"] == 2
    assert "truncated_events" in report["summary"]["loss_counters_nonzero"]


@pytest.mark.parametrize("max_events,max_bytes,total", [(3, 1048576, 10), (1000, 65536, 120)])
def test_retention_limits_event_count_and_database_bytes(tmp_path, max_events, max_bytes, total):
    traces = TraceStore(tmp_path / "ring.sqlite3", max_events=max_events, max_bytes=max_bytes)
    try:
        context = traces.context("mission", "request")
        for number in range(total):
            context.emit("sample", number=number, text="x" * 1500)
            if number % 10 == 0:
                assert traces.flush()
        report = exported(traces)
        assert len(report["events"]) <= max_events
        assert report["events"][-1]["data"]["number"] == total - 1
        assert report["health"]["trimmed_events"] == total - len(report["events"])
        assert traces.path.stat().st_size <= max_bytes
        assert not traces.health()["write_errors"]
    finally:
        assert traces.close()


def test_writer_failure_is_visible_and_recovers_without_raising_into_caller(store, monkeypatch):
    original = store._write

    def fail(item):
        raise OSError("disk failed with my-configured-credential")

    monkeypatch.setattr(store, "_write", fail)
    store.context("mission", "request").emit("failed.write")
    assert store.flush()
    assert store.health()["write_errors"] == 1
    assert store.health()["last_error_type"] == "OSError"
    monkeypatch.setattr(store, "_write", original)
    store.context("mission", "request").emit("recovered")
    report = exported(store)
    assert report["health"]["write_errors"] == 1
    assert [e["stage"] for e in report["events"]] == ["recovered"]


def test_blocked_writer_and_full_queue_do_not_block_cancellation(tmp_path, monkeypatch):
    rig = _Rig(tmp_path)
    rig.traces.close()
    rig.traces = TraceStore(tmp_path / "small.sqlite3", queue_size=2)
    rig.commands = rig.new_controller()
    entered, release, canceled = threading.Event(), threading.Event(), threading.Event()
    original = rig.traces._write

    def blocked(item):
        entered.set()
        assert release.wait(5)
        original(item)

    try:
        rig.start()
        rig.client.accept()
        assert rig.traces.flush()
        monkeypatch.setattr(rig.traces, "_write", blocked)
        context = rig.traces.context("diagnostic", "diagnostic")
        context.emit("block")
        assert entered.wait(2)
        for _ in range(5):
            context.emit("overflow")

        def stop():
            rig.commands.handle("stop")
            canceled.set()

        thread = threading.Thread(target=stop)
        thread.start()
        assert canceled.wait(1), "cancel callback waited on the blocked diagnostic writer"
        thread.join(1)
        assert rig.client.handles[0].cancel_calls == 1 and rig.commands.busy
        assert rig.traces.health()["dropped_events"] > 0
        assert not rig.traces.flush(timeout=0.01)
        assert not rig.traces.close(timeout=0.01)
    finally:
        release.set()
        rig.close()


def test_trace_storage_failure_does_not_change_successful_mission(tmp_path, monkeypatch):
    rig = _Rig(tmp_path)

    def fail(item):
        raise sqlite3.OperationalError("trace disk unavailable")

    monkeypatch.setattr(rig.traces, "_write", fail)
    try:
        _control(rig, "control_ordered_mission")
        assert all(c["passed"] for c in rig.checks)
        assert rig.traces.flush()
        assert rig.traces.health()["write_errors"] > 0
        assert rig.state == "succeeded" and len(rig.client.goals) == 2
    finally:
        rig.close()


def test_shutdown_idle_does_not_overwrite_a_completed_mission_trace(tmp_path):
    rig = _Rig(tmp_path)
    try:
        _control(rig, "control_ordered_mission")
        rig.commands.close()
        assert rig.state == "idle"  # Existing ROS status behavior remains unchanged.
        assert exported(rig.traces)["summary"]["last_status"]["state"] == "succeeded"
    finally:
        rig.close()


def test_late_model_reply_remains_in_original_mission_after_new_request(tmp_path):
    rig = _Rig(tmp_path)

    def replace_mission():
        rig.model.before = lambda: None
        rig.commands.handle("stop")
        rig.commands.handle("Visit printer and cupboard again")

    rig.model.before = replace_mission
    try:
        rig.start()
        report = exported(rig.traces)
        roots = [
            e["mission_id"] for e in report["events"] if e["stage"] == "instruction" and e["data"]["text"] != "stop"
        ]
        assert len(roots) == 2 and roots[0] != roots[1]
        old, new = (read_trace(rig.traces.path, root) for root in roots)
        assert old["summary"]["last_status"]["state"] == "canceled"
        assert any(e["stage"] == "plan.proposed" for e in old["events"])
        assert not any(e["stage"] == "nav2.dispatch" for e in old["events"])
        assert any(e["stage"] == "nav2.dispatch" for e in new["events"])
        assert len(rig.client.goals) == 1
    finally:
        rig.close()


def test_ambiguity_selection_and_stop_share_the_original_mission(tmp_path):
    rig = _Rig(tmp_path)
    try:
        evidence = rig.memory_destination()
        rig.advance(0.1)  # Distinct capture identity; equal robot/camera/timestamp would replace the first memory.
        memory = Memory.create("fault-robot", "front", rig.stamp, Pose(8, 2, map_id="office"), evidence, "printer")
        memory = replace(memory, confidence=1.0, localization_checked=True)
        rig.store.upsert(
            [memory.with_embedding(rig.embedder.embed_text(["printer"])[0], rig.embedder.model_name, kind="caption")]
        )
        rig.start()
        assert rig.state == "ambiguous" and not rig.client.goals
        rig.commands.handle("option two")
        rig.drain()
        rig.client.accept()
        rig.commands.handle("stop")
        rig.finish(5)
        trace = exported(rig.traces)
        inputs = [e for e in trace["events"] if e["stage"] == "instruction"]
        assert len(inputs) == 3 and len({e["mission_id"] for e in inputs}) == 1
        assert len({e["request_id"] for e in inputs}) == 3
        assert trace["summary"]["last_status"]["state"] == "canceled"
        assert any(e["stage"] == "destination.selected" for e in trace["events"])
    finally:
        rig.close()


def test_extreme_text_and_nested_values_are_bounded(store):
    store.context("mission", "request").emit(
        "large", text="x" * 100000, nested={"a": {"b": {"c": {"d": {"e": {"f": {"g": "deep"}}}}}}}
    )
    trace = exported(store)
    assert trace["events"][0]["truncated"]
    assert "x" * 100 not in json.dumps(trace)
    assert "depth limit" in json.dumps(trace)


@pytest.mark.parametrize(
    "case",
    [
        "control_ordered_mission",
        "control_visual_arrival",
        "planner_malformed",
        "review_reject",
        "nav_lost_result",
        "nav_timeout_late_success",
    ],
)
def test_fault_artifacts_explain_plan_visual_checks_and_transport(case):
    result = run_faults(cases=[case])["results"][0]
    assert result["passed"], result
    trace = result["trace"]
    stages = {e["stage"] for e in trace["events"]}
    assert {"instruction", "planning", "model_call", "status"} <= stages
    if case == "control_visual_arrival":
        assert {
            "retrieval.candidates",
            "candidate.verdict",
            "arrival.observation_accepted",
            "arrival.verdict",
        } <= stages
        assert trace["summary"]["last_status"]["state"] == "succeeded"
    if case == "control_ordered_mission":
        goals = [e for e in trace["events"] if e["stage"] == "nav2.dispatch"]
        assert [e["step"] for e in goals] == [1, 2]
        assert goals[0]["mission_id"] == goals[1]["mission_id"]
        assert goals[0]["request_id"] != goals[1]["request_id"]
    if case == "nav_lost_result":
        assert any(span["stage"] == "navigation" for span in trace["summary"]["unfinished_spans"])
    if case == "nav_timeout_late_success":
        assert trace["summary"]["last_status"]["state"] == "canceled"
        assert "nav2.cancel_acknowledgement" in stages
    assert trace["summary"]["provider_response_usage"]["cost_usd"]["total"] is None


def test_reported_http_usage_and_credentials_without_recording_payloads(store):
    transport = FakeTransport(
        [
            (500, {}, {"error": "transient"}),
            (
                200,
                {},
                {
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "total_tokens": 15,
                        "cost_usd": 0.002,
                    }
                },
            ),
        ]
    )
    endpoint = Endpoint.build(
        "https://host.test?key=url-secret",
        "/chat",
        "custom-credential",
        2,
        transport,
        RetryPolicy(attempts=2),
        lambda _: None,
        None,
    )
    with trace_scope(store.context("mission", "request")):
        endpoint.post({"model": "test-model", "messages": ["private-system-prompt"]})
        trace_event("status", message="custom-credential")
    report = exported(store)
    end = next(e for e in report["events"] if e["kind"] == "end")
    assert end["data"]["attempts"] == 2
    assert end["data"]["usage"]["total_tokens"] == 15
    assert report["summary"]["provider_response_usage"]["cost_usd"]["known_sum"] == 0.002
    for value in ("custom-credential", "url-secret", "private-system-prompt"):
        assert value not in json.dumps(report)
    with trace_scope(store.context("mission", "request")), pytest.raises(ProviderError):
        Endpoint.build(
            "https://host.test",
            "/chat",
            None,
            2,
            FakeTransport([(400, {}, {"error": "secret-response"})]),
            None,
            lambda _: None,
            None,
        ).post({})
    assert exported(store)["summary"]["provider_response_usage"]["cost_usd"]["total"] is None


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"usage": None},
        {"usage": {"prompt_tokens": True, "completion_tokens": -1, "total_tokens": "12", "cost_usd": float("nan")}},
    ],
)
def test_absent_or_invalid_usage_is_unknown(body):
    assert all(value is None for value in provider_usage(body).values())


def test_gemini_usage_and_unspecified_currency_cost():
    assert (
        provider_usage({"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7}})[
            "total_tokens"
        ]
        == 7
    )
    assert provider_usage({"usage": {"cost": 0.1}})["cost_usd"] is None


def test_process_exit_leaves_incomplete_trace_and_reopen_reports_it(tmp_path):
    path = tmp_path / "interrupted.sqlite3"
    code = """
import os,sys
from placecell.tracing import TraceStore,trace_scope,trace_span
store=TraceStore(sys.argv[1])
with trace_scope(store.context('mission','request')):
    with trace_span('model_call'):
        assert store.flush()
        os._exit(17)
"""
    result = subprocess.run(  # noqa: S603 -- fixed child program, current interpreter, temporary trace database
        [sys.executable, "-c", code, str(path)], check=False, capture_output=True, timeout=10
    )
    assert result.returncode == 17, result.stderr
    report = read_trace(path, "mission")
    assert report["summary"]["capture_status"] == "open_or_unclean"
    assert report["summary"]["unfinished_spans"][0]["stage"] == "model_call"
    traces = TraceStore(path)
    try:
        assert exported(traces)["health"]["unclean_shutdowns"] == 1
    finally:
        assert traces.close()


def test_export_cli_read_only_and_no_overwrite(store, tmp_path, capsys):
    store.context("mission", "request").emit("status", state="failed")
    exported(store)
    before = store.path.read_bytes()
    output = tmp_path / "export" / "mission.json"
    args = ["--database", str(store.path), "--mission-id", "mission", "--output", str(output)]
    assert main(args) == 0
    assert json.loads(output.read_text())["summary"]["last_status"]["state"] == "failed"
    assert store.path.read_bytes() == before
    assert json.loads(capsys.readouterr().out)["events"] == 1
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["--database", str(store.path), "--mission-id", "unknown", "--output", str(tmp_path / "unknown.json")])
    assert error.value.code == 2
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        read_trace(missing)
    assert not missing.exists()


@pytest.mark.parametrize("options", [{"max_events": 0}, {"max_bytes": 10}, {"queue_size": False}])
def test_invalid_trace_limits(tmp_path, options):
    with pytest.raises(ValidationError):
        TraceStore(tmp_path / "invalid.sqlite3", **options)


def test_schema_isolation_and_single_writer(store, tmp_path):
    with pytest.raises(BlockingIOError):
        TraceStore(store.path)
    other = tmp_path / "other.sqlite3"
    with sqlite3.connect(other) as db:
        db.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(ValidationError):
        TraceStore(other)
    with pytest.raises(ValidationError):
        read_trace(other)
    with pytest.raises(ValidationError):
        TraceStore(":memory:")
    assert store.close()
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE trace_meta SET value='2' WHERE key='schema_version'")
    with pytest.raises(ValidationError):
        read_trace(store.path)
    with pytest.raises(ValidationError):
        TraceStore(store.path)


def test_ros_trace_configuration_and_configured_secret_redaction(tmp_path, monkeypatch):
    assert build_trace_store({"mission_trace_path": ""}) is None
    monkeypatch.setenv("TRACE_TEST_KEY", "example-configured-value")
    traces = build_trace_store(
        {
            "mission_trace_path": str(tmp_path / "ros.sqlite3"),
            "mission_trace_max_events": 5,
            "mission_trace_max_bytes": 1048576,
            "mission_trace_queue_size": 5,
            "api_key_env": "TRACE_TEST_KEY",
        }
    )
    try:
        traces.context("mission", "request").emit("instruction", text="example-configured-value")
        assert "example-configured-value" not in json.dumps(exported(traces))
    finally:
        traces.close()
