from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from placecell.errors import ProviderError, ValidationError
from placecell.evaluation_budget import ENDPOINT, EvaluationBudget, EvaluationStoppedError
from placecell.live_evaluation import MeteredChat, camera_smoke, main, preflight, read_key, run_planning
from placecell.mission_evaluation import load_dataset
from placecell.missions import MissionPlanner, PlanReviewAgent
from tests.conftest import FakeTransport

DATA = Path(__file__).resolve().parents[1] / "evaluation/missions/baseline-v1.json"
PAYLOAD = {"model": "test", "max_tokens": 100, "messages": [{"role": "user", "content": "go to printer"}]}


def budget(tmp_path, transport=None, **kwargs):
    return EvaluationBudget(
        tmp_path / "requests.jsonl",
        transport=transport,
        **{"max_requests": 100, "max_usd": 0.5, "max_seconds": 10, **kwargs},
    )


def response(cost=0.001):
    return (
        200,
        {},
        {"id": "gen-test", "model": "test", "provider": "test-provider", "usage": {"cost": cost, "prompt_tokens": 10}},
    )


def test_reservation_written_before_send_and_no_secrets_in_ledger(tmp_path):
    secret = "sk-or-fake-private-credential"

    class Transport:
        def post_json(self, url, headers, payload, timeout_s):
            row = json.loads((tmp_path / "requests.jsonl").read_text())
            assert row["event"] == "reserved" and row["reserved_usd"] > 0
            assert headers["Authorization"] == secret
            assert payload["provider"]["max_price"]["completion"] == 5
            assert timeout_s <= 10
            return response()

    b = budget(tmp_path, Transport())
    b.post_json(ENDPOINT, {"Authorization": secret}, PAYLOAD, 30)
    assert b.summary()["known_cost_usd"] == 0.001
    assert b.charged_or_reserved == pytest.approx(0.001)
    assert secret not in b.path.read_text() and "go to printer" not in b.path.read_text()
    assert [json.loads(x)["event"] for x in b.path.read_text().splitlines()] == ["reserved", "completed"]
    with pytest.raises(FileExistsError):
        budget(tmp_path)


@pytest.mark.parametrize("cost", [None, True, -1, float("nan"), float("inf"), "0.1"])
def test_unknown_cost_stops_before_next_call_and_keeps_reservation(tmp_path, cost):
    t = FakeTransport([response(cost)])
    b = budget(tmp_path, t)
    b.post_json(ENDPOINT, {}, PAYLOAD, 10)
    assert b.stop_reason == "unknown_cost" and b.charged_or_reserved > 0 and b.known_cost == 0
    with pytest.raises(EvaluationStoppedError):
        b.post_json(ENDPOINT, {}, PAYLOAD, 10)
    assert len(t.requests) == 1 and b.summary()["unknown_cost_requests"] == 1


@pytest.mark.parametrize("limit,value", [("max_requests", 1), ("max_usd", 0.02)])
def test_budget_limits_stop_before_another_request(tmp_path, limit, value):
    t = FakeTransport([response(0.01)])
    b = budget(tmp_path, t, **{limit: value})
    b.post_json(ENDPOINT, {}, PAYLOAD, 10)
    with pytest.raises(EvaluationStoppedError):
        b.post_json(ENDPOINT, {}, PAYLOAD, 10)
    assert len(t.requests) == 1


def test_expired_wall_budget_and_endpoint_refused_without_send(tmp_path):
    t = FakeTransport([])
    b = budget(tmp_path, t)
    with pytest.raises(ValidationError):
        b.post_json("https://elsewhere.invalid/chat/completions", {}, PAYLOAD, 10)
    b.deadline = time.monotonic() - 1
    with pytest.raises(EvaluationStoppedError, match="wall_time"):
        b.post_json(ENDPOINT, {}, PAYLOAD, 10)
    assert not t.requests


def test_provider_above_reserved_cost_stops_and_records_actual_bill(tmp_path):
    b = budget(tmp_path, FakeTransport([response(0.2)]))
    b.post_json(ENDPOINT, {}, PAYLOAD, 10)
    assert b.known_cost == 0.2 and b.charged_or_reserved == pytest.approx(0.2)
    assert b.stop_reason == "provider_exceeded_reservation"


def test_network_failure_keeps_unknown_cost_and_redacts_exception(tmp_path):
    class Transport:
        def post_json(self, *args):
            raise ProviderError("private response sk-or-fake-secret")

    b = budget(tmp_path, Transport())
    with pytest.raises(ProviderError):
        b.post_json(ENDPOINT, {}, PAYLOAD, 10)
    assert b.stop_reason == "transport_or_accounting_error"
    assert "sk-or-" not in b.path.read_text() and b.records[0]["cost_usd"] is None


def test_concurrent_calls_cannot_oversubscribe_request_budget(tmp_path):
    t = FakeTransport([response()])
    b = budget(tmp_path, t, max_requests=1)
    outcomes = []

    def send():
        try:
            b.post_json(ENDPOINT, {}, PAYLOAD, 10)
            outcomes.append("sent")
        except EvaluationStoppedError:
            outcomes.append("stopped")

    threads = [threading.Thread(target=send) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert outcomes.count("sent") == len(t.requests) == 1 and outcomes.count("stopped") == 9


def test_receipt_write_failure_stops_spending_after_successful_request(tmp_path, monkeypatch):
    t = FakeTransport([response()])
    b = budget(tmp_path, t)
    append = b._append

    def fail_receipt(row):
        if row["event"] == "completed":
            raise OSError("disk full")
        append(row)

    monkeypatch.setattr(b, "_append", fail_receipt)
    with pytest.raises(OSError, match="disk full"):
        b.post_json(ENDPOINT, {}, PAYLOAD, 10)
    assert b.stop_reason == "accounting_write_error" and b.known_cost == 0.001
    with pytest.raises(EvaluationStoppedError):
        b.post_json(ENDPOINT, {}, PAYLOAD, 10)
    assert len(t.requests) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"max_tokens": 2049},
        {"max_tokens": True},
        {"stream": True},
        {"messages": [{"role": "user", "content": "x" * 32769}]},
    ],
)
def test_unbounded_payload_rejected_before_spending(tmp_path, change):
    t = FakeTransport([])
    b = budget(tmp_path, t)
    with pytest.raises(ValidationError):
        b.post_json(ENDPOINT, {}, {**PAYLOAD, **change}, 10)
    assert not t.requests and not b.records


@pytest.mark.parametrize("url", ["https://example.invalid/picture.jpg", "data:image/png;base64," + "x" * 2_000_000])
def test_remote_or_oversized_image_refused(url):
    p = {**PAYLOAD, "messages": [{"content": [{"type": "image_url", "image_url": {"url": url}}]}]}
    with pytest.raises(ValidationError):
        EvaluationBudget.prepare(p)


def test_inline_image_cost_is_reserved_without_mutating_input():
    url = "data:image/png;base64,AAA="
    p = {**PAYLOAD, "messages": [{"content": [{"type": "image_url", "image_url": {"url": url}}]}]}
    outgoing, amount = EvaluationBudget.prepare(p)
    assert outgoing["messages"][0]["content"][0]["image_url"]["url"] == url
    assert amount > 0.04
    with pytest.raises(ValidationError):
        EvaluationBudget.prepare({**p, "messages": p["messages"] * 3})


@pytest.mark.parametrize(
    "kwargs",
    [{"max_requests": 0}, {"max_requests": True}, {"max_usd": float("nan")}, {"max_usd": -1}, {"max_seconds": 0}],
)
def test_invalid_budget_fails_before_creating_ledger(tmp_path, kwargs):
    with pytest.raises(ValidationError):
        budget(tmp_path, **kwargs)
    assert not (tmp_path / "requests.jsonl").exists()


def tool_reply(name, arguments):
    return (
        200,
        {},
        {
            "usage": {"cost": 0.001},
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    },
                }
            ],
        },
    )


def planner(b):
    return MissionPlanner(
        MeteredChat("planner", b, "sk-or-fake", "test"),
        PlanReviewAgent(MeteredChat("reviewer", b, "sk-or-fake", "test")),
    )


def test_live_runner_repeats_real_planner_and_review_without_label_leak(tmp_path):
    d = load_dataset(DATA)
    d = replace(d, cases=(replace(d.cases[0], scenario="SECRET_SCENARIO", destinations=(("SECRET_LABEL",),)),))
    replies = [
        tool_reply(
            "propose_navigation_plan", {"decision": "ready", "destinations": ["printer"], "message": "Visit printer"}
        ),
        tool_reply("review_navigation_plan", {"decision": "approve", "message": "Order preserved"}),
    ] * 2
    t = FakeTransport(replies)
    b = budget(tmp_path, t)
    result = run_planning(d, planner(b), b, tmp_path, split="development", repeats=2, configuration={})
    assert result["plan"] == {"eligible": 2, "assessed": 2, "passed": 0, "failed": 2}
    assert result["completed"] and not result["all_plans_match_labels"]
    assert len(t.requests) == 4 and result["budget"]["known_cost_usd"] == 0.004
    assert "SECRET_" not in json.dumps(t.requests)
    assert [r["stage"] for r in b.records] == ["planner", "reviewer"] * 2
    for index in [1, 2]:
        report = json.loads((tmp_path / f"report-{index:02}.json").read_text())
        assert report["execution"]["unassessed"] == 1
        assert report["cases"][0]["plan_destinations"] == ["printer"]


def test_interruption_preserves_denominators_and_does_not_retry(tmp_path):
    d = load_dataset(DATA)
    b = budget(tmp_path, FakeTransport([(401, {}, {"error": "sk-or-private"})]))
    r = run_planning(d, planner(b), b, tmp_path, split="development", repeats=2, configuration={})
    assert r["plan"]["eligible"] == r["plan"]["failed"] == 48
    assert r["attempted_trials"] == 1 and r["unrun_trials"] == 47 and not r["completed"]
    assert len(b.records) == 1 and r["budget"]["stop_reason"] == "http_401"
    assert "sk-or-private" not in (tmp_path / "trials-01.json").read_text()
    assert json.loads((tmp_path / "report-02.json").read_text())["missing_trials"] == 24


def test_held_out_never_falls_back_to_draft_development():
    d = load_dataset(DATA)
    with pytest.raises(ValidationError, match="no cases"):
        preflight(d, "held_out")
    groups = {k: {**v, "split": "held_out"} for k, v in d.groups.items()}
    with pytest.raises(ValidationError, match="human-reviewed"):
        preflight(replace(d, groups=groups), "held_out")
    assert preflight(replace(d, groups=groups, label_status="human_reviewed"), "held_out")["cases"] == 24


def test_private_key_and_cli_preflight_need_no_model_calls(tmp_path):
    key = tmp_path / "key"
    key.write_text("sk-or-fake-key")
    key.chmod(0o644)
    with pytest.raises(ValidationError, match="private"):
        read_key(key)
    key.chmod(0o600)
    assert read_key(key) == "sk-or-fake-key"
    key.write_text("bad")
    with pytest.raises(ValidationError):
        read_key(key)
    out = tmp_path / "preflight"
    main(["preflight", "--dataset", str(DATA), "--split", "development", "--output", str(out)])
    assert json.loads((out / "preflight.json").read_text())["cases"] == 24
    assert not (out / "requests.jsonl").exists()
    with pytest.raises(FileExistsError):
        main(["preflight", "--dataset", str(DATA), "--split", "development", "--output", str(out)])


def test_camera_smoke_does_not_turn_failure_into_accuracy(tmp_path):
    path = tmp_path / "observations.jsonl"
    path.write_text("")
    b = budget(tmp_path, FakeTransport([]))
    r = camera_smoke(path, b, "sk-or-fake", tmp_path, "test")
    assert not r["passed"] and r["error_type"] == "StopIteration"
    assert r["requests"] == 0 and "no independent visual labels" in r["scope"]


def test_cli_live_run_uses_budget_and_returns_failure_for_label_mismatch(tmp_path, monkeypatch):
    data = json.loads(DATA.read_text())
    data["cases"] = data["cases"][:1]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(data))
    key = tmp_path / "key"
    key.write_text("sk-or-fake-key")
    key.chmod(0o600)
    fake = FakeTransport(
        [tool_reply("propose_navigation_plan", {"decision": "reject", "destinations": [], "message": "No visit"})]
    )
    monkeypatch.setattr("placecell.evaluation_budget.UrllibTransport", lambda: fake)
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as status:
        main(
            [
                "run",
                "--dataset",
                str(dataset),
                "--split",
                "development",
                "--output",
                str(out),
                "--key-file",
                str(key),
                "--max-usd",
                "0.5",
                "--max-requests",
                "2",
                "--repeats",
                "1",
            ]
        )
    assert status.value.code == 1 and len(fake.requests) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed"] and summary["plan"]["failed"] == 1
    assert summary["budget"]["known_cost_usd"] == 0.001
    assert "sk-or-" not in (out / "summary.json").read_text()


def test_camera_smoke_accepts_valid_negative_verdict_without_accuracy_claim(tmp_path):
    from PIL import Image

    from placecell import Evidence, EvidenceKind, Pose
    from placecell.pipeline import Observation
    from placecell.recordings import RecordingWriter

    image = tmp_path / "picture.png"
    Image.new("RGB", (10, 10)).save(image)
    writer = RecordingWriter(tmp_path / "recording")
    writer.append(Observation("r", "c", 100, Pose(0, 0), Evidence(EvidenceKind.FRAME, str(image))))

    def completion(text):
        return 200, {}, {"usage": {"cost": 0.001}, "choices": [{"finish_reason": "stop", "message": {"content": text}}]}

    t = FakeTransport([completion("Dark image"), completion('{"result":"not_matched","reason":"No visible printer"}')])
    b = budget(tmp_path, t)
    r = camera_smoke(writer.directory / "observations.jsonl", b, "sk-or-fake", tmp_path, "test")
    assert r["passed"] and r["verdict"]["result"] == "not_matched" and len(t.requests) == 2
    assert "no independent visual labels" in r["scope"]


@pytest.mark.parametrize("extra", [[], ["--repeats", "0"]])
def test_cli_missing_budget_or_invalid_repeats_do_not_start_calls(tmp_path, extra):
    with pytest.raises(ValidationError):
        main(["run", "--dataset", str(DATA), "--split", "development", "--output", str(tmp_path / "out"), *extra])
    assert not (tmp_path / "out").exists()
