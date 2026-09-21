from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from placecell.errors import ValidationError
from placecell.mission_evaluation import load_dataset, main, run_scripted, score_trials

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation/missions/baseline-v1.json"
REPLIES = ROOT / "evaluation/missions/scripted-replies-v1.json"


@pytest.fixture
def dataset():
    return load_dataset(DATA)


@pytest.fixture
def trials(dataset):
    return run_scripted(dataset, json.loads(REPLIES.read_text()), run_id="test")


def write_json(tmp_path, data, name="dataset.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def execution(outcome="succeeded", dispatched=("printer-a",), confirmed=("printer-a",)):
    return {
        "outcome": outcome,
        "dispatched_targets": list(dispatched),
        "confirmed_targets": list(confirmed),
        "latency_ms": 200.0,
    }


def test_offline_baseline_is_not_model_or_execution_accuracy(dataset, trials):
    report = score_trials(dataset, trials)
    assert report["plan"] == {
        "eligible": 24,
        "assessed": 24,
        "passed": 24,
        "failed": 0,
        "unassessed": 0,
        "pass_rate": 1.0,
    }
    assert report["execution"]["unassessed"] == 24
    assert report["execution"]["pass_rate"] is None
    assert report["mission_completion"]["pass_rate"] is None
    assert report["appropriate_abstention"]["pass_rate"] is None
    assert report["total_cost_usd"] == 0.0
    assert report["label_status"] == "draft" and report["runner"] == "scripted"
    assert "not model accuracy" in report["claim"]
    assert report["plan_latency"]["count"] == 24
    assert report["execution_latency"]["count"] == 0


def test_expected_labels_do_not_supply_model_outputs(dataset):
    changed = replace(dataset.cases[0], destinations=(("intentionally incorrect label",),))
    dataset = replace(dataset, cases=(changed, *dataset.cases[1:]))
    trials = run_scripted(dataset, json.loads(REPLIES.read_text()), run_id="no-label-leak")
    assert trials["trials"][0]["plan"]["destinations"] == ["printer"]
    assert score_trials(dataset, trials)["cases"][0]["plan"] == "failed"


def test_order_and_repeats_are_scored_exactly(dataset, trials):
    rows = {row["case_id"]: row for row in trials["trials"]}
    rows["chain"]["plan"]["destinations"].reverse()
    rows["repeat"]["plan"]["destinations"].pop()
    report = score_trials(dataset, trials)
    assert report["plan"]["failed"] == 2
    assert report["categories"]["repeated_visit"]["plan"]["failed"] == 1


def test_missing_errors_and_timeouts_remain_in_denominator(dataset):
    fixtures = json.loads(REPLIES.read_text())
    fixtures["cases"]["single"]["planner"] = {"error": "timeout"}
    fixtures["cases"]["polite"]["planner"] = {"content": "malformed prose", "tool_calls": []}
    fixtures["cases"]["chain"]["reviewer"] = {"error": "provider"}
    trials = run_scripted(dataset, fixtures, run_id="faults")
    trials["trials"].pop()
    report = score_trials(dataset, trials)
    assert report["plan"]["eligible"] == 24 and report["plan"]["failed"] == 4
    assert report["plan"]["pass_rate"] == 20 / 24
    assert report["timeouts"] == 1 and report["errors"] == 2 and report["missing_trials"] == 1
    assert report["total_cost_usd"] is None
    assert report["plan_latency"]["count"] == 23


@pytest.mark.parametrize(
    "dispatched,confirmed,outcome,wrong,false_success",
    [
        (("other-printer",), ("other-printer",), "succeeded", True, True),
        (("printer-a",), (), "succeeded", False, True),
        ((), (), "succeeded", False, True),
        (("printer-a", "printer-a"), ("printer-a", "printer-a"), "succeeded", True, True),
        (("printer-a",), ("different-instance",), "succeeded", False, True),
        (("printer-a",), ("printer-a",), "timeout", False, False),
    ],
)
def test_execution_errors_cannot_hide_behind_a_correct_plan(
    dataset, trials, dispatched, confirmed, outcome, wrong, false_success
):
    trials["trials"][0]["execution"] = execution(outcome, dispatched, confirmed)
    report = score_trials(dataset, trials)
    assert report["plan"]["passed"] == 24
    assert report["execution"]["failed"] == 1
    assert report["wrong_destination_dispatch_trials"] == int(wrong)
    assert report["false_success_trials"] == int(false_success)
    assert report["cases"][0]["execution_outcome"] == outcome
    assert report["execution_timeouts"] == int(outcome == "timeout")


def test_wrong_order_in_dispatched_targets_is_an_execution_failure(dataset, trials):
    row = next(row for row in trials["trials"] if row["case_id"] == "chain")
    row["execution"] = execution(dispatched=("shelf-a", "printer-a"), confirmed=("shelf-a", "printer-a"))
    assert score_trials(dataset, trials)["wrong_destination_dispatch_trials"] == 1


def test_correct_partial_trip_failure_is_not_wrong_destination_motion(dataset, trials):
    row = next(row for row in trials["trials"] if row["case_id"] == "chain")
    row["execution"] = execution(outcome="failed")
    report = score_trials(dataset, trials)
    assert report["execution"]["failed"] == 1 and report["wrong_destination_dispatch_trials"] == 0


def test_unnecessary_abstention_and_dispatch_after_rejection(dataset, trials):
    trials["trials"][0]["plan"].update(decision="reject", destinations=[])
    trials["trials"][0]["execution"] = execution()
    report = score_trials(dataset, trials)
    assert report["unnecessary_abstentions"] == 1
    assert report["dispatch_without_ready_plan_trials"] == 1
    assert report["execution"]["failed"] == 1


def test_nonmovement_case_cannot_claim_success_or_phantom_confirmation(dataset, trials):
    row = next(row for row in trials["trials"] if row["case_id"] == "negated")
    row["execution"] = execution(dispatched=(), confirmed=())
    assert score_trials(dataset, trials)["false_success_trials"] == 1
    row["execution"] = execution(outcome="reject", dispatched=(), confirmed=("printer-a",))
    assert score_trials(dataset, trials)["invalid_confirmation_trials"] == 1


def test_correct_abstention_differs_from_mission_completion(dataset, trials):
    trials["trials"][0]["execution"] = execution()
    missing = next(row for row in trials["trials"] if row["case_id"] == "missing")
    missing["execution"] = execution("not_found", (), ())
    report = score_trials(dataset, trials)
    assert report["mission_completion"]["passed"] == 1
    assert report["appropriate_abstention"]["passed"] == 1
    assert report["execution"]["passed"] == 2 and report["execution"]["unassessed"] == 22


def test_gazebo_runner_cannot_omit_execution_rows(dataset, trials):
    trials["runner"] = "gazebo_live_model"
    trials["trials"].pop()
    report = score_trials(dataset, trials)
    assert report["execution"]["failed"] == 24 and report["execution"]["unassessed"] == 0


def test_unknown_cost_is_not_zero(dataset, trials):
    trials["runner"] = "live_model"
    trials["trials"][0]["cost_usd"] = None
    trials["trials"][1]["cost_usd"] = 0.25
    report = score_trials(dataset, trials)
    assert report["known_cost_usd"] == 0.25 and report["total_cost_usd"] is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "fast"])
def test_invalid_measurements_are_rejected(dataset, trials, value):
    trials["trials"][0]["plan"]["latency_ms"] = value
    with pytest.raises(ValidationError):
        score_trials(dataset, trials)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(dataset_sha256="wrong"),
        lambda data: data.update(schema_version=True),
        lambda data: data.update(runner=[]),
        lambda data: data["trials"].append(copy.deepcopy(data["trials"][0])),
        lambda data: data["trials"][0].update(case_id="unknown"),
        lambda data: data["trials"][0]["plan"].update(status="timeout"),
        lambda data: data["trials"][0]["plan"].update(decision="reject"),
        lambda data: data["trials"][0]["plan"].update(unknown_field=True),
        lambda data: data["trials"][0].update(execution={}),
        lambda data: data.update(configuration=[]),
    ],
)
def test_malformed_or_wrong_dataset_trials_are_rejected(dataset, trials, mutation):
    mutation(trials)
    with pytest.raises(ValidationError):
        score_trials(dataset, trials)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(schema_version=2),
        lambda data: data.update(label_status="approved_by_model"),
        lambda data: data.update(groups=[]),
        lambda data: data.update(cases=[]),
        lambda data: data["groups"].append(copy.deepcopy(data["groups"][0])),
        lambda data: data["groups"][0].update(split="random_frames"),
        lambda data: data["groups"][1].update(recording_sha256=["bad"]),
        lambda data: data["groups"][1].update(recording_sha256=[]),
        lambda data: data["cases"].append(copy.deepcopy(data["cases"][0])),
        lambda data: data["cases"][0].update(group="missing"),
        lambda data: data["cases"][0].update(instruction=""),
        lambda data: data["cases"][0].update(context=[{}] * 21),
        lambda data: data["cases"][0]["expected"].update(decisions=["ready", "reject"]),
        lambda data: data["cases"][0]["expected"].update(destinations=[]),
        lambda data: data["cases"][0]["expected"].update(target_ids=[]),
        lambda data: data["cases"][0]["expected"].update(visual_required=1),
    ],
)
def test_bad_label_contracts_cannot_enter_a_baseline(tmp_path, mutation):
    data = json.loads(DATA.read_text())
    mutation(data)
    with pytest.raises(ValidationError):
        load_dataset(write_json(tmp_path, data))


@pytest.mark.parametrize("leak", ["layout", "recording", "instruction"])
def test_split_leakage_is_rejected(tmp_path, leak):
    data = json.loads(DATA.read_text())
    group = copy.deepcopy(data["groups"][1 if leak == "recording" else 0])
    group.update(id="held-out", split="held_out")
    if leak != "layout":
        group["layout_id"] = "new-layout"
    data["groups"].append(group)
    if leak == "instruction":
        case = copy.deepcopy(data["cases"][0])
        case.update(id="held-out-case", group="held-out")
        data["cases"].append(case)
    with pytest.raises(ValidationError, match=r"cross|recording"):
        load_dataset(write_json(tmp_path, data))


def test_empty_heldout_is_explicit_and_never_falls_back(dataset, trials):
    with pytest.raises(ValidationError, match="no cases"):
        score_trials(dataset, trials, split="held_out")
    with pytest.raises(ValidationError, match="unknown evaluation split"):
        score_trials(dataset, trials, split="random")


def test_scripted_response_coverage_and_reviewer_veto(dataset):
    fixtures = json.loads(REPLIES.read_text())
    fixtures["cases"]["single"]["reviewer"]["tool_calls"][0]["arguments"]["decision"] = "reject"
    trials = run_scripted(dataset, fixtures, run_id="veto")
    assert trials["trials"][0]["plan"]["decision"] == "reject"
    assert score_trials(dataset, trials)["plan"]["failed"] == 1
    fixtures["cases"].pop("single")
    with pytest.raises(ValidationError, match="exactly"):
        run_scripted(dataset, fixtures, run_id="missing-script")


def test_cli_reports_can_be_rescored_and_never_overwritten(tmp_path):
    output, trials, rescored = (tmp_path / name for name in ("report.json", "trials.json", "rescored.json"))
    args = [
        "scripted",
        "--dataset",
        str(DATA),
        "--replies",
        str(REPLIES),
        "--output",
        str(output),
        "--save-trials",
        str(trials),
    ]
    main(args)
    main(["score", "--dataset", str(DATA), "--trials", str(trials), "--output", str(rescored)])
    assert json.loads(output.read_text()) == json.loads(rescored.read_text())
    with pytest.raises(ValidationError, match="new files"):
        main(args)
    summary = tmp_path / "validated.json"
    main(["validate", "--dataset", str(DATA), "--output", str(summary)])
    assert json.loads(summary.read_text())["splits"] == {"development": 24, "held_out": 0}


def test_cli_saves_failure_evidence_and_exits_nonzero(tmp_path, trials):
    trials["trials"].pop()
    source = write_json(tmp_path, trials, "trials.json")
    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as error:
        main(["score", "--dataset", str(DATA), "--trials", str(source), "--output", str(output)])
    assert error.value.code == 1
    assert json.loads(output.read_text())["missing_trials"] == 1


@pytest.mark.parametrize(
    "arguments",
    [
        ["scripted"],
        ["score"],
        ["scripted", "--replies", str(REPLIES), "--split", "held_out"],
        ["validate", "--save-trials", "unused.json"],
    ],
)
def test_cli_rejects_incomplete_or_misleading_modes(tmp_path, arguments):
    with pytest.raises(ValidationError):
        main([*arguments, "--dataset", str(DATA), "--output", str(tmp_path / "output.json")])


def test_bad_and_oversized_json_files_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not JSON")
    with pytest.raises(ValidationError, match="invalid evaluation JSON"):
        load_dataset(path)
    path.write_text(" " * 4_000_001)
    with pytest.raises(ValidationError, match="4 MB"):
        load_dataset(path)
