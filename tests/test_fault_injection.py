from __future__ import annotations

import json
from dataclasses import replace

import pytest

from placecell import fault_injection as faults
from placecell.errors import ValidationError


@pytest.mark.parametrize("case", faults.CASES, ids=lambda case: case.id)
def test_fault_contract(case):
    report = faults.run_faults(cases=[case.id])
    result = report["results"][0]
    assert result["passed"], result
    assert report["summary"]["runs"] == report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 0
    assert report["paid_api_calls"] == report["cost_usd"] == 0
    assert result["checks"] and all(check["passed"] for check in result["checks"])
    assert report["sources_sha256"]["navigation.py"]
    json.dumps(report, allow_nan=False)


def test_timeout_late_success_never_dispatches_second_destination():
    result = faults.run_faults(cases=["nav_timeout_late_success"])["results"][0]
    assert result["final_state"] == "canceled"
    assert len(result["dispatch_attempts"]) == 1
    assert result["cancel_requests"] == 1
    assert not result["mission_owned"] and result["queued_tasks"] == 0
    states = [event["state"] for event in result["events"]]
    assert "uncertain" in states and "busy" in states
    assert not {"succeeded", "step_succeeded", "awaiting_observation"}.intersection(states)


def test_lost_result_retains_ownership_without_claiming_a_stop():
    result = faults.run_faults(cases=["nav_lost_result"])["results"][0]
    assert result["mission_owned"] and result["cancel_requests"] == 1
    assert len(result["dispatch_attempts"]) == 1
    assert not {"succeeded", "step_succeeded", "canceled"}.intersection(event["state"] for event in result["events"])


def test_failure_and_exception_remain_in_denominator_and_report(monkeypatch):
    def mismatch(rig, case):
        rig.check("injected bad outcome", "unsafe", "blocked")

    def broken(rig, case):
        raise RuntimeError("scenario failed before its assertions")

    monkeypatch.setattr(
        faults,
        "CASES",
        (
            replace(faults.CASES[0], id="mismatch", exercise=mismatch),
            replace(faults.CASES[0], id="broken", exercise=broken),
        ),
    )
    report = faults.run_faults(repeat=2)
    assert report["summary"]["runs"] == report["summary"]["failed"] == 4
    assert report["summary"]["passed"] == 0
    assert [r["iteration"] for r in report["results"]] == [1, 1, 2, 2]
    assert report["results"][1]["error"] == "RuntimeError: scenario failed before its assertions"
    assert not report["results"][0]["checks"][0]["passed"]


def test_no_checks_cannot_be_a_passing_scenario(monkeypatch):
    monkeypatch.setattr(faults, "CASES", (replace(faults.CASES[0], exercise=lambda *_: None),))
    assert faults.run_faults()["summary"]["failed"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cases": ["missing"]},
        {"cases": ["planner_timeout", "planner_timeout"]},
        {"repeat": 0},
        {"repeat": 1001},
        {"repeat": True},
        {"repeat": 1.5},
    ],
)
def test_invalid_selection_or_repeat(kwargs):
    with pytest.raises(ValidationError):
        faults.run_faults(**kwargs)


def test_cli_report_exit_code_and_no_overwrite(tmp_path, monkeypatch, capsys):
    path = tmp_path / "nested" / "report.json"
    assert faults.main(["--output", str(path), "--case", "planner_timeout", "--repeat", "2"]) == 0
    original = path.read_bytes()
    assert json.loads(original)["summary"]["runs"] == 2
    assert json.loads(capsys.readouterr().out)["passed"] == 2
    with pytest.raises(SystemExit) as exc:
        faults.main(["--output", str(path)])
    assert exc.value.code == 2 and path.read_bytes() == original

    def mismatch(rig, case):
        rig.check("failed assertion", 1, 0)

    monkeypatch.setattr(faults, "CASES", (replace(faults.CASES[0], exercise=mismatch),))
    failed_path = tmp_path / "failed.json"
    assert faults.main(["--output", str(failed_path)]) == 1
    assert json.loads(failed_path.read_text())["summary"]["failed"] == 1


@pytest.mark.parametrize(
    "args",
    [
        ["--repeat", "0"],
        ["--repeat", "1001"],
        ["--case", "unknown"],
        ["--case", "planner_timeout", "--case", "planner_timeout"],
    ],
)
def test_cli_rejects_invalid_selection(tmp_path, args):
    path = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        faults.main(["--output", str(path), *args])
    assert exc.value.code == 2 and not path.exists()


def test_execution_gate_excludes_components_and_requires_case_diversity():
    component = {"case_id": "depth", "scope": "sensor_boundary", "passed": True}
    mission = {"case_id": "one", "scope": "mission", "passed": True}
    gate = faults.execution_gate([component] * 1000 + [mission] * 1000)
    assert gate["runs"] == 1000 and gate["cases"] == 1
    assert gate["excluded_component_runs"] == 1000
    assert not gate["coverage_met"] and not gate["passed"]
    distinct = [{**mission, "case_id": str(i)} for i in range(100)]
    assert not faults.execution_gate(distinct)["passed"]
    assert faults.execution_gate(distinct * 10)["passed"]
    assert not faults.execution_gate(distinct * 10 + [{**component, "passed": False}])["passed"]
    assert not faults.execution_gate(distinct * 10 + [{**mission, "passed": False}])["passed"]


def test_execution_gate_cli_refuses_insufficient_passing_evidence(tmp_path, capsys):
    path = tmp_path / "gate.json"
    assert faults.main(["--output", str(path), "--case", "control_ordered_mission", "--execution-gate"]) == 1
    report = json.loads(path.read_text())
    assert report["summary"]["failed"] == 0
    assert report["execution_gate"]["runs"] == 1 and not report["execution_gate"]["passed"]
    assert json.loads(capsys.readouterr().out)["passed"] == 1
