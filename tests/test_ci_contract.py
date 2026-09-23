"""Regression checks for previously omitted integration work and failure propagation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from scripts import check_distribution

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["ci", "simulation"])
def test_core_changes_cannot_be_filtered_out_of_required_workflows(name):
    workflow = yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
    events = workflow["on"]
    assert {"pull_request", "push", "merge_group", "workflow_dispatch"} <= events.keys()
    # No workflow/job path filter can omit a new planner, store, ROS adapter or dependency file.
    assert events["pull_request"] in (None, {}) and events["merge_group"] in (None, {})
    assert events["push"] == {"branches": ["main"]}
    for job in workflow["jobs"].values():
        assert "if" not in job and "continue-on-error" not in job


def test_supported_python_versions_install_packages_and_ros_contracts_are_a_gate():
    core = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    for name in ("checks", "package"):
        assert core["jobs"][name]["strategy"]["matrix"]["python-version"] == ["3.10", "3.11", "3.12"]
    sim = yaml.safe_load((ROOT / ".github/workflows/simulation.yml").read_text())
    steps = sim["jobs"]["office"]["steps"]
    for command in ("check-operator", "check-cancel", "check", "check-nav"):
        step = next(step for step in steps if step.get("run") == f"simulation/sim {command}")
        assert "if" not in step and "continue-on-error" not in step


def test_failed_package_install_cannot_produce_a_successful_gate(tmp_path, monkeypatch):
    dist, wheels, output = tmp_path / "dist", tmp_path / "wheels", tmp_path / "result"
    dist.mkdir()
    wheels.mkdir()
    (dist / "placecell-0.0.0-py3-none-any.whl").touch()
    (dist / "placecell-0.0.0.tar.gz").touch()
    monkeypatch.setattr(
        sys, "argv", ["check", "--dist-dir", str(dist), "--wheelhouse", str(wheels), "--output", str(output)]
    )
    monkeypatch.setattr(check_distribution.venv.EnvBuilder, "create", lambda *_: None)
    calls = []

    def failed_install(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(check_distribution, "run", failed_install)
    assert check_distribution.main() == 1
    report = json.loads((output / "report.json").read_text())
    assert not report["passed"] and len(calls) == 2
    assert {row["kind"] for row in report["results"]} == {"wheel", "sdist"}
    assert all(not row["passed"] and "error" in row for row in report["results"])


def test_subprocess_failure_retains_install_output(tmp_path):
    log = tmp_path / "install.log"
    with pytest.raises(subprocess.CalledProcessError):
        check_distribution.run(
            [sys.executable, "-c", "import sys; print('installation failure'); sys.exit(7)"],
            cwd=tmp_path,
            env={},
            log=log,
        )
    assert "installation failure" in log.read_text()
