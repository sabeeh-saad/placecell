"""Regression checks for previously omitted integration work and failure propagation."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from scripts import check_distribution

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ROS_CHECKS = (
    "build",
    "check-operator",
    "check-cancel",
    "check-sensors",
    "check-retention",
    "check-overload --samples 100",
)
BASH = shutil.which("bash")


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
    (job,) = sim["jobs"].values()
    commands = [step for step in job["steps"] if "run" in step]
    # Every push runs exactly the offline ROS checks, in order; Gazebo is left to e2e.yml.
    assert [step["run"] for step in commands] == [f"simulation/sim {check}" for check in OFFLINE_ROS_CHECKS]
    for step in commands:
        assert "if" not in step and "continue-on-error" not in step
    upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["if"] == "always()" and upload["with"]["path"] == "simulation/artifacts/"


def test_gazebo_end_to_end_runs_only_on_demand_and_always_cleans_up():
    e2e = yaml.safe_load((ROOT / ".github/workflows/e2e.yml").read_text())
    assert list(e2e["on"]) == ["workflow_dispatch"]
    live = e2e["on"]["workflow_dispatch"]["inputs"]["live"]
    assert live["type"] == "boolean" and live["default"] is False
    assert e2e["permissions"] == {"contents": "read"}
    (job,) = e2e["jobs"].values()
    steps = job["steps"]
    run = next(step for step in steps if "simulation/sim e2e" in step.get("run", ""))
    assert "if" not in run and "continue-on-error" not in run
    # The secret reaches the step only for a live request.
    assert run["env"]["OPENROUTER_API_KEY"].replace(" ", "") == "${{inputs.live&&secrets.OPENROUTER_API_KEY||''}}"
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["if"] == "always()" and upload["with"]["path"] == "simulation/artifacts/"
    stop = next(step for step in steps if step.get("run") == "simulation/sim stop")
    assert stop["if"] == "always()" and steps.index(stop) > steps.index(run)


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize(
    ("live", "key", "expected"),
    [
        ("true", "secret", "e2e --live key=set"),
        ("true", "", "e2e key=unset"),
        ("false", "secret", "e2e key=unset"),
        ("false", "", "e2e key=unset"),
    ],
)
def test_e2e_workflow_adds_live_only_with_the_input_and_a_key(tmp_path, live, key, expected):
    e2e = yaml.safe_load((ROOT / ".github/workflows/e2e.yml").read_text())
    (job,) = e2e["jobs"].values()
    script = next(step["run"] for step in job["steps"] if "simulation/sim e2e" in step.get("run", ""))
    stub = tmp_path / "simulation/sim"
    stub.parent.mkdir()
    stub.write_text(
        '#!/usr/bin/env bash\necho "$* key=$([[ -n ${OPENROUTER_API_KEY:-} ]] && echo set || echo unset)"\n'
    )
    stub.chmod(0o755)
    env = {**os.environ, "LIVE": live, "OPENROUTER_API_KEY": key}
    result = subprocess.run(  # noqa: S603 - the workflow's own step script against a local stub
        [BASH, "-e", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=True
    )
    assert result.stdout.splitlines()[-1] == expected
    assert ("::warning::" in result.stdout) == (live == "true" and not key)


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_live_e2e_without_a_key_refuses_before_building(tmp_path):
    sim = tmp_path / "simulation/sim"
    sim.parent.mkdir()
    shutil.copy2(ROOT / "simulation/sim", sim)
    calls = tmp_path / "docker-calls"
    docker = tmp_path / "bin/docker"
    docker.parent.mkdir()
    docker.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\nexit 1\n')
    docker.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    env["PATH"] = f"{docker.parent}{os.pathsep}{env.get('PATH', '')}"
    for arguments, code, message in ((["--live"], 1, "OPENROUTER_API_KEY"), (["--bogus"], 2, "Unknown e2e option")):
        result = subprocess.run(  # noqa: S603 - the repository's own launcher with a stub docker
            [BASH, str(sim), "e2e", *arguments], env=env, capture_output=True, text=True, timeout=30, check=False
        )
        assert result.returncode == code and message in result.stderr
    assert not calls.exists() and not (sim.parent / "artifacts").exists()


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


def offline_check(tmp_path, report, *, github, code=1):
    """Run simulation/sim check-cancel against a stub docker that writes report and exits with code."""
    sim = tmp_path / "simulation/sim"
    sim.parent.mkdir(exist_ok=True)
    for name in ("sim", "ci_annotation.py"):
        shutil.copy2(ROOT / "simulation" / name, sim.parent / name)
    docker = tmp_path / "bin/docker"
    docker.parent.mkdir(exist_ok=True)
    (tmp_path / "report.json").write_text(json.dumps(report))
    write = "" if report is None else f'cp "{tmp_path}/report.json" "${{mount%%:*}}/report.json"\n'
    docker.write_text(
        '#!/usr/bin/env bash\nwhile (($#)); do if [[ "$1" == -v ]]; then mount="$2"; fi; shift; done\n'
        f"{write}echo 'check output'\nexit {code}\n"
    )
    docker.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"}
    env["PATH"] = f"{docker.parent}{os.pathsep}{env.get('PATH', '')}"
    if github:
        env["GITHUB_ACTIONS"] = "true"
    return subprocess.run(  # noqa: S603 - the repository's own launcher with a stub docker
        [BASH, str(sim), "check-cancel"], env=env, capture_output=True, text=True, timeout=30, check=False
    )


FAILED_CANCEL = {
    "passed": False,
    "checks": ["stop while planner blocked; late reply discarded"],
    "error": (
        "Traceback (most recent call last):\n"
        '  File "/opt/placecell-sim/scripts/cancellation_test.py", line 300, in main\n'
        "    check.run(args.samples)\n"
        '  File "/opt/placecell-sim/scripts/cancellation_test.py", line 230, in blocked_model\n'
        '    assert handled_ms <= 500, f"stop took {handled_ms:.1f} ms"\n'
        "           ^^^^^^^^^^^^^^^^^\n"
        "AssertionError: stop while reviewer blocked took 612.5 ms to handle (limit 500 ms, 100%)\r\n"
    ),
    "trials": [
        {"id": 1, "mode": "planner", "passed": True},
        {"id": 2, "mode": "reviewer", "passed": False, "received_s": 1.0, "command_to_cancel_ms": 3.25},
    ],
    "capture_error": "writer stalled\rat close",
    "trace_health": {"dropped_events": 0, "write_errors": 2},
}


@pytest.mark.skipif(BASH is None or shutil.which("python3") is None, reason="needs bash and python3")
def test_failed_offline_check_reports_an_escaped_github_annotation(tmp_path):
    result = offline_check(tmp_path, FAILED_CANCEL, github=True)
    assert result.returncode == 1
    (line,) = [line for line in result.stdout.splitlines() if line.startswith("::")]
    assert line.startswith("::error title=check-cancel::exit code 1%0A")
    message = line.split("::", 2)[2]
    assert "\r" not in message and "\n" not in message
    # Every literal % is escaped, so the message decodes back exactly.
    assert not re.search(r"%(?!25|0A|0D)", message)
    decoded = message.replace("%0A", "\n").replace("%0D", "\r").replace("%25", "%")
    assert decoded.splitlines()[1:4] == [
        "AssertionError: stop while reviewer blocked took 612.5 ms to handle (limit 500 ms, 100%)",
        "cancellation_test.py:300 in main: check.run(args.samples)",
        'cancellation_test.py:230 in blocked_model: assert handled_ms <= 500, f"stop took {handled_ms:.1f} ms"',
    ]
    assert 'failing trial: {"command_to_cancel_ms": 3.25, "id": 2, "mode": "reviewer"}' in decoded
    assert 'trace_health: {"write_errors": 2}' in decoded
    assert "capture: writer stalled%0Dat close%0A" in message
    assert "checks passed: 1, last: stop while planner blocked" in decoded


@pytest.mark.skipif(BASH is None or shutil.which("python3") is None, reason="needs bash and python3")
def test_offline_check_annotation_is_bounded_and_only_on_github(tmp_path):
    huge = {**FAILED_CANCEL, "error": "AssertionError: " + "%\n" * 5000}
    line = offline_check(tmp_path, huge, github=True).stdout.splitlines()[-1]
    assert line.startswith("::error title=check-cancel::") and len(line.encode()) <= 4100
    # The cut never splits an escape sequence.
    assert line.endswith("%25 [truncated]") and not re.search(r"%(?!25|0A|0D)", line)
    missing = offline_check(tmp_path, None, github=True, code=124)
    assert missing.returncode == 124
    assert missing.stdout.splitlines()[-1] == (
        "::error title=check-cancel::exit code 124%0Ano report.json was written (480 s timeout)"
    )
    for code in (0, 1):
        local = offline_check(tmp_path, FAILED_CANCEL, github=False, code=code)
        assert local.returncode == code and local.stdout == "check output\n"
    assert offline_check(tmp_path, FAILED_CANCEL, github=True, code=0).stdout == "check output\n"
