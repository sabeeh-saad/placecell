"""Budgeted real-model planning evaluation; development results never qualify held-out quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import time
from pathlib import Path
from typing import Any

from placecell.errors import ValidationError
from placecell.evaluation_budget import EvaluationBudget, EvaluationStoppedError
from placecell.mission_evaluation import MissionDataset, _selected, load_dataset, score_trials
from placecell.missions import MissionPlanner, PlanReviewAgent
from placecell.providers._http import RetryPolicy
from placecell.providers.chat import OpenAICompatibleChat

MODEL = "google/gemini-2.5-flash"
BASE_URL = "https://openrouter.ai/api/v1"


def preflight(dataset: MissionDataset, split: str) -> dict[str, Any]:
    cases = _selected(dataset, split)
    if split == "held_out" and dataset.label_status != "human_reviewed":
        raise ValidationError("held-out live runs require independently human-reviewed labels")
    return {
        "dataset_id": dataset.id,
        "dataset_sha256": dataset.sha256,
        "label_status": dataset.label_status,
        "split": split,
        "cases": len(cases),
        "groups": sorted({case.group for case in cases}),
        "claim": "Development diagnostic; no held-out accuracy claim"
        if split == "development"
        else "Label status is a supplied attestation; independence still requires review",
        "unassessed": ["grounded execution", "vision identity", "retrieval channel accuracy", "release qualification"],
    }


def save(path: Path, value: Any) -> None:
    """Replace checkpoints atomically inside a newly reserved output directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_key(path: Path) -> str:
    with path.open() as stream:
        mode = os.fstat(stream.fileno()).st_mode
        if not stat.S_ISREG(mode) or mode & 0o077:
            raise ValidationError("credential must be a private regular file (mode 0600)")
        value = stream.read(512).strip()
    if not value.startswith("sk-or-") or any(c.isspace() for c in value) or len(value) >= 512:
        raise ValidationError("invalid OpenRouter credential file")
    return value


class MeteredChat(OpenAICompatibleChat):
    def __init__(self, stage: str, budget: EvaluationBudget, key: str, model: str) -> None:
        super().__init__(
            model, BASE_URL, key, max_tokens=2048, timeout_s=30, retry=RetryPolicy(attempts=1), transport=budget
        )
        self.stage, self.budget = stage, budget

    def complete(self, messages: Any, tools: Any) -> Any:
        self.budget.stage = self.stage
        return super().complete(messages, tools)


def run_planning(
    dataset: MissionDataset,
    planner: MissionPlanner,
    budget: EvaluationBudget,
    output: Path,
    *,
    split: str,
    repeats: int,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    admission = preflight(dataset, split)
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValidationError("repeats must be within 1..10")
    cases = _selected(dataset, split)
    reports = []
    for repetition in range(1, repeats + 1):
        document: dict[str, Any] = {
            "schema_version": 1,
            "dataset_sha256": dataset.sha256,
            "runner": "live_model",
            "run_id": f"{output.name}-{repetition:02}",
            "configuration": {**configuration, "repetition": repetition},
            "trials": [],
        }

        # A complete empty report establishes denominators before the first request.
        def checkpoint(document: dict[str, Any] = document, repetition: int = repetition) -> dict[str, Any]:
            report = score_trials(dataset, document, split=split)
            save(output / f"trials-{repetition:02}.json", document)
            save(output / f"report-{repetition:02}.json", report)
            return report

        report = checkpoint()
        for case in cases:
            if budget.stop_reason:
                break
            budget.trial_id = f"{repetition}:{case.id}"
            started, first_request = time.perf_counter(), len(budget.records)
            result: dict[str, Any] = {"status": "ok", "decision": None, "destinations": []}
            try:
                # No category, scenario, expected alias/target ID or split crosses this boundary.
                plan = planner.plan(case.instruction, context=case.context)
                result.update(decision=plan.decision, destinations=list(plan.destinations))
            except Exception as error:
                result.update(
                    status="timeout" if isinstance(error, TimeoutError) else "error", error_type=type(error).__name__
                )
            result["latency_ms"] = (time.perf_counter() - started) * 1000
            billed = budget.records[first_request:]
            cost = sum(r["cost_usd"] for r in billed) if all(r["cost_usd"] is not None for r in billed) else None
            document["trials"].append({"case_id": case.id, "plan": result, "execution": None, "cost_usd": cost})
            report = checkpoint()
        reports.append(report)
    totals = {key: sum(r["plan"][key] for r in reports) for key in ("eligible", "assessed", "passed", "failed")}
    summary = {
        **admission,
        "repetitions": repeats,
        "configuration": configuration,
        "plan": totals,
        "unique_cases": len(cases),
        "attempted_trials": sum(row["plan_status"] != "missing" for r in reports for row in r["cases"]),
        "unrun_trials": sum(r["missing_trials"] for r in reports),
        "repeat_warning": "Repeated cases are correlated; repetitions do not add independent scene diversity.",
        "budget": budget.summary(),
        "completed": all(r["missing_trials"] == 0 for r in reports),
        "all_plans_match_labels": all(r["plan"]["failed"] == 0 for r in reports),
    }
    save(output / "summary.json", summary)
    return summary


def camera_smoke(recording: Path, budget: EvaluationBudget, key: str, output: Path, model: str) -> dict[str, Any]:
    """Check two real vision adapter calls, with no fabricated accuracy labels."""
    from placecell.providers.captioning import OpenAICompatibleCaptioner, data_url
    from placecell.recordings import read_recording
    from placecell.verification import VisionVerifier

    report: dict[str, Any] = {
        "scope": "Development API/protocol smoke check only; no independent visual labels or accuracy score",
        "recording_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "passed": False,
        "query": "printer",
    }
    first = len(budget.records)
    budget.trial_id = "camera_smoke"
    try:
        if budget.stop_reason:
            raise EvaluationStoppedError(budget.stop_reason)
        observation = next(read_recording(recording))
        image = Path(observation.evidence.uri)
        report["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
        report["timestamp"] = observation.timestamp
        budget.stage = "camera_caption_smoke"
        captioner = OpenAICompatibleCaptioner(
            model, BASE_URL, key, timeout_s=30, retry=RetryPolicy(attempts=1), transport=budget
        )
        report["caption"] = captioner.caption([observation.evidence])[0]
        budget.stage = "camera_verification_smoke"
        verdict = VisionVerifier(model, BASE_URL, key, timeout_s=30, transport=budget).verify(
            "printer", data_url(str(image))
        )
        report["verdict"] = {"result": verdict.result, "reason": verdict.reason}
        report["passed"] = True  # Protocol validity only; no assumption that printer is present.
    except Exception as error:
        report["error_type"] = type(error).__name__
    report["requests"] = len(budget.records) - first
    report["budget"] = budget.summary()
    save(output / "camera-smoke.json", report)
    return report


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "held_out"), default="held_out")
    parser.add_argument("--output", type=Path, required=True, help="New output directory, never an existing run")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--max-usd", type=float)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--review-model", default=MODEL)
    parser.add_argument("--recording", type=Path, help="Optional development-only vision protocol smoke check")
    options = parser.parse_args(args)
    dataset = load_dataset(options.dataset)
    admission = preflight(dataset, options.split)
    if options.recording is not None and options.split != "development":
        raise ValidationError("unlabelled camera smoke is development-only")
    if options.mode == "run" and (options.key_file is None or options.max_usd is None or options.max_requests is None):
        raise ValidationError("live runs require a private credential and explicit cost/request limits")
    if not 1 <= options.repeats <= 10:
        raise ValidationError("repeats must be within 1..10")
    options.output.mkdir(parents=True, exist_ok=False)
    save(options.output / "preflight.json", admission)
    if options.mode == "preflight":
        return
    budget = EvaluationBudget(
        options.output / "requests.jsonl",
        max_requests=options.max_requests,
        max_usd=options.max_usd,
        max_seconds=options.max_seconds,
    )
    key = read_key(options.key_file)
    configuration = {
        "planner_model": options.model,
        "review_model": options.review_model,
        "model_version_note": "Hosted IDs are not immutable weight snapshots; inspect response metadata.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_unix_s": time.time(),
        "max_tokens": 2048,
        "request_timeout_s": 30,
        "http_attempts": 1,
        "source_sha256": {
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        },
    }
    planner = MissionPlanner(
        MeteredChat("planner", budget, key, options.model),
        PlanReviewAgent(MeteredChat("reviewer", budget, key, options.review_model)),
        max_destinations=8,
    )
    summary = run_planning(
        dataset,
        planner,
        budget,
        options.output,
        split=options.split,
        repeats=options.repeats,
        configuration=configuration,
    )
    if options.recording is not None:
        summary["camera_smoke"] = camera_smoke(options.recording, budget, key, options.output, options.model)
        summary["budget"] = budget.summary()
        save(options.output / "summary.json", summary)
    if (
        not summary["all_plans_match_labels"]
        or budget.stop_reason
        or not summary.get("camera_smoke", {"passed": True})["passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
