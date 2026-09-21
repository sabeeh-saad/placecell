"""Score labelled mission trials; scripted runs exercise planning without model or robot calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from placecell.chat import ChatMessage, ChatReply, ToolCall
from placecell.errors import ProviderError, ValidationError
from placecell.missions import MissionPlanner, PlanReviewAgent

SPLITS = {"development", "held_out"}
DECISIONS = {"ready", "clarify", "reject"}
OUTCOMES = {"succeeded", "clarify", "reject", "not_found", "canceled", "failed", "error", "timeout"}


def _object(value: Any, required: set[str], optional: Set[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValidationError(f"expected fields {sorted(required)} with optional {sorted(optional)}")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValidationError("expected nonempty text of at most 2000 characters")
    return value


def _strings(value: Any, *, empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 100 or (not value and not empty):
        raise ValidationError("expected a bounded list of strings")
    return tuple(_text(item) for item in value)


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValidationError("measurements must be finite nonnegative numbers")
    return float(value)


def _choice(value: Any, allowed: set[str]) -> bool:
    return isinstance(value, str) and value in allowed


def _read(path: str | Path) -> tuple[Any, str]:
    with Path(path).open("rb") as stream:
        raw = stream.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValidationError("evaluation file exceeds 4 MB")
    try:
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    except (ValueError, UnicodeError) as error:
        raise ValidationError("invalid evaluation JSON") from error


@dataclass(frozen=True)
class MissionCase:
    id: str
    category: str
    group: str
    scenario: str
    instruction: str
    context: tuple[dict[str, Any], ...]
    decisions: tuple[str, ...]
    destinations: tuple[tuple[str, ...], ...]
    outcomes: tuple[str, ...]
    target_ids: tuple[str, ...]
    visual_required: bool


@dataclass(frozen=True)
class MissionDataset:
    id: str
    sha256: str
    label_status: str
    groups: dict[str, dict[str, Any]]
    cases: tuple[MissionCase, ...]


def load_dataset(path: str | Path) -> MissionDataset:
    data, digest = _read(path)
    data = _object(data, {"schema_version", "id", "label_status", "annotation_notes", "groups", "cases"})
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValidationError("unsupported mission dataset schema")
    if not _choice(data["label_status"], {"draft", "human_reviewed"}):
        raise ValidationError("labels must be draft or human_reviewed")
    _text(data["annotation_notes"])
    if not isinstance(data["groups"], list) or not data["groups"]:
        raise ValidationError("dataset needs groups")
    groups: dict[str, dict[str, Any]] = {}
    layouts: dict[str, str] = {}
    recordings: dict[str, str] = {}
    for group in data["groups"]:
        group = _object(group, {"id", "split", "kind", "layout_id", "recording_sha256"})
        identity, layout = _text(group["id"]), _text(group["layout_id"])
        if (
            identity in groups
            or not _choice(group["split"], SPLITS)
            or not _choice(group["kind"], {"synthetic", "recording"})
        ):
            raise ValidationError("invalid or duplicate evaluation group")
        split = group["split"]
        if layout in layouts and layouts[layout] != split:
            raise ValidationError("a layout cannot cross development and held-out splits")
        layouts[layout] = split
        for recording in _strings(group["recording_sha256"], empty=True):
            if len(recording) != 64 or any(c not in "0123456789abcdef" for c in recording):
                raise ValidationError("recordings need lowercase SHA-256 digests")
            if recording in recordings:
                raise ValidationError("a recording may belong to only one group")
            recordings[recording] = identity
        if group["kind"] == "recording" and not group["recording_sha256"]:
            raise ValidationError("recording groups need at least one recording digest")
        groups[identity] = group
    if not isinstance(data["cases"], list) or not 1 <= len(data["cases"]) <= 10000:
        raise ValidationError("dataset needs 1..10000 cases")
    cases: list[MissionCase] = []
    identities: set[str] = set()
    inputs: dict[str, str] = {}
    for row in data["cases"]:
        row = _object(row, {"id", "category", "group", "scenario", "instruction", "context", "expected"})
        identity, group_id = _text(row["id"]), _text(row["group"])
        if identity in identities or group_id not in groups:
            raise ValidationError("duplicate case ID or unknown group")
        identities.add(identity)
        context = row["context"]
        if (
            not isinstance(context, list)
            or len(context) > 20
            or any(not isinstance(event, dict) for event in context)
            or len(json.dumps(context)) > 16000
        ):
            raise ValidationError("context must fit the 20-event / 16000-character reference window")
        instruction = _text(row["instruction"])
        signature = json.dumps([" ".join(instruction.casefold().split()), context], sort_keys=True)
        split = groups[group_id]["split"]
        if signature in inputs and inputs[signature] != split:
            raise ValidationError("identical instruction/context cannot cross splits")
        inputs[signature] = split
        expected = _object(row["expected"], {"decisions", "destinations", "outcomes", "target_ids", "visual_required"})
        decisions = _strings(expected["decisions"])
        if not set(decisions) <= DECISIONS or ("ready" in decisions and len(decisions) != 1):
            raise ValidationError("ready cannot be interchangeable with abstention")
        if not isinstance(expected["destinations"], list) or len(expected["destinations"]) > 20:
            raise ValidationError("expected destinations must be an ordered list of alias lists")
        destinations = tuple(_strings(aliases) for aliases in expected["destinations"])
        if bool(destinations) != (decisions == ("ready",)):
            raise ValidationError("only ready labels have destinations")
        outcomes = _strings(expected["outcomes"])
        targets = _strings(expected["target_ids"], empty=True)
        if not set(outcomes) <= OUTCOMES or type(expected["visual_required"]) is not bool:
            raise ValidationError("invalid mission outcome labels")
        if "succeeded" in outcomes and (not targets or len(targets) != len(destinations)):
            raise ValidationError("successful missions need one target ID per requested visit")
        cases.append(
            MissionCase(
                identity,
                _text(row["category"]),
                group_id,
                _text(row["scenario"]),
                instruction,
                tuple(context),
                decisions,
                destinations,
                outcomes,
                targets,
                expected["visual_required"],
            )
        )
    return MissionDataset(_text(data["id"]), digest, data["label_status"], groups, tuple(cases))


def _selected(dataset: MissionDataset, split: str) -> tuple[MissionCase, ...]:
    if split not in SPLITS:
        raise ValidationError("unknown evaluation split")
    cases = tuple(case for case in dataset.cases if dataset.groups[case.group]["split"] == split)
    if not cases:
        raise ValidationError(f"no cases in {split}; do not substitute development data")
    return cases


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _plan_result(plan: Any) -> dict[str, Any]:
    plan = _object(plan, {"status", "decision", "destinations", "latency_ms"}, {"error_type"})
    if not _choice(plan["status"], {"ok", "error", "timeout"}):
        raise ValidationError("invalid plan result status")
    _number(plan["latency_ms"])
    destinations = _strings(plan["destinations"], empty=True)
    if plan["status"] == "ok":
        if not _choice(plan["decision"], DECISIONS) or bool(destinations) != (plan["decision"] == "ready"):
            raise ValidationError("invalid plan decision/destinations")
    elif plan["decision"] is not None or destinations:
        raise ValidationError("failed plans cannot contain a decision or destinations")
    return cast(dict[str, Any], plan)


def _execution_result(execution: Any) -> dict[str, Any] | None:
    if execution is None:
        return None
    execution = _object(execution, {"outcome", "dispatched_targets", "confirmed_targets", "latency_ms"})
    if not _choice(execution["outcome"], OUTCOMES):
        raise ValidationError("invalid execution outcome")
    _strings(execution["dispatched_targets"], empty=True)
    _strings(execution["confirmed_targets"], empty=True)
    _number(execution["latency_ms"])
    return cast(dict[str, Any], execution)


def _counts(values: Sequence[str]) -> dict[str, Any]:
    assessed = sum(value != "unassessed" for value in values)
    passed = values.count("passed")
    return {
        "eligible": len(values),
        "assessed": assessed,
        "passed": passed,
        "failed": values.count("failed"),
        "unassessed": len(values) - assessed,
        "pass_rate": passed / len(values) if values and assessed else None,
    }


def _latencies(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(values),
        **{f"p{p}_ms": ordered[math.ceil(p * len(values) / 100) - 1] if values else None for p in (50, 95, 99)},
    }


def score_trials(dataset: MissionDataset, document: Any, *, split: str = "development") -> dict[str, Any]:
    """Include every selected case; missing trials fail, unrun execution remains unassessed.

    Dispatched targets mean physical target identities, not proposed descriptions.
    The importer trusts the runner's evidence: this is a scorer, not a trace attestation.
    """
    document = _object(document, {"schema_version", "dataset_sha256", "runner", "run_id", "configuration", "trials"})
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValidationError("unsupported trial schema")
    if document["dataset_sha256"] != dataset.sha256:
        raise ValidationError("trial labels do not match the dataset hash")
    if not _choice(document["runner"], {"scripted", "live_model", "gazebo_live_model"}):
        raise ValidationError("unknown runner evidence type")
    _text(document["run_id"])
    if not isinstance(document["configuration"], dict) or not isinstance(document["trials"], list):
        raise ValidationError("trials need configuration provenance and result rows")
    cases = _selected(dataset, split)
    expected_ids = {case.id for case in cases}
    trials: dict[str, dict[str, Any]] = {}
    for trial in document["trials"]:
        trial = _object(trial, {"case_id", "plan", "execution", "cost_usd"})
        identity = _text(trial["case_id"])
        if identity not in expected_ids or identity in trials:
            raise ValidationError("duplicate, unknown or out-of-split trial ID")
        _plan_result(trial["plan"])
        _execution_result(trial["execution"])
        if trial["cost_usd"] is not None:
            _number(trial["cost_usd"])
        trials[identity] = trial
    rows: list[dict[str, Any]] = []
    timings: list[float] = []
    for case in cases:
        trial = trials.get(case.id)
        row: dict[str, Any] = {
            "case_id": case.id,
            "category": case.category,
            "group": case.group,
            "plan": "failed",
            "execution": "unassessed",
            "reason": "missing_trial",
            "wrong_destination_dispatch": False,
            "false_success": False,
            "unnecessary_abstention": False,
            "dispatch_without_ready_plan": False,
            "invalid_confirmation": False,
            "plan_status": "missing",
            "execution_outcome": None,
        }
        if trial is not None:
            plan = trial["plan"]
            row.update(
                plan_status=plan["status"],
                plan_decision=plan["decision"],
                plan_destinations=plan["destinations"],
                plan_latency_ms=plan["latency_ms"],
            )
            timings.append(plan["latency_ms"])
            matching = len(plan["destinations"]) == len(case.destinations) and all(
                _normalize(actual) in {_normalize(alias) for alias in aliases}
                for actual, aliases in zip(plan["destinations"], case.destinations, strict=True)
            )
            correct = plan["status"] == "ok" and plan["decision"] in case.decisions and matching
            row.update(
                plan="passed" if correct else "failed",
                reason="matched" if correct else (plan["status"] if plan["status"] != "ok" else "plan_mismatch"),
                unnecessary_abstention=plan["status"] == "ok"
                and plan["decision"] != "ready"
                and case.decisions == ("ready",),
            )
            execution = trial["execution"]
            if execution is not None:
                row.update(
                    execution_outcome=execution["outcome"],
                    dispatched_targets=execution["dispatched_targets"],
                    confirmed_targets=execution["confirmed_targets"],
                    execution_latency_ms=execution["latency_ms"],
                )
                visited, confirmed = tuple(execution["dispatched_targets"]), tuple(execution["confirmed_targets"])
                wrong = visited != case.target_ids[: len(visited)]
                success = execution["outcome"] == "succeeded"
                unauthorized = bool(visited) and (plan["status"] != "ok" or plan["decision"] != "ready")
                invalid_confirmation = confirmed != visited[: len(confirmed)]
                false_success = success and (
                    "succeeded" not in case.outcomes
                    or visited != case.target_ids
                    or (case.visual_required and confirmed != visited)
                )
                complete = (
                    (not success or correct)
                    and not wrong
                    and not false_success
                    and not unauthorized
                    and not invalid_confirmation
                    and execution["outcome"] in case.outcomes
                )
                row.update(
                    execution="passed" if complete else "failed",
                    wrong_destination_dispatch=wrong,
                    false_success=false_success,
                    dispatch_without_ready_plan=unauthorized,
                    invalid_confirmation=invalid_confirmation,
                )
            elif document["runner"] == "gazebo_live_model":
                row["execution"] = "failed"
        elif document["runner"] == "gazebo_live_model":
            row["execution"] = "failed"
        rows.append(row)
    categories = {
        name: {
            stage: _counts([row[stage] for row in rows if row["category"] == name]) for stage in ("plan", "execution")
        }
        for name in sorted({case.category for case in cases})
    }
    costs = [trial["cost_usd"] for trial in trials.values()]
    return {
        "schema_version": 1,
        "dataset_id": dataset.id,
        "dataset_sha256": dataset.sha256,
        "label_status": dataset.label_status,
        "split": split,
        "runner": document["runner"],
        "run_id": document["run_id"],
        "configuration": document["configuration"],
        "claim": "Scripted software check; not model accuracy"
        if document["runner"] == "scripted"
        else "Imported runner outcomes; inspect evidence and labels before making quality claims",
        "plan": _counts([row["plan"] for row in rows]),
        "execution": _counts([row["execution"] for row in rows]),
        "categories": categories,
        "plan_latency": _latencies(timings),
        "execution_latency": _latencies(
            [trial["execution"]["latency_ms"] for trial in trials.values() if trial["execution"] is not None]
        ),
        "mission_completion": _counts(
            [row["execution"] for row, case in zip(rows, cases, strict=True) if "succeeded" in case.outcomes]
        ),
        "appropriate_abstention": _counts(
            [row["execution"] for row, case in zip(rows, cases, strict=True) if "succeeded" not in case.outcomes]
        ),
        "missing_trials": len(cases) - len(trials),
        "timeouts": sum(trial["plan"]["status"] == "timeout" for trial in trials.values()),
        "errors": sum(trial["plan"]["status"] == "error" for trial in trials.values()),
        "execution_timeouts": sum(row["execution_outcome"] == "timeout" for row in rows),
        "execution_errors": sum(row["execution_outcome"] == "error" for row in rows),
        "unnecessary_abstentions": sum(row["unnecessary_abstention"] for row in rows),
        "wrong_destination_dispatch_trials": sum(row["wrong_destination_dispatch"] for row in rows),
        "false_success_trials": sum(row["false_success"] for row in rows),
        "dispatch_without_ready_plan_trials": sum(row["dispatch_without_ready_plan"] for row in rows),
        "invalid_confirmation_trials": sum(row["invalid_confirmation"] for row in rows),
        "known_cost_usd": sum(value for value in costs if value is not None),
        "total_cost_usd": sum(costs) if len(costs) == len(cases) and all(v is not None for v in costs) else None,
        "qualification": "unassessed: this report alone does not satisfy release gates",
        "cases": rows,
    }


class _ScriptedChat:
    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]) -> ChatReply:
        self.calls += 1
        if self.calls > 1 or self.reply is None:
            raise ProviderError("script has no reply for this call")
        if self.reply.get("error") == "timeout":
            raise TimeoutError("scripted timeout")
        if self.reply.get("error"):
            raise ProviderError("scripted provider error")
        return ChatReply(
            self.reply.get("content"),
            tuple(
                ToolCall(str(index), call["name"], call["arguments"])
                for index, call in enumerate(self.reply.get("tool_calls", []))
            ),
        )


def run_scripted(dataset: MissionDataset, fixtures: Any, *, run_id: str) -> dict[str, Any]:
    """Only instruction/context cross into the real planner; replies never derive from labels."""
    fixtures = _object(fixtures, {"schema_version", "cases"})
    if (
        type(fixtures["schema_version"]) is not int
        or fixtures["schema_version"] != 1
        or not isinstance(fixtures["cases"], dict)
    ):
        raise ValidationError("invalid scripted fixture schema")
    cases = _selected(dataset, "development")
    if set(fixtures["cases"]) != {case.id for case in cases}:
        raise ValidationError("scripted replies must cover exactly the development cases")
    rows = []
    for case in cases:
        fixture = _object(fixtures["cases"][case.id], {"planner", "reviewer"})
        planner = MissionPlanner(_ScriptedChat(fixture["planner"]), PlanReviewAgent(_ScriptedChat(fixture["reviewer"])))
        started = time.perf_counter()
        result: dict[str, Any] = {"status": "ok", "decision": None, "destinations": []}
        try:
            plan = planner.plan(case.instruction, context=case.context)
            result.update(decision=plan.decision, destinations=list(plan.destinations))
        except Exception as error:
            result.update(
                status="timeout" if isinstance(error, TimeoutError) else "error", error_type=type(error).__name__
            )
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        rows.append({"case_id": case.id, "plan": result, "execution": None, "cost_usd": 0.0})
    return {
        "schema_version": 1,
        "dataset_sha256": dataset.sha256,
        "runner": "scripted",
        "run_id": _text(run_id),
        "configuration": {
            "planner": "MissionPlanner",
            "reviewer": "PlanReviewAgent",
            "model": "scripted",
            "network_calls": 0,
            "robot_calls": 0,
            "python": platform.python_version(),
            "implementation_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("mission_evaluation.py", "missions.py")
            },
        },
        "trials": rows,
    }


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "scripted", "score"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, help="New report; existing files are never overwritten")
    parser.add_argument("--replies", help="Separate scripted model replies; never the expected labels")
    parser.add_argument("--trials", help="Runner outcomes to score")
    parser.add_argument("--save-trials", help="New file for the scripted planner's actual outputs")
    parser.add_argument("--run-id", default="offline-baseline")
    parser.add_argument("--split", choices=sorted(SPLITS), default="development")
    options = parser.parse_args(args)
    output = Path(options.output)
    destinations = [output] + ([Path(options.save_trials)] if options.save_trials else [])
    if len({path.resolve() for path in destinations}) != len(destinations) or any(
        path.exists() for path in destinations
    ):
        raise ValidationError("reports must be distinct new files")
    if options.mode == "scripted" and (not options.replies or options.split != "development"):
        raise ValidationError("scripted runs require --replies and use only the development split")
    if options.mode == "score" and not options.trials:
        raise ValidationError("score requires --trials")
    if options.save_trials and options.mode != "scripted":
        raise ValidationError("--save-trials is only valid for scripted runs")
    dataset = load_dataset(options.dataset)
    report: dict[str, Any]
    if options.mode == "validate":
        report = {
            "dataset_id": dataset.id,
            "dataset_sha256": dataset.sha256,
            "label_status": dataset.label_status,
            "cases": len(dataset.cases),
            "splits": {
                split: sum(dataset.groups[c.group]["split"] == split for c in dataset.cases) for split in sorted(SPLITS)
            },
        }
    else:
        if options.mode == "scripted":
            fixtures, fixtures_digest = _read(options.replies)
            trials = run_scripted(dataset, fixtures, run_id=options.run_id)
            trials["configuration"]["replies_sha256"] = fixtures_digest
        else:
            trials, _ = _read(options.trials)
        report = score_trials(dataset, trials, split=options.split)
        if options.save_trials:
            with Path(options.save_trials).open("x") as stream:
                json.dump(trials, stream, indent=2, allow_nan=False)
                stream.write("\n")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    sys.stdout.write(f"Saved {options.mode} report to {output}\n")
    if options.mode != "validate" and (report["plan"]["failed"] or report["execution"]["failed"]):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
