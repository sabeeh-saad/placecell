"""Replay RGB-D recordings against human object labels, without issuing navigation commands."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from placecell.depth import Box
from placecell.errors import ValidationError
from placecell.memory import memory_id
from placecell.objects import ObjectPolicy, ObjectTracker
from placecell.pipeline import Observation
from placecell.providers._http import Transport, UrllibTransport
from placecell.recordings import read_recording


@dataclass(frozen=True)
class ObjectLabel:
    id: str
    visibility: Literal["visible", "occluded", "absent"]
    box: Box | None = None
    position: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id.strip()
            or self.visibility not in {"visible", "occluded", "absent"}
        ):
            raise ValidationError("labels require a physical object ID and visibility")
        if (self.visibility == "visible") != (self.box is not None):
            raise ValidationError("only visible objects need a bounding box")
        if self.position is not None and (len(self.position) != 3 or not all(math.isfinite(v) for v in self.position)):
            raise ValidationError("ground-truth surface positions must be finite XYZ coordinates")


@dataclass(frozen=True)
class FrameLabels:
    observation_id: str
    objects: tuple[ObjectLabel, ...]
    complete: bool = True

    def __post_init__(self) -> None:
        if not self.observation_id or type(self.complete) is not bool:
            raise ValidationError("frame labels need an observation ID and completeness flag")
        if len(self.objects) > 64 or len({o.id for o in self.objects}) != len(self.objects):
            raise ValidationError("frame labels contain duplicate IDs or exceed 64 objects")


def load_object_labels(path: str | Path) -> dict[str, FrameLabels]:
    labels = {}
    with Path(path).open() as stream:
        for number, line in enumerate(iter(lambda: stream.readline(100_001), ""), 1):
            if number > 100_000 or len(line) > 100_000:
                raise ValidationError("label file exceeds evaluation limits")
            try:
                row = json.loads(line)
                objects = tuple(
                    ObjectLabel(
                        item["id"],
                        item["visibility"],
                        Box(*item["box"]) if "box" in item else None,
                        tuple(item["position"]) if "position" in item else None,
                    )
                    for item in row["objects"]
                )
                frame = FrameLabels(row["observation_id"], objects, row.get("complete", True))
            except (KeyError, TypeError, ValueError) as e:
                raise ValidationError(f"invalid object labels at line {number}: {e}") from e
            if frame.observation_id in labels:
                raise ValidationError("duplicate labeled observation")
            labels[frame.observation_id] = frame
    if not labels:
        raise ValidationError("at least one labeled frame is required")
    return labels


class UsageTransport:
    """Count requests and provider-reported tokens; never retain images, prompts, keys or responses."""

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        input_usd_per_million: float | None = None,
        output_usd_per_million: float | None = None,
    ) -> None:
        if any(
            v is not None and (not math.isfinite(v) or v < 0) for v in (input_usd_per_million, output_usd_per_million)
        ):
            raise ValidationError("token prices must be finite and nonnegative")
        self.transport = transport or UrllibTransport()
        self.input_rate, self.output_rate = input_usd_per_million, output_usd_per_million
        self.requests = self.failures = self.input_tokens = self.output_tokens = self.without_usage = 0
        self.elapsed_ms = 0.0

    def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, str], Any]:
        self.requests += 1
        started = time.perf_counter()
        try:
            status, response_headers, body = self.transport.post_json(url, headers, payload, timeout_s)
        except Exception:
            self.failures += 1
            self.without_usage += 1
            raise
        finally:
            self.elapsed_ms += (time.perf_counter() - started) * 1000
        self.failures += status >= 400
        usage = body.get("usageMetadata", {}) if isinstance(body, dict) else {}
        incoming = usage.get("promptTokenCount") if isinstance(usage, dict) else None
        outgoing = usage.get("candidatesTokenCount", 0) if isinstance(usage, dict) else None
        thoughts = usage.get("thoughtsTokenCount", 0) if isinstance(usage, dict) else None
        if all(type(v) is int and v >= 0 for v in (incoming, outgoing, thoughts)):
            assert isinstance(incoming, int) and isinstance(outgoing, int) and isinstance(thoughts, int)
            self.input_tokens += incoming
            self.output_tokens += outgoing + thoughts
        else:
            self.without_usage += 1
        return status, response_headers, body

    def report(self) -> dict[str, Any]:
        cost = None
        if self.without_usage == 0 and self.input_rate is not None and self.output_rate is not None:
            cost = (self.input_tokens * self.input_rate + self.output_tokens * self.output_rate) / 1_000_000
        return {
            "requests": self.requests,
            "failed_requests": self.failures,
            "elapsed_ms": self.elapsed_ms,
            "reported_input_tokens": self.input_tokens,
            "reported_output_tokens_including_thoughts": self.output_tokens,
            "requests_without_token_usage": self.without_usage,
            "estimated_cost_usd": cost,
            "input_usd_per_million": self.input_rate,
            "output_usd_per_million": self.output_rate,
        }


def evaluate_objects(
    tracker: ObjectTracker,
    observations: Iterable[Observation],
    labels: Mapping[str, FrameLabels],
    *,
    min_iou: float = 0.5,
    max_frames: int = 10_000,
) -> dict[str, Any]:
    """Labels never enter detection, embedding or association. Unlabeled frames still update memory.

    IoU matching is greedy, one-to-one, in descending overlap order. Physical IDs must
    be globally unique across the recording. Report ambiguous hypotheses as such, not successes.
    """
    if not labels or not 0 < min_iou <= 1 or not 1 <= max_frames <= 100_000:
        raise ValidationError("evaluation needs labels, an IoU threshold and bounded frame limit")
    if tracker.store.count() or tracker.store.objects.count():
        raise ValidationError("object evaluation requires an empty, separate store")
    counts = dict.fromkeys(
        (
            "frames",
            "scans",
            "labeled_scans",
            "visible_labels",
            "matched_detections",
            "unmatched_detections_in_complete_frames",
            "identity_switches",
            "false_merges",
            "ambiguous_detections",
            "false_missing",
            "presence_checks",
            "absence_checks",
            "confirmed_absence",
            "unknown_identity_checks",
        ),
        0,
    )
    last_prediction: dict[str, str] = {}
    prediction_owner: dict[str, str] = {}
    truth_predictions: dict[str, set[str]] = {}
    seen_labels: set[str] = set()
    position_errors: list[float] = []
    latencies: list[float] = []
    failure_examples: list[dict[str, Any]] = []
    last_time: float | None = None
    for observation in observations:
        counts["frames"] += 1
        if counts["frames"] > max_frames:
            raise ValidationError("recording exceeds max_frames; use a bounded evaluation session")
        if last_time is not None and observation.timestamp <= last_time:
            raise ValidationError("evaluation frames must be in increasing timestamp order")
        last_time = observation.timestamp
        identity = memory_id(observation.robot_id, observation.camera_id, observation.timestamp)
        frame = labels.get(identity)
        if frame is not None:
            seen_labels.add(identity)
        started = time.perf_counter()
        prepared = tracker.prepare(observation)
        tracker.commit(prepared)
        if not prepared.scan_key:
            continue
        counts["scans"] += 1
        latencies.append((time.perf_counter() - started) * 1000)
        if frame is None:
            continue
        counts["labeled_scans"] += 1
        visible = [label for label in frame.objects if label.visibility == "visible"]
        predicted = [(record, view) for record, view, _ in prepared.updates if view is not None]
        counts["visible_labels"] += len(visible)
        counts["ambiguous_detections"] += sum(record.status == "ambiguous" for record, _ in predicted)
        edges = []
        for i, label in enumerate(visible):
            assert label.box is not None
            for j, (_, view) in enumerate(predicted):
                assert view is not None
                overlap = label.box.overlap(view.box)
                if overlap >= min_iou:
                    edges.append((overlap, i, j))
        matched_truth: set[int] = set()
        matched_prediction: set[int] = set()
        for _, i, j in sorted(edges, key=lambda edge: (-edge[0], edge[1], edge[2])):
            if i in matched_truth or j in matched_prediction:
                continue
            matched_truth.add(i)
            matched_prediction.add(j)
            counts["matched_detections"] += 1
            label, (record, _) = visible[i], predicted[j]
            switched = label.id in last_prediction and last_prediction[label.id] != record.id
            merged = record.id in prediction_owner and prediction_owner[record.id] != label.id
            counts["identity_switches"] += switched
            counts["false_merges"] += merged
            last_prediction[label.id], prediction_owner[record.id] = record.id, label.id
            truth_predictions.setdefault(label.id, set()).add(record.id)
            if (switched or merged) and len(failure_examples) < 32:
                failure_examples.append(
                    {
                        "observation_id": identity,
                        "truth_id": label.id,
                        "prediction_id": record.id,
                        "identity_switch": switched,
                        "false_merge": merged,
                    }
                )
            if (
                label.position is not None
                and record.position is not None
                and record.position_timestamp == observation.timestamp
            ):
                position_errors.append(
                    math.dist(label.position, (record.position.x, record.position.y, record.position.z))
                )
        if frame.complete:
            counts["unmatched_detections_in_complete_frames"] += len(predicted) - len(matched_prediction)
        for label in frame.objects:
            prediction = last_prediction.get(label.id)
            known = tracker.store.objects.get(prediction) if prediction is not None else None
            if known is None or prediction_owner.get(known.id) != label.id:
                counts["unknown_identity_checks"] += 1
                continue
            if label.visibility == "absent":
                counts["absence_checks"] += 1
                counts["confirmed_absence"] += known.status == "missing"
            else:
                counts["presence_checks"] += 1
                counts["false_missing"] += known.status == "missing"
    if seen_labels != set(labels) or not counts["labeled_scans"]:
        raise ValidationError("labels must refer to recording frames and include at least one evaluated scan")

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "format_version": 1,
        "model": tracker.store.info.model,
        "policy": asdict(tracker.policy),
        "min_iou": min_iou,
        "counts": counts,
        "labeled_frames_skipped_by_scan_interval": len(seen_labels) - counts["labeled_scans"],
        "visible_detection_recall": counts["matched_detections"] / counts["visible_labels"]
        if counts["visible_labels"]
        else None,
        "track_fragmentations": sum(max(0, len(ids) - 1) for ids in truth_predictions.values()),
        "false_missing_rate": counts["false_missing"] / counts["presence_checks"]
        if counts["presence_checks"]
        else None,
        "absence_confirmation_rate": counts["confirmed_absence"] / counts["absence_checks"]
        if counts["absence_checks"]
        else None,
        "position_samples": len(position_errors),
        "mean_surface_position_error_m": mean(position_errors),
        "p95_surface_position_error_m": float(np.quantile(position_errors, 0.95)) if position_errors else None,
        "mean_scan_latency_ms": mean(latencies),
        "p95_scan_latency_ms": float(np.quantile(latencies, 0.95)) if latencies else None,
        "identity_failure_examples": failure_examples,
    }


def main(args: list[str] | None = None) -> None:
    from placecell.providers import GeminiEmbedder, GeminiObjectDetector
    from placecell.store import CollectionInfo, InMemoryStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, help="Exported observations.jsonl")
    parser.add_argument("--labels", required=True, help="Human-labeled frames as JSONL")
    parser.add_argument("--output", required=True, help="New JSON report path")
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--embedding-model", default="gemini-embedding-2")
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument("--scan-interval", type=float, default=15)
    parser.add_argument("--embedding-input-usd-per-million", type=float)
    parser.add_argument("--vision-input-usd-per-million", type=float)
    parser.add_argument("--vision-output-usd-per-million", type=float)
    options = parser.parse_args(args)
    output = Path(options.output)
    if output.exists():
        raise ValidationError("evaluation report already exists")
    labels = load_object_labels(options.labels)
    embedding_usage = UsageTransport(
        input_usd_per_million=options.embedding_input_usd_per_million, output_usd_per_million=0
    )
    vision_usage = UsageTransport(
        input_usd_per_million=options.vision_input_usd_per_million,
        output_usd_per_million=options.vision_output_usd_per_million,
    )
    key = os.environ.get(options.api_key_env, "")
    embedder = GeminiEmbedder(
        options.embedding_model, api_key=key, dimension=options.dimension, transport=embedding_usage
    )
    detector = GeminiObjectDetector(options.vision_model, api_key=key, transport=vision_usage)
    store = InMemoryStore(CollectionInfo("object-evaluation", embedder.model_name, embedder.dimension))
    try:
        tracker = ObjectTracker(store, embedder, detector, ObjectPolicy(min_interval_s=options.scan_interval))
        report = evaluate_objects(tracker, read_recording(options.recording), labels, max_frames=options.max_frames)
        report["vision_model"] = options.vision_model
        report["api_usage"] = {"embedding": embedding_usage.report(), "vision": vision_usage.report()}
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    finally:
        store.close()


if __name__ == "__main__":  # pragma: no cover
    main()
