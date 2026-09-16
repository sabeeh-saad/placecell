"""Read-only arrival checks against an object's pre-departure visual evidence."""

from __future__ import annotations

import base64
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from placecell.depth import ObjectPosition
from placecell.errors import ValidationError
from placecell.memory import Evidence, EvidenceKind, Memory
from placecell.object_types import ObjectRecord, ObjectView
from placecell.objects import ObjectTracker
from placecell.pipeline import Observation
from placecell.verification import SceneVerdict


class ObjectComparator(Protocol):
    def compare(self, references: tuple[bytes, ...], candidate: bytes) -> SceneVerdict: ...


@dataclass(frozen=True)
class ObjectReference:
    record: ObjectRecord
    views: tuple[ObjectView, ...] = field(repr=False)
    rivals: tuple[Memory, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class ObjectArrivalPolicy:
    min_similarity: float = 0.85
    moved_similarity: float = 0.95
    similarity_margin: float = 0.08
    nearby_m: float = 0.35
    max_move_m: float = 3.0
    max_uncertainty_m: float = 0.35
    max_position_age_s: float = 300
    max_observation_age_s: float = 5
    same_view_m: float = 0.1
    same_heading_rad: float = 0.1
    min_overlap: float = 0.6

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValidationError("arrival thresholds must be finite and positive")
        if (
            max(self.min_similarity, self.moved_similarity, self.similarity_margin, self.min_overlap) > 1
            or self.moved_similarity < self.min_similarity
            or self.max_move_m < self.nearby_m
        ):
            raise ValidationError("invalid object arrival thresholds")


@dataclass(frozen=True)
class ObjectArrivalVerdict:
    result: Literal["matched", "missing", "ambiguous", "unobserved", "unavailable"]
    reason: str
    position: ObjectPosition | None = None


class ObjectArrivalVerifier:
    """Appearance, geometry and a paired-image check must agree; verification never writes memory."""

    def __init__(
        self,
        tracker: ObjectTracker,
        comparator: ObjectComparator,
        policy: ObjectArrivalPolicy | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.tracker, self.comparator = tracker, comparator
        self.policy, self.clock = policy or ObjectArrivalPolicy(), clock

    def capture(self, identity: str) -> ObjectReference:
        journal = self.tracker.store.objects
        with self.tracker.store.transaction():
            record = journal.get(identity)
            if record is None or record.status != "present" or record.misses:
                raise ValidationError("object reference is unavailable")
            views = tuple(v for v in journal.views(identity, limit=4) if v.crop_png and v.memory.localization_checked)
            if not views:
                raise ValidationError("object reference needs localized saved crops")
            records = journal.records(
                robot_id=record.robot_id, camera_id=record.camera_id, frame_id=record.frame_id, map_id=record.map_id
            )
            if len(records) > 1000:
                raise ValidationError("arrival comparison exceeds the object scope limit")
            rivals = tuple(
                view.memory
                for other in records
                if other.id != identity and other.label == record.label
                for view in journal.views(other.id, include_crops=False, limit=4)
            )
            return ObjectReference(record, views, rivals)

    def available(self, reference: ObjectReference) -> bool:
        current = self.tracker.store.objects.get(reference.record.id)
        return bool(
            current is not None
            and current.status != "ambiguous"
            and (current.robot_id, current.camera_id, current.frame_id, current.map_id, current.label)
            == (
                reference.record.robot_id,
                reference.record.camera_id,
                reference.record.frame_id,
                reference.record.map_id,
                reference.record.label,
            )
        )

    @staticmethod
    def _similarity(view: ObjectView, memories: tuple[Memory, ...]) -> float:
        vector = view.memory.embedding
        if vector is None or not np.linalg.norm(vector):
            return -1.0
        return max(
            (
                float(vector @ m.embedding / (np.linalg.norm(vector) * np.linalg.norm(m.embedding)))
                for m in memories
                if m.embedding is not None and np.linalg.norm(m.embedding)
            ),
            default=-1.0,
        )

    def verify_image(
        self, reference: ObjectReference, observation: Observation, image_url: str, canceled: Callable[[], bool]
    ) -> ObjectArrivalVerdict:
        """Use snapshotted bytes so concurrent keyframe cleanup cannot change the arrival evidence."""
        prefix, separator, encoded = image_url.partition(",")
        if (
            prefix not in {"data:image/png;base64", "data:image/jpeg;base64"}
            or not separator
            or len(encoded) > 11_000_000
        ):
            raise ValidationError("arrival snapshot must be a bounded PNG or JPEG")
        raw = base64.b64decode(encoded, validate=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("frame.png" if "png" in prefix else "frame.jpg")
            path.write_bytes(raw)
            copied = replace(observation, evidence=Evidence(EvidenceKind.FRAME, str(path)))
            return self.verify(reference, copied, canceled)

    def verify(
        self, reference: ObjectReference, observation: Observation, canceled: Callable[[], bool] = lambda: False
    ) -> ObjectArrivalVerdict:
        p, record = self.policy, reference.record

        def check() -> None:
            if canceled() or self.clock() < observation.timestamp or not self.available(reference):
                raise ValidationError("object arrival check canceled or reference removed")

        if (
            not observation.localization_checked
            or not 0 <= self.clock() - observation.timestamp <= p.max_observation_age_s
            or (observation.robot_id, observation.camera_id, observation.pose.frame_id, observation.pose.map_id)
            != (record.robot_id, record.camera_id, record.frame_id, record.map_id)
            or observation.timestamp <= record.last_seen
        ):
            raise ValidationError("object arrival needs a fresh localized observation in the original scope")

        check()
        fresh = self.tracker.detect_views(observation)
        check()
        memories = tuple(view.memory for view in reference.views)
        scored = sorted(((self._similarity(view, memories), i) for i, view in enumerate(fresh)), reverse=True)
        position_known = (
            record.position is not None
            and record.position_timestamp is not None
            and 0 <= self.clock() - record.position_timestamp <= p.max_position_age_s
            and record.position.uncertainty_m <= p.max_uncertainty_m
        )

        def absent() -> bool:
            if not position_known or observation.depth is None or record.position is None:
                return False
            region = observation.depth.clear_region(record.position)
            if region is None:
                return False
            check()
            answer = self.tracker.detector.absent(reference.views[0].crop_png, observation.evidence, region)
            check()
            return answer

        if not scored or scored[0][0] < p.min_similarity:
            if absent():
                return ObjectArrivalVerdict("missing", "The previously occupied region is visible and empty.")
            return ObjectArrivalVerdict("unobserved", "The selected object was not observed clearly in this view.")
        score, index = scored[0]
        candidate = fresh[index]
        alternative = max(scored[1][0] if len(scored) > 1 else -1, self._similarity(candidate, reference.rivals))
        if score - alternative < p.similarity_margin:
            return ObjectArrivalVerdict(
                "ambiguous", "Similar objects cannot be distinguished. Please identify the target."
            )

        position = observation.depth.locate(candidate.box) if observation.depth else None
        moved = False
        if position_known and position is not None and record.position is not None:
            if position.uncertainty_m > p.max_uncertainty_m:
                return ObjectArrivalVerdict("unavailable", "The fresh object position is too uncertain.")
            distance = position.distance(record.position)
            nearby = max(p.nearby_m, position.uncertainty_m + record.position.uncertainty_m)
            if distance > nearby:
                if score < p.moved_similarity or distance > p.max_move_m or not absent():
                    return ObjectArrivalVerdict(
                        "ambiguous", "Appearance suggests a match, but movement is unconfirmed."
                    )
                moved = True
        elif not any(
            observation.pose.distance_to(view.memory.pose) <= p.same_view_m
            and observation.pose.heading_difference(view.memory.pose) <= p.same_heading_rad
            and candidate.box.overlap(view.box) >= p.min_overlap
            for view in reference.views
        ):
            return ObjectArrivalVerdict("unavailable", "Reliable depth or a matching recorded viewpoint is required.")

        check()
        verdict = self.comparator.compare(tuple(view.crop_png for view in reference.views), candidate.crop_png)
        check()
        if verdict.result not in {"matched", "not_matched", "uncertain"} or not verdict.reason.strip():
            raise ValidationError("invalid object comparison verdict")
        if verdict.result == "uncertain":
            return ObjectArrivalVerdict("ambiguous", verdict.reason)
        if verdict.result != "matched":
            return ObjectArrivalVerdict("unobserved", verdict.reason)
        reason = "Saved views and fresh appearance/geometry agree. "
        if moved:
            reason += "The previous location is confirmed empty. "
        return ObjectArrivalVerdict("matched", reason + verdict.reason, position)
