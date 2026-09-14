"""Conservative object association and retrieval over continuously refreshed RGB-D views."""

from __future__ import annotations

import io
import json
import math
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from placecell.errors import ModelMismatchError, PlacecellError, ValidationError
from placecell.memory import Evidence, EvidenceKind, Memory
from placecell.object_types import Detection, ObjectDetector, ObjectHit, ObjectRecord, ObjectView
from placecell.providers.base import EmbeddingProvider, QueryEmbeddingProvider, normalise_rows
from placecell.providers.embedding import embed_memories
from placecell.store.base import VectorStore

if TYPE_CHECKING:
    from placecell.pipeline import Observation


@dataclass(frozen=True)
class ObjectPolicy:
    max_objects: int = 1000
    max_detections: int = 16
    max_views: int = 4
    max_events: int = 32
    max_absence_checks: int = 4
    association_similarity: float = 0.85
    association_margin: float = 0.08
    nearby_m: float = 0.35
    moved_similarity: float = 0.95
    max_move_m: float = 3.0
    misses_to_missing: int = 3
    visit_interval_s: float = 600
    retention_s: float = 30 * 86400
    min_interval_s: float = 15

    def __post_init__(self) -> None:
        if (
            min(
                self.max_objects,
                self.max_detections,
                self.max_views,
                self.max_events,
                self.max_absence_checks,
                self.misses_to_missing,
            )
            < 1
        ):
            raise ValidationError("object limits must be positive")
        if not all(
            math.isfinite(v) and v > 0
            for v in (
                self.association_margin,
                self.nearby_m,
                self.max_move_m,
                self.visit_interval_s,
                self.retention_s,
                self.min_interval_s,
            )
        ):
            raise ValidationError("object thresholds must be finite and positive")
        if not 0 < self.association_similarity <= self.moved_similarity <= 1 or self.association_margin > 1:
            raise ValidationError("invalid object similarity thresholds")


@dataclass(frozen=True)
class PreparedObjects:
    generation: int
    updates: tuple[tuple[ObjectRecord, ObjectView | None, str], ...]
    timestamp: float
    scan_key: str = ""


class ObjectTracker:
    """Provider work precedes a single atomic commit with its parent scene observation.

    Similar labels alone never join identities. Movement additionally requires a
    unique appearance match and a visibly empty old location in the same RGB-D frame.
    Ambiguous associations are recorded as separate, non-navigable hypotheses.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        detector: ObjectDetector,
        policy: ObjectPolicy | None = None,
    ) -> None:
        if embedder.model_name != store.info.model or embedder.dimension != store.info.dimension:
            raise ModelMismatchError("object embedder must match the collection")
        if not embedder.capabilities.image or not embedder.capabilities.text:
            raise ValidationError("object memory requires an image and text embedding provider")
        self.store, self.embedder, self.detector = store, embedder, detector
        self.policy = policy or ObjectPolicy()

    def prepare(self, observation: Observation) -> PreparedObjects:
        p, journal = self.policy, self.store.objects
        scan_key = json.dumps(
            (observation.robot_id, observation.camera_id, observation.pose.frame_id, observation.pose.map_id)
        )
        with self.store.transaction():
            generation = journal.generation
            last_scan = journal.scan_time(scan_key)
            if last_scan is not None and observation.timestamp - last_scan < p.min_interval_s:
                return PreparedObjects(generation, (), observation.timestamp)
            records = journal.records(
                robot_id=observation.robot_id,
                camera_id=observation.camera_id,
                frame_id=observation.pose.frame_id,
                map_id=observation.pose.map_id,
            )
            old_views = {r.id: journal.views(r.id, include_crops=False) for r in records}
            count = journal.count()
        by_id = {r.id: r for r in records}
        fresh = self._views(observation)
        positions = [
            observation.depth.locate(v.box) if observation.depth and observation.localization_checked else None
            for v in fresh
        ]
        similarities: dict[tuple[int, str], float] = {}
        for i, view in enumerate(fresh):
            assert view.memory.embedding is not None
            for record in records:
                if record.label != view.memory.caption.split(":", 1)[0] or observation.timestamp <= record.last_seen:
                    continue
                vectors = [v.memory.embedding for v in old_views[record.id] if v.memory.embedding is not None]
                if vectors:
                    similarities[i, record.id] = max(float(view.memory.embedding @ vector) for vector in vectors)

        # Nearby assignments need unambiguous best matches on BOTH sides of the association.
        edges: dict[tuple[int, str], float] = {}
        for (i, identity), score in similarities.items():
            record = by_id[identity]
            location, previous = positions[i], record.position
            if location is not None and previous is not None:
                nearby = location.distance(previous) <= max(p.nearby_m, location.uncertainty_m + previous.uncertainty_m)
            else:
                # RGB-only observations may update the same image region from almost the same viewpoint.
                nearby = any(
                    observation.pose.distance_to(v.memory.pose) <= 0.1
                    and observation.pose.heading_difference(v.memory.pose) <= 0.1
                    and fresh[i].box.overlap(v.box) >= 0.6
                    for v in old_views[identity]
                )
            if nearby:
                edges[i, identity] = score
        assignments = self._unique(edges)
        assigned_ids = set(assignments.values())
        absent: set[str] = set()
        # A detector omission is never sufficient. Geometry AND a visual check must agree.
        if observation.depth and observation.localization_checked:
            checks = 0
            for record in sorted(records, key=lambda r: (r.last_miss, r.id)):
                if (
                    record.id in assigned_ids
                    or record.position is None
                    or not old_views[record.id]
                    or observation.timestamp <= record.last_seen
                    or (record.misses and observation.timestamp - record.last_miss < p.visit_interval_s)
                ):
                    continue
                region = observation.depth.clear_region(record.position)
                if region is None:
                    continue
                if checks >= p.max_absence_checks:
                    break
                checks += 1
                references = journal.views(record.id, limit=1)
                if references and self.detector.absent(references[0].crop_png, observation.evidence, region):
                    absent.add(record.id)

        moves = {}
        for (i, identity), score in similarities.items():
            record = by_id[identity]
            current = positions[i]
            if (
                i not in assignments
                and identity in absent
                and record.position is not None
                and current is not None
                and score >= p.moved_similarity
                and current.distance(record.position) <= p.max_move_m
            ):
                # Another similar instance anywhere in this scope makes a move unverifiable.
                alternatives = [
                    v for (j, k), v in similarities.items() if (j == i or k == identity) and (j, k) != (i, identity)
                ]
                if not alternatives or score - max(alternatives) >= p.association_margin:
                    moves[i, identity] = score
        moved = self._unique(moves)
        assignments.update(moved)
        assigned_ids = set(assignments.values())
        updates: list[tuple[ObjectRecord, ObjectView | None, str]] = []
        for i, view in enumerate(fresh):
            label = view.memory.caption.split(":", 1)[0]
            assigned_identity = assignments.get(i)
            if assigned_identity is None:
                if count >= p.max_objects:
                    raise ValidationError("object capacity reached; prune old objects or raise max_objects")
                identity = view.object_id
                # A plausible unassigned match remains uncertain instead of silently merging identities.
                ambiguous = any(score >= p.association_similarity for (j, _), score in edges.items() if j == i)
                record = ObjectRecord(
                    identity,
                    observation.robot_id,
                    observation.camera_id,
                    observation.pose.frame_id,
                    observation.pose.map_id,
                    label,
                    observation.timestamp,
                    observation.timestamp,
                    positions[i],
                    "ambiguous" if ambiguous else "present",
                )
                event = "ambiguous" if ambiguous else "created"
                count += 1
            else:
                identity = assigned_identity
                before = by_id[identity]
                event = "moved" if i in moved else ("returned" if before.status == "missing" else "")
                record = replace(
                    before,
                    last_seen=observation.timestamp,
                    position=positions[i] or before.position,
                    misses=0,
                    last_miss=0,
                    revision=before.revision + 1,
                    status="ambiguous" if before.status == "ambiguous" else "present",
                )
            updates.append((record, replace(view, object_id=identity), event))
        for record in records:
            if record.id not in absent or record.id in assigned_ids or record.status == "missing":
                continue
            misses = record.misses + 1
            status = "missing" if misses >= p.misses_to_missing else record.status
            updates.append(
                (
                    replace(
                        record,
                        misses=misses,
                        last_miss=observation.timestamp,
                        status=status,
                        revision=record.revision + 1,
                    ),
                    None,
                    "missing" if status == "missing" else "absence_observed",
                )
            )
        return PreparedObjects(generation, tuple(updates), observation.timestamp, scan_key)

    def _unique(self, edges: dict[tuple[int, str], float]) -> dict[int, str]:
        result = {}
        for (i, identity), score in edges.items():
            if score < self.policy.association_similarity:
                continue
            alternatives = [v for (j, k), v in edges.items() if (j == i or k == identity) and (j, k) != (i, identity)]
            if not alternatives or score - max(alternatives) >= self.policy.association_margin:
                result[i] = identity
        return result

    def _views(self, observation: Observation) -> list[ObjectView]:
        try:
            from PIL import Image
        except ImportError as e:  # pragma: no cover
            raise PlacecellError("object crops require pip install 'placecell[objects]'") from e
        detections = self.detector.detect(observation.evidence)
        if len(detections) > self.policy.max_detections:
            raise ValidationError("detector exceeded max_detections")
        kept: list[Detection] = []
        for detection in detections:
            if all(detection.box.overlap(d.box) < 0.7 for d in kept):
                kept.append(detection)
        views = []
        memories = []
        crops = []
        with (
            Image.open(observation.evidence.uri.removeprefix("file://")) as source,
            tempfile.TemporaryDirectory() as tmp,
        ):
            if source.width * source.height > 16_000_000:
                raise ValidationError("object images must not exceed 16 megapixels")
            source.load()
            if (
                observation.depth is not None
                and observation.depth.image_width
                and (source.width != observation.depth.image_width or source.height != observation.depth.image_height)
            ):
                raise ValidationError("RGB image dimensions do not match aligned depth")
            for i, detection in enumerate(kept):
                box = detection.box
                bounds = (
                    int(box.left * source.width),
                    int(box.top * source.height),
                    math.ceil(box.right * source.width),
                    math.ceil(box.bottom * source.height),
                )
                crop = source.crop(bounds).convert("RGB")
                crop.thumbnail((512, 512))
                buffer = io.BytesIO()
                crop.save(buffer, format="PNG")
                data = buffer.getvalue()
                if len(data) > 1_000_000:
                    raise ValidationError("object crop exceeds size limit")
                path = Path(tmp) / f"{i}.png"
                path.write_bytes(data)
                memory = Memory.create(
                    observation.robot_id,
                    observation.camera_id,
                    observation.timestamp,
                    observation.pose,
                    Evidence(EvidenceKind.FRAME, str(path)),
                    f"{detection.label.casefold().strip()}: {detection.description}",
                )
                identity = str(uuid.uuid5(uuid.NAMESPACE_URL, f"placecell:object:{memory.id}:{i}"))
                memories.append(replace(memory, id=identity))
                crops.append((box, data))
            embedded, rejected = embed_memories(memories, self.embedder)
            if rejected:
                raise ValidationError("object embedder rejected a crop")
            for memory, (box, data) in zip(embedded, crops, strict=True):
                if memory.embedding is None or not np.any(memory.embedding):
                    raise ValidationError("object crop produced an empty embedding")
                view_memory = replace(
                    memory, evidence=observation.evidence, localization_checked=observation.localization_checked
                )
                views.append(ObjectView(memory.id, view_memory, box, data))
        return views

    def commit(self, prepared: PreparedObjects) -> None:
        with self.store.transaction():
            if self.store.objects.generation != prepared.generation:
                raise ValidationError("object memory changed during detection; retry this observation")
            for record, view, event in prepared.updates:
                self.store.objects.save(
                    record,
                    view,
                    event=event,
                    event_time=prepared.timestamp,
                    max_views=self.policy.max_views,
                    max_events=self.policy.max_events,
                )
            if prepared.scan_key:
                self.store.objects.record_scan(prepared.scan_key, prepared.timestamp)


class ObjectRecall:
    def __init__(
        self, store: VectorStore, embedder: EmbeddingProvider, *, clock: Callable[[], float] = time.time
    ) -> None:
        if store.info.model != embedder.model_name or store.info.dimension != embedder.dimension:
            raise ModelMismatchError("object query embedder must match collection")
        self.store, self.embedder, self.clock = store, embedder, clock

    def similar(
        self,
        text: str,
        *,
        robot_id: str,
        camera_id: str,
        frame_id: str,
        map_id: str,
        k: int = 10,
        max_age_s: float = 7 * 86400,
    ) -> list[ObjectHit]:
        if not text.strip() or k < 1 or not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValidationError("object search requires text, positive k and maximum age")
        encoded = (
            self.embedder.embed_queries([text])
            if isinstance(self.embedder, QueryEmbeddingProvider)
            else (self.embedder.embed_text([text]))
        )
        vector = normalise_rows(encoded, 1, self.embedder.dimension)[0]
        if not np.any(vector):
            return []
        scores: dict[str, float] = {}
        for identity, score in self.store.objects.scores(
            vector, robot_id=robot_id, camera_id=camera_id, frame_id=frame_id, map_id=map_id
        ):
            scores[identity] = max(scores.get(identity, -1), score)
        hits = []
        for identity in sorted(scores, key=lambda identity: (-scores[identity], identity)):
            record = self.store.objects.get(identity)
            if record is None or not 0 <= self.clock() - record.last_seen <= max_age_s:
                continue
            views = self.store.objects.views(identity, limit=1)
            # Historical views help identify an object; only its latest location is a navigation candidate.
            if views:
                hits.append(ObjectHit(record, views[0], scores[identity]))
            if len(hits) == k:
                break
        return hits
