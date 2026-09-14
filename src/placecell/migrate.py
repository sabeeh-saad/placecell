"""Re-embedding: move a collection to a different embedding model without a robot.

Captions and keyframes were kept for exactly this. Every memory is re-embedded from its
caption and, when supported, its evidence. Independent vectors for both are
written to a target collection bound to the new model with all lifecycle fields intact.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from placecell.errors import ModelMismatchError, ValidationError
from placecell.memory import Evidence, EvidenceKind
from placecell.providers.base import EmbeddingProvider
from placecell.providers.embedding import embed_memories
from placecell.store.base import EVERYTHING, VectorStore


@dataclass(frozen=True, slots=True)
class MigrationReport:
    read: int = 0
    written: int = 0
    skipped: int = 0
    skipped_ids: tuple[str, ...] = field(default=())
    objects_written: int = 0


def reembed(
    source: VectorStore, target: VectorStore, embedder: EmbeddingProvider, batch_size: int = 64
) -> MigrationReport:
    """Copy `source` into `target`, re-embedded by `embedder`. Skips memories nothing can embed."""
    if target.info.model != embedder.model_name or target.info.dimension != embedder.dimension:
        raise ModelMismatchError(
            f"target collection {target.info.model!r}/{target.info.dimension} does not match "
            f"embedder {embedder.model_name!r}/{embedder.dimension}"
        )
    if batch_size < 1:
        raise ValidationError("batch_size must be positive")
    if source.objects.count() and not (embedder.capabilities.image and embedder.capabilities.text):
        raise ValidationError("a collection with objects requires an image and text embedding model")
    if source is target:
        raise ValidationError("re-embedding requires a separate target collection")
    read = written = 0
    skipped: list[str] = []
    for batch in source.iter_query(EVERYTHING, batch_size):
        read += len(batch)
        out, rejected = embed_memories(batch, embedder)
        skipped.extend(m.id for m in rejected)
        written += target.upsert(out)
        for memory in out:
            after = None
            while history := source.sightings(memory.id, limit=batch_size, after=after):
                target.append_sightings(memory.id, history)
                after = (history[-1].timestamp, history[-1].id)
    objects_written = 0
    for record in source.objects.iter_records():
        views = source.objects.views(record.id)
        with tempfile.TemporaryDirectory() as directory:
            crops = []
            for i, view in enumerate(views):
                path = Path(directory) / f"{i}.png"
                path.write_bytes(view.crop_png)
                crops.append(replace(view.memory, evidence=Evidence(EvidenceKind.FRAME, str(path))))
            embedded, rejected = embed_memories(crops, embedder)
            if rejected:
                raise ValidationError("target model rejected an object crop")
        object_history = source.objects.history(record.id)
        with target.transaction():
            target.objects.delete(record.id)
            target.objects.save(record)
            scope = json.dumps((record.robot_id, record.camera_id, record.frame_id, record.map_id))
            last_scan = max(
                source.objects.scan_time(scope) or record.last_seen, target.objects.scan_time(scope) or record.last_seen
            )
            target.objects.record_scan(scope, last_scan)
            for view, memory in zip(views, embedded, strict=True):
                target.objects.save(
                    record,
                    replace(view, memory=replace(memory, evidence=view.memory.evidence)),
                    max_views=max(1, len(views)),
                )
            for event in object_history:
                # Event positions belong to that historical revision, not necessarily the current one.
                target.objects.save(
                    replace(record, position=event.position),
                    event=event.kind,
                    event_time=event.timestamp,
                    max_events=max(1, len(object_history)),
                )
            target.objects.save(record)
        objects_written += 1
    return MigrationReport(read, written, len(skipped), tuple(skipped), objects_written)
