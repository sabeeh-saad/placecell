"""Re-embedding: move a collection to a different embedding model without a robot.

Captions and keyframes were kept for exactly this. Every memory is re-embedded from its
caption and, when supported, its evidence. Independent vectors for both are
written to a target collection bound to the new model with all lifecycle fields intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from placecell.errors import ModelMismatchError, ValidationError
from placecell.providers.base import EmbeddingProvider
from placecell.providers.embedding import embed_memories
from placecell.store.base import EVERYTHING, VectorStore


@dataclass(frozen=True, slots=True)
class MigrationReport:
    read: int = 0
    written: int = 0
    skipped: int = 0
    skipped_ids: tuple[str, ...] = field(default=())


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
    return MigrationReport(read, written, len(skipped), tuple(skipped))
