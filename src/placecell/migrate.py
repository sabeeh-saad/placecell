"""Re-embedding: move a collection to a different embedding model without a robot.

Captions and keyframes were kept for exactly this. Every memory is re-embedded from its
caption, or from its evidence if the new model takes images and there is no caption, and
written to a target collection bound to the new model with all lifecycle fields intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from placecell.errors import ModelMismatchError, ValidationError
from placecell.memory import Memory
from placecell.providers.base import EmbeddingProvider
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
    rows = source.query(EVERYTHING)
    written = 0
    skipped: list[str] = []
    caps = embedder.capabilities
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        by_text = [m for m in batch if m.caption and caps.text]
        by_media = [m for m in batch if m not in by_text and m.evidence is not None and caps.supports(m.evidence)]
        skipped.extend(m.id for m in batch if m not in by_text and m not in by_media)
        out: list[Memory] = []
        if by_text:
            vectors = embedder.embed_text([m.caption for m in by_text])
            out.extend(
                replace(m, embedding=v, model=embedder.model_name) for m, v in zip(by_text, vectors, strict=True)
            )
        if by_media:
            vectors = embedder.embed_media([m.evidence for m in by_media])  # type: ignore[misc]
            out.extend(
                replace(m, embedding=v, model=embedder.model_name) for m, v in zip(by_media, vectors, strict=True)
            )
        written += target.upsert(out)
    return MigrationReport(len(rows), written, len(skipped), tuple(skipped))
