"""The three questions an agent can ask a memory: what looks like this, what happened when, what was where.

Every answer is ranked by similarity times decayed confidence, so a memory seen fifty
times last week outranks one seen once a month ago, and a superseded memory never appears.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from placecell.corrections import CorrectionLog, Verdicts
from placecell.errors import ModelMismatchError, ValidationError
from placecell.memory import Memory, Pose
from placecell.providers.base import EmbeddingProvider
from placecell.store.base import Filter, VectorStore

WEEK_S = 7 * 24 * 3600.0


@dataclass(frozen=True, slots=True)
class RankedMemory:
    memory: Memory
    confidence: float
    """Effective confidence at query time, decay applied."""
    similarity: float | None = None
    """Cosine similarity to the query for similarity searches, None for time and place lookups."""

    @property
    def score(self) -> float:
        return (1.0 if self.similarity is None else self.similarity) * self.confidence


class Recall:
    """Read-only retrieval tools over one collection."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        half_life_s: float = WEEK_S,
        clock: Callable[[], float] = time.time,
        oversample: int = 4,
        corrections: CorrectionLog | None = None,
    ) -> None:
        if embedder.model_name != store.info.model:
            raise ModelMismatchError(
                f"embedder {embedder.model_name!r} does not match collection model {store.info.model!r}"
            )
        if half_life_s <= 0 or oversample < 1:
            raise ValidationError("half_life_s must be positive and oversample at least 1")
        self._store = store
        self._embedder = embedder
        self._half_life_s = half_life_s
        self._clock = clock
        self._oversample = oversample
        self._corrections = corrections

    def similar(self, text: str, k: int = 10, where: Filter | None = None) -> list[RankedMemory]:
        """Memories whose content resembles the text, best first."""
        if not text.strip():
            raise ValidationError("query text must not be empty")
        if k < 1:
            raise ValidationError("k must be at least 1")
        vector = self._embedder.embed_text([text])[0]
        now = self._clock()
        hits = self._store.search(vector, k * self._oversample, where)
        weights = self._corrections.verdicts(h.memory.id for h in hits) if self._corrections else {}
        ranked = [
            RankedMemory(
                h.memory,
                h.memory.effective_confidence(now, self._half_life_s) * weights.get(h.memory.id, Verdicts()).weight,
                h.score,
            )
            for h in hits
        ]
        ranked.sort(key=lambda r: -r.score)
        return ranked[:k]

    def between(
        self, time_from: float, time_to: float, where: Filter | None = None, limit: int | None = None
    ) -> list[RankedMemory]:
        """Memories observed in [time_from, time_to), oldest first."""
        scoped = replace(where or Filter(), time_from=time_from, time_to=time_to)
        return self._plain(scoped, limit)

    def near(
        self, pose: Pose, radius_m: float, where: Filter | None = None, limit: int | None = None
    ) -> list[RankedMemory]:
        """Memories observed within radius_m of the pose, oldest first."""
        scoped = replace(where or Filter(), near=pose, radius=radius_m)
        return self._plain(scoped, limit)

    def _plain(self, where: Filter, limit: int | None) -> list[RankedMemory]:
        now = self._clock()
        rows = self._store.query(where, limit)
        return [RankedMemory(m, m.effective_confidence(now, self._half_life_s)) for m in rows]
