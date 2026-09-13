"""The three questions an agent can ask a memory: what looks like this, what happened when, what was where.

Every answer is ranked by similarity times decayed confidence, so a memory seen fifty
times last week outranks one seen once a month ago, and a superseded memory never appears.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from placecell.corrections import CorrectionLog, Verdicts
from placecell.errors import ModelMismatchError, ValidationError
from placecell.memory import Memory, Pose, SearchChannel
from placecell.providers.base import EmbeddingProvider, QueryEmbeddingProvider, normalise_rows
from placecell.store.base import Filter, VectorStore

RetrievalMode = Literal["combined", "image", "caption"]

WEEK_S = 7 * 24 * 3600.0


@dataclass(frozen=True, slots=True)
class RankedMemory:
    memory: Memory
    confidence: float
    """Effective confidence at query time, decay applied."""
    similarity: float | None = None
    """Cosine similarity to the query for similarity searches, None for time and place lookups."""
    observed_at: tuple[float, ...] = ()
    """Sighting times inside a requested time window, oldest first."""

    image_similarity: float | None = None
    caption_similarity: float | None = None

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
        if embedder.model_name != store.info.model or embedder.dimension != store.info.dimension:
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

    def confidence(self, memory: Memory) -> float:
        """Current decayed confidence, including operator verdicts, for a stored memory."""
        verdicts = self._corrections.verdicts([memory.id]) if self._corrections else {}
        return (
            memory.effective_confidence(self._clock(), self._half_life_s) * verdicts.get(memory.id, Verdicts()).weight
        )

    def similar(
        self, text: str, k: int = 10, where: Filter | None = None, *, mode: RetrievalMode = "combined"
    ) -> list[RankedMemory]:
        """Search each channel independently, then rank by the strongest cosine times confidence.

        Combined includes legacy primary vectors of unknown modality. Channel-only searches
        require known provenance. Missing channels never dilute a match from another channel.
        Scores are model-dependent similarities, not calibrated probabilities.
        """
        if not text.strip():
            raise ValidationError("query text must not be empty")
        if k < 1:
            raise ValidationError("k must be at least 1")
        if mode not in {"combined", "image", "caption"}:
            raise ValidationError("retrieval mode must be combined, image or caption")
        encoded = (
            self._embedder.embed_queries([text])
            if isinstance(self._embedder, QueryEmbeddingProvider)
            else self._embedder.embed_text([text])
        )
        vector = normalise_rows(encoded, 1, self._embedder.dimension)[0]
        if not np.any(vector):
            return []
        now = self._clock()
        channels: tuple[SearchChannel, ...] = ("primary", "caption") if mode == "combined" else (mode,)
        candidates: dict[str, Memory] = {}
        for channel in channels:
            for hit in self._store.search(vector, k * self._oversample, where, channel=channel):
                candidates[hit.memory.id] = hit.memory
        # Fresh candidates enter before decay reranking, even if old rows have higher cosine.
        for memory in self._store.query(where, limit=max(64, k * self._oversample), order="recent"):
            candidates[memory.id] = memory
        weights = self._corrections.verdicts(candidates) if self._corrections else {}
        ranked = []
        for memory in candidates.values():
            scores: dict[SearchChannel, float] = {}
            for channel in ("primary", "image", "caption"):
                other = memory.vector_for(channel)
                if other is not None and (norm := float(np.linalg.norm(other))):
                    scores[channel] = float(np.clip(vector @ other / norm, -1.0, 1.0))
            selected = [scores[c] for c in channels if c in scores]
            if not selected:
                continue
            ranked.append(
                RankedMemory(
                    memory,
                    memory.effective_confidence(now, self._half_life_s) * weights.get(memory.id, Verdicts()).weight,
                    max(selected),
                    image_similarity=scores.get("image"),
                    caption_similarity=scores.get("caption"),
                )
            )
        ranked.sort(key=lambda r: (-r.score, r.memory.id))
        groups: set[str] = set()
        result = []
        for item in ranked:
            group = item.memory.consolidated_into or item.memory.id
            if group not in groups:
                groups.add(group)
                result.append(item)
                if len(result) == k:
                    break
        return result

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
        return [
            RankedMemory(
                m,
                m.effective_confidence(now, self._half_life_s),
                observed_at=tuple(
                    dict.fromkeys(
                        s.timestamp
                        for s in self._store.sightings(m.id, limit=64, time_from=where.time_from, time_to=where.time_to)
                    )
                ),
            )
            for m in rows
        ]
