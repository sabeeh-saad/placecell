"""How memories strengthen, fade and disappear.

Reinforcement merges a new observation into an existing memory of the same thing at the
same place. Decay is computed at read time from `last_seen`, so nothing has to be rewritten
as time passes. The curator removes what has decayed below usefulness, what is too old, and
what was superseded long enough ago, and hands the evidence to a remover so no orphan files
are left behind.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from placecell.corrections import CorrectionLog
from placecell.errors import ValidationError
from placecell.memory import Evidence, Memory
from placecell.store.base import EVERYTHING, Filter, VectorStore

EvidenceRemover = Callable[[Evidence], None]
"""Called once for every piece of evidence whose memory has been deleted."""


def remove_local_file(evidence: Evidence) -> None:
    """Evidence remover for media stored as local files. Missing files are not an error."""
    uri = evidence.uri.removeprefix("file://")
    Path(uri).unlink(missing_ok=True)


def remove_unreferenced(store: VectorStore, evidence: Iterable[Evidence], remover: EvidenceRemover) -> None:
    """Remove evidence only after the final reference, including superseded memories, is gone."""
    store.enqueue_cleanup(evidence)
    store.drain_cleanup(remover)


@dataclass(frozen=True, slots=True)
class ReinforcementPolicy:
    radius_m: float = 0.75
    """Two observations closer than this are at the same place."""
    min_similarity: float = 0.9
    """Two observations at least this similar describe the same thing."""
    gain: float = 0.5
    """Each repeat closes this fraction of the gap between the confidence and 1."""
    keep_newest_evidence: bool = True

    def __post_init__(self) -> None:
        if self.radius_m <= 0 or not (0 < self.gain <= 1) or not (-1 <= self.min_similarity <= 1):
            raise ValidationError("reinforcement policy out of range")


class Reinforcer:
    """Writes memories into a store, merging repeats instead of duplicating them."""

    def __init__(
        self,
        store: VectorStore,
        policy: ReinforcementPolicy | None = None,
        remover: EvidenceRemover | None = remove_local_file,
    ) -> None:
        self._store = store
        self._policy = policy or ReinforcementPolicy()
        self._remover = remover

    def reinforce_or_insert(self, memory: Memory) -> tuple[Memory, bool]:
        """Store the memory. Returns the memory as stored and whether it merged into an existing one.

        A memory whose id is already present is a replay of the same observation and is
        left untouched, which makes ingestion idempotent. A superseded memory of the same
        thing at the same place is revived: the object came back.
        """
        with self._store.transaction():
            return self._reinforce(memory)

    def _reinforce(self, memory: Memory) -> tuple[Memory, bool]:
        if memory.embedding is None:
            raise ValidationError("only embedded memories can be stored")
        existing = self.find_observation(memory.id)
        if existing is not None:
            self._discard_evidence([memory])
            return existing, True
        where = Filter(near=memory.pose, radius=self._policy.radius_m, include_superseded=True, role="episodic")
        hits = self._store.search(memory.embedding, 1, where)
        if hits and hits[0].score >= self._policy.min_similarity:
            merged = self._merge(hits[0].memory, memory)
            self._store.upsert([merged])
            self._discard_evidence([hits[0].memory, memory])
            return merged, True
        self._store.upsert([memory])
        return memory, False

    def find_observation(self, observation_id: str) -> Memory | None:
        """Find an observation even when it was folded into another memory."""
        existing = self._store.get(observation_id)
        if existing is not None:
            return existing
        matches = self._store.query(Filter(observation_id=observation_id, include_superseded=True), limit=1)
        return matches[0] if matches else None

    def _discard_evidence(self, memories: Iterable[Memory]) -> None:
        if self._remover is not None:
            remove_unreferenced(
                self._store,
                (m.evidence for m in memories if m.evidence is not None and m.evidence.managed),
                self._remover,
            )

    def _merge(self, existing: Memory, repeat: Memory) -> Memory:
        p = self._policy
        confidence = existing.confidence + (1.0 - existing.confidence) * p.gain
        newest = repeat.timestamp >= existing.last_seen
        return replace(
            existing,
            observations=existing.observations + 1,
            timestamp=min(existing.timestamp, repeat.timestamp),
            sightings=tuple(dict.fromkeys((*existing.sightings, *repeat.sightings)))[-64:],
            confidence=min(1.0, confidence),
            last_seen=max(existing.last_seen, repeat.timestamp),
            evidence=repeat.evidence if (p.keep_newest_evidence and newest and repeat.evidence) else existing.evidence,
            caption=existing.caption or repeat.caption,
            superseded=False,
            superseded_at=None,
            misses=0,
            last_miss=0.0,
        )


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    half_life_s: float = 7 * 24 * 3600.0
    min_confidence: float = 0.05
    """A memory decayed below this is expired, unless it has been reinforced often enough."""
    protected_observations: int = 5
    """Memories seen at least this often survive confidence expiry, but not max_age."""
    max_age_s: float | None = None
    """Hard cap on the age of the first observation. None keeps reinforced memories forever."""
    drop_superseded_after_s: float = 24 * 3600.0
    max_idle_s: float | None = 90 * 24 * 3600.0
    """Expire even reinforced memories after this long without a sighting. None disables it."""
    history_age_s: float = 90 * 24 * 3600.0
    """Retain detailed sightings for this long, plus the latest sighting of each memory."""
    wrong_verdicts_to_supersede: int = 3
    """A memory judged wrong this often, net of right verdicts, is superseded."""

    def __post_init__(self) -> None:
        if self.history_age_s <= 0 or (self.max_idle_s is not None and self.max_idle_s <= 0):
            raise ValidationError("retention durations must be positive")
        if self.wrong_verdicts_to_supersede < 1:
            raise ValidationError("retention policy out of range")
        if self.half_life_s <= 0 or not (0 <= self.min_confidence <= 1) or self.protected_observations < 1:
            raise ValidationError("retention policy out of range")
        if (self.max_age_s is not None and self.max_age_s <= 0) or self.drop_superseded_after_s < 0:
            raise ValidationError("retention policy out of range")


@dataclass(frozen=True, slots=True)
class CuratorReport:
    scanned: int = 0
    expired: int = 0
    aged_out: int = 0
    superseded_dropped: int = 0
    discredited: int = 0
    """Superseded in this pass because operators judged answers based on them wrong."""

    @property
    def removed(self) -> int:
        return self.expired + self.aged_out + self.superseded_dropped


class Curator:
    """Background maintenance over a collection. Safe to run repeatedly; each pass is independent.

    A pass scans the memories the scope filter keeps. On a fleet, scope a pass per robot or
    per time window instead of scanning everything at once.
    """

    def __init__(
        self,
        store: VectorStore,
        policy: RetentionPolicy | None = None,
        remover: EvidenceRemover | None = None,
        clock: Callable[[], float] = time.time,
        corrections: CorrectionLog | None = None,
    ) -> None:
        self._store = store
        self._policy = policy or RetentionPolicy()
        self._remover = remover
        self._clock = clock
        self._corrections = corrections

    def run(self, scope: Filter = EVERYTHING, now: float | None = None) -> CuratorReport:
        now = self._clock() if now is None else now
        p = self._policy
        scanned = expired_count = aged_count = dropped_count = discredited = 0
        for batch in self._store.iter_query(scope):
            with self._store.transaction():
                expired, aged, dropped, alive = [], [], [], []
                for candidate in batch:
                    m = self._store.get(candidate.id)
                    if m is None:
                        continue
                    scanned += 1
                    if m.superseded:
                        if m.superseded_at is not None and now - m.superseded_at >= p.drop_superseded_after_s:
                            dropped.append(m)
                    elif (p.max_age_s is not None and now - m.timestamp > p.max_age_s) or (
                        p.max_idle_s is not None and now - m.last_seen > p.max_idle_s
                    ):
                        aged.append(m)
                    elif (
                        m.observations < p.protected_observations
                        and m.effective_confidence(now, p.half_life_s) < p.min_confidence
                    ):
                        expired.append(m)
                    else:
                        alive.append(m)
                self._remove(expired + aged + dropped)
                discredited += self._discredit(alive, now)
                expired_count += len(expired)
                aged_count += len(aged)
                dropped_count += len(dropped)
        self._store.prune_history(now - p.history_age_s)
        if self._remover is not None:
            self._store.drain_cleanup(self._remover)
        return CuratorReport(scanned, expired_count, aged_count, dropped_count, discredited)

    def _discredit(self, alive: list[Memory], now: float) -> int:
        if self._corrections is None or not alive:
            return 0
        verdicts = self._corrections.verdicts(m.id for m in alive)
        doomed = [
            replace(m, superseded=True, superseded_at=now)
            for m in alive
            if m.id in verdicts
            and verdicts[m.id].wrong - verdicts[m.id].right >= self._policy.wrong_verdicts_to_supersede
        ]
        if doomed:
            self._store.upsert(doomed)
        return len(doomed)

    def supersede(self, memory_id: str, now: float | None = None) -> Memory | None:
        """Mark a memory as no longer true, keeping it for a grace period as a record."""
        m = self._store.get(memory_id)
        if m is None:
            return None
        now = self._clock() if now is None else now
        updated = replace(m, superseded=True, superseded_at=m.superseded_at if m.superseded else now)
        self._store.upsert([updated])
        return updated

    def forget(self, where: Filter) -> int:
        """Delete every memory the filter matches, evidence included. The explicit-deletion path."""
        removed = 0
        for doomed in self._store.iter_query(where):
            with self._store.transaction():
                self._remove(doomed)
                removed += len(doomed)
        return removed

    def _remove(self, memories: Iterable[Memory]) -> None:
        batch = list(memories)
        if not batch:
            return
        self._store.delete(m.id for m in batch)
        if self._remover is not None:
            remove_unreferenced(self._store, (m.evidence for m in batch if m.evidence is not None), self._remover)
