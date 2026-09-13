"""Contradiction: noticing that something remembered is no longer there.

After a new observation is embedded, the observer looks up memories taken from about the
same place looking in about the same direction. If the new view does not resemble such a
memory, that counts as a miss, but only once per visit. After enough misses on separate
visits the memory is superseded. One pass with a person standing in front of the shelf is
not evidence that the shelf is gone; three visits over three days are.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from placecell.errors import ValidationError
from placecell.memory import Memory, Vector
from placecell.store.base import Filter, VectorStore


@dataclass(frozen=True, slots=True)
class ContradictionPolicy:
    same_place_m: float = 0.75
    """Two observations closer than this were taken from the same place."""
    same_heading_rad: float = 0.5
    """And looking within this angle of each other, so they saw the same scene."""
    confirm_similarity: float = 0.6
    """A new view at least this similar to the memory confirms it; below is a miss."""
    misses_to_supersede: int = 3
    visit_gap_s: float = 600.0
    """Misses closer together than this are the same visit and count once."""

    def __post_init__(self) -> None:
        if self.same_place_m <= 0 or self.same_heading_rad <= 0 or self.visit_gap_s < 0:
            raise ValidationError("contradiction policy out of range")
        if self.misses_to_supersede < 1 or not (-1 <= self.confirm_similarity <= 1):
            raise ValidationError("contradiction policy out of range")


@dataclass(frozen=True, slots=True)
class ObserverReport:
    in_view: int = 0
    confirmed: int = 0
    missed: int = 0
    superseded: int = 0


class Observer:
    """Compares a fresh observation against what memory expects at that place and heading."""

    def __init__(
        self, store: VectorStore, policy: ContradictionPolicy | None = None, clock: Callable[[], float] = time.time
    ) -> None:
        self._store = store
        self._policy = policy or ContradictionPolicy()
        self._clock = clock

    def observe(self, fresh: Memory, stored_as: str | None = None) -> ObserverReport:
        """Judge the memories in view of `fresh`. `stored_as` is the id the fresh memory was stored under."""
        if fresh.embedding is None:
            raise ValidationError("the fresh memory must be embedded")
        p = self._policy
        candidates = self._store.query(Filter(near=fresh.pose, radius=p.same_place_m))
        in_view = [
            m
            for m in candidates
            if m.id not in (fresh.id, stored_as)
            and m.role == "episodic"
            and m.pose.heading_difference(fresh.pose) <= p.same_heading_rad
        ]
        confirmed = missed = superseded = 0
        updates: list[Memory] = []
        for m in in_view:
            similarity = _cosine(fresh.embedding, m.embedding)
            if similarity >= p.confirm_similarity:
                confirmed += 1
                if m.misses:
                    updates.append(replace(m, misses=0, last_miss=0.0))
                continue
            if m.last_miss and fresh.timestamp - m.last_miss < p.visit_gap_s:
                continue  # same visit, already counted
            missed += 1
            misses = m.misses + 1
            gone = misses >= p.misses_to_supersede
            superseded += gone
            updates.append(
                replace(
                    m,
                    misses=misses,
                    last_miss=fresh.timestamp,
                    superseded=m.superseded or gone,
                    superseded_at=fresh.timestamp if gone else m.superseded_at,
                )
            )
        if updates:
            self._store.upsert(updates)
        return ObserverReport(len(in_view), confirmed, missed, superseded)


def _cosine(a: Vector, b: Vector | None) -> float:
    if b is None:  # pragma: no cover - stores refuse unembedded memories
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na and nb else 0.0
