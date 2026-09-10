"""In-process store backed by a dictionary and a brute-force cosine search.

The reference implementation: small, obviously correct, and the semantics every other
backend is tested against. Fine for a single robot's session or for tests; not for a fleet.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

import numpy as np

from placecell.errors import ModelMismatchError, ValidationError
from placecell.memory import Memory
from placecell.store.base import CollectionInfo, Filter, Hit


class InMemoryStore:
    def __init__(self, info: CollectionInfo) -> None:
        self._info = info
        self._rows: dict[str, Memory] = {}
        self._lock = threading.RLock()
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []

    @property
    def info(self) -> CollectionInfo:
        return self._info

    def upsert(self, memories: Iterable[Memory]) -> int:
        with self._lock:
            written = 0
            for memory in memories:
                self._check(memory)
                self._rows[memory.id] = memory
                written += 1
            if written:
                self._matrix = None
            return written

    def get(self, memory_id: str) -> Memory | None:
        with self._lock:
            return self._rows.get(memory_id)

    def delete(self, ids: Iterable[str]) -> int:
        with self._lock:
            removed = sum(1 for i in ids if self._rows.pop(i, None) is not None)
            if removed:
                self._matrix = None
            return removed

    def delete_where(self, where: Filter) -> int:
        with self._lock:
            doomed = [m.id for m in self._rows.values() if where.matches(m)]
            return self.delete(doomed)

    def query(self, where: Filter | None = None, limit: int | None = None) -> list[Memory]:
        if limit is not None and limit < 0:
            raise ValidationError("limit must not be negative")
        with self._lock:
            rows = [m for m in self._rows.values() if (where or Filter()).matches(m)]
        rows.sort(key=lambda m: (m.timestamp, m.id))
        return rows[:limit] if limit is not None else rows

    def search(self, vector: np.ndarray, k: int, where: Filter | None = None) -> list[Hit]:
        if k < 1:
            raise ValidationError("k must be at least 1")
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (self._info.dimension,):
            raise ValidationError(f"query vector must have shape ({self._info.dimension},), got {query.shape}")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm
        with self._lock:
            matrix, ids = self._index()
            keep = np.fromiter(((where or Filter()).matches(self._rows[i]) for i in ids), dtype=bool, count=len(ids))
            if not keep.any():
                return []
            scores = matrix[keep] @ query
            kept_ids = [i for i, flag in zip(ids, keep, strict=True) if flag]
            order = np.argsort(-scores, kind="stable")[:k]
            return [Hit(self._rows[kept_ids[j]], float(scores[j])) for j in order]

    def count(self, where: Filter | None = None) -> int:
        with self._lock:
            return sum(1 for m in self._rows.values() if (where or Filter()).matches(m))

    def close(self) -> None:
        with self._lock:
            self._rows.clear()
            self._matrix = None
            self._ids = []

    def _check(self, memory: Memory) -> None:
        if memory.embedding is None:
            raise ValidationError(f"memory {memory.id} has no embedding")
        if memory.model != self._info.model:
            raise ModelMismatchError(
                f"collection {self._info.name!r} is bound to {self._info.model!r}, "
                f"memory {memory.id} was embedded by {memory.model!r}"
            )
        if memory.embedding.shape != (self._info.dimension,):
            raise ValidationError(
                f"memory {memory.id} has dimension {memory.embedding.shape[0]}, collection has {self._info.dimension}"
            )

    def _index(self) -> tuple[np.ndarray, list[str]]:
        """Row matrix with unit-length rows, rebuilt lazily after writes."""
        if self._matrix is None:
            self._ids = list(self._rows)
            if self._ids:
                stacked = np.stack([_vector_of(self._rows[i]) for i in self._ids])
                norms = np.linalg.norm(stacked, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._matrix = stacked / norms
            else:
                self._matrix = np.zeros((0, self._info.dimension), dtype=np.float32)
        return self._matrix, self._ids


def _vector_of(memory: Memory) -> np.ndarray:
    if memory.embedding is None:  # pragma: no cover - upsert refuses unembedded memories
        raise ValidationError(f"memory {memory.id} has no embedding")
    return memory.embedding
