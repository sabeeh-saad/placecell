"""Store contract.

A store holds one collection: memories embedded by exactly one model. It answers three
kinds of question, similarity, time and place, and every filter is meant to be pushed
down into the backend rather than applied after fetching. `Filter.matches` is the
reference semantics that backends must reproduce.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike

from placecell.errors import ValidationError
from placecell.memory import SCHEMA_VERSION, Memory, Pose


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """Identity of a collection. Model and dimension are fixed for its whole life."""

    name: str
    model: str
    dimension: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name or not self.model:
            raise ValidationError("collection name and model must not be empty")
        if self.dimension < 1:
            raise ValidationError("dimension must be positive")


@dataclass(frozen=True, slots=True)
class Filter:
    """Scalar constraints on memories. All set fields must hold (logical and)."""

    robot_id: str | None = None
    camera_id: str | None = None
    time_from: float | None = None
    """Inclusive lower bound on the observation time."""
    time_to: float | None = None
    """Exclusive upper bound on the observation time."""
    near: Pose | None = None
    radius: float | None = None
    include_superseded: bool = False

    def __post_init__(self) -> None:
        if (self.near is None) != (self.radius is None):
            raise ValidationError("near and radius are set together")
        if self.radius is not None and (self.radius <= 0 or not math.isfinite(self.radius)):
            raise ValidationError("radius must be a positive, finite distance")
        if self.time_from is not None and self.time_to is not None and self.time_from > self.time_to:
            raise ValidationError("time_from must not exceed time_to")

    def matches(self, memory: Memory) -> bool:
        """Reference semantics of the filter, used by in-process stores and by tests of others."""
        if not self.include_superseded and memory.superseded:
            return False
        if self.robot_id is not None and memory.robot_id != self.robot_id:
            return False
        if self.camera_id is not None and memory.camera_id != self.camera_id:
            return False
        if self.time_from is not None and memory.timestamp < self.time_from:
            return False
        if self.time_to is not None and memory.timestamp >= self.time_to:
            return False
        if self.near is not None and self.radius is not None:
            if not memory.pose.same_frame(self.near):
                return False
            if memory.pose.distance_to(self.near) > self.radius:
                return False
        return True


EVERYTHING = Filter(include_superseded=True)
"""A filter that keeps every memory, superseded ones included."""


@dataclass(frozen=True, slots=True)
class Hit:
    """A memory returned by a similarity search with its cosine score in [-1, 1]."""

    memory: Memory
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Contract every store backend fulfils."""

    @property
    def info(self) -> CollectionInfo: ...

    def upsert(self, memories: Iterable[Memory]) -> int:
        """Insert or replace by id. Returns the number written. Rejects foreign models."""
        ...

    def get(self, memory_id: str) -> Memory | None: ...

    def delete(self, ids: Iterable[str]) -> int:
        """Remove by id. Returns how many existed."""
        ...

    def delete_where(self, where: Filter) -> int:
        """Remove every memory the filter matches. Returns how many were removed."""
        ...

    def query(self, where: Filter | None = None, limit: int | None = None) -> list[Memory]:
        """Memories matching the filter, oldest first. No vector involved."""
        ...

    def search(self, vector: ArrayLike, k: int, where: Filter | None = None) -> list[Hit]:
        """The k most similar memories among those the filter keeps, best first."""
        ...

    def count(self, where: Filter | None = None) -> int: ...

    def close(self) -> None: ...
