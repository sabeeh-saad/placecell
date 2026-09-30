"""Admission limits for authoritative memory state, independent of background maintenance."""

from dataclasses import dataclass

from placecell.errors import ValidationError


@dataclass(frozen=True)
class StoreLimits:
    max_memories: int = 10000
    max_sightings: int = 1024
    max_refinement_jobs: int = 256
    max_cleanup: int = 2048
    evict_at_capacity: bool = True
    """At max_memories, a new memory replaces the least valuable one instead of being refused."""

    def __post_init__(self) -> None:
        sizes = (self.max_memories, self.max_sightings, self.max_refinement_jobs, self.max_cleanup)
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValidationError("store limits must be positive integers")
        if type(self.evict_at_capacity) is not bool:
            raise ValidationError("evict_at_capacity must be a boolean")
