"""Admission limits for authoritative memory state, independent of background maintenance."""

from dataclasses import dataclass

from placecell.errors import ValidationError


@dataclass(frozen=True)
class StoreLimits:
    max_memories: int = 10000
    max_sightings: int = 1024
    max_refinement_jobs: int = 256
    max_cleanup: int = 2048

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValidationError("store limits must be positive integers")
