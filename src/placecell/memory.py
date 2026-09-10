"""The data model: a memory is one moment the robot saw, with where and when it saw it."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from placecell.errors import FrameMismatchError, ValidationError

SCHEMA_VERSION = 1
"""Bumped whenever the stored shape of a memory changes. Stores record it per collection."""


@dataclass(frozen=True, slots=True)
class Pose:
    """Planar robot pose in a map frame. Yaw in radians, counter-clockwise from +x."""

    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"
    map_id: str = ""

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw"):
            if not math.isfinite(getattr(self, name)):
                raise ValidationError(f"pose.{name} must be finite")
        if not self.frame_id:
            raise ValidationError("pose.frame_id must not be empty")

    def same_frame(self, other: Pose) -> bool:
        return self.frame_id == other.frame_id and self.map_id == other.map_id

    def distance_to(self, other: Pose) -> float:
        """Euclidean distance in the plane. Poses must share frame and map."""
        if not self.same_frame(other):
            raise FrameMismatchError(
                f"cannot compare {self.frame_id}/{self.map_id!r} with {other.frame_id}/{other.map_id!r}"
            )
        return math.hypot(self.x - other.x, self.y - other.y)

    def heading_difference(self, other: Pose) -> float:
        """Absolute yaw difference wrapped into [0, pi]."""
        d = (self.yaw - other.yaw + math.pi) % (2 * math.pi) - math.pi
        return abs(d)


class EvidenceKind(str, Enum):
    """What kind of media backs a memory. New kinds are added here, never as new tables."""

    FRAME = "frame"
    CLIP = "clip"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Reference to the media a memory is based on. The bytes live outside the store."""

    kind: EvidenceKind
    uri: str
    digest: str = ""
    """Content hash of the media, so a replayed source is recognised as the same evidence."""
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValidationError("evidence.uri must not be empty")
        if self.duration_s < 0 or not math.isfinite(self.duration_s):
            raise ValidationError("evidence.duration_s must be a finite, non-negative number")
        if self.kind is EvidenceKind.CLIP and self.duration_s == 0:
            raise ValidationError("a clip needs a positive duration_s")
        if self.kind is EvidenceKind.FRAME and self.duration_s != 0:
            raise ValidationError("a frame has no duration")


def memory_id(robot_id: str, camera_id: str, timestamp: float) -> str:
    """Deterministic id, so retries and replays of the same observation never duplicate it."""
    for name, value in (("robot_id", robot_id), ("camera_id", camera_id)):
        if not value or ":" in value:
            raise ValidationError(f"{name} must be non-empty and must not contain ':'")
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValidationError("timestamp must be a finite unix time in seconds")
    return f"{robot_id}:{camera_id}:{round(timestamp * 1000)}"


@dataclass(frozen=True, slots=True)
class Memory:
    """One remembered moment. Immutable; lifecycle operations return updated copies."""

    id: str
    robot_id: str
    camera_id: str
    timestamp: float
    pose: Pose
    evidence: Evidence | None = None
    caption: str = ""
    embedding: np.ndarray | None = field(default=None, compare=False, repr=False)
    model: str = ""
    """Name of the embedding model that produced `embedding`. Empty while unembedded."""
    confidence: float = 1.0
    observations: int = 1
    last_seen: float = -1.0
    superseded: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.last_seen < 0:
            object.__setattr__(self, "last_seen", self.timestamp)
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError("confidence must be within [0, 1]")
        if self.observations < 1:
            raise ValidationError("observations must be at least 1")
        if self.last_seen < self.timestamp:
            raise ValidationError("last_seen cannot precede the first observation")
        if (self.embedding is None) != (self.model == ""):
            raise ValidationError("embedding and model are set together or not at all")
        if self.embedding is not None:
            object.__setattr__(self, "embedding", as_vector(self.embedding))

    @classmethod
    def create(
        cls,
        robot_id: str,
        camera_id: str,
        timestamp: float,
        pose: Pose,
        evidence: Evidence | None = None,
        caption: str = "",
    ) -> Memory:
        """Build an unembedded memory with its id derived from robot, camera and time."""
        return cls(
            id=memory_id(robot_id, camera_id, timestamp),
            robot_id=robot_id,
            camera_id=camera_id,
            timestamp=timestamp,
            pose=pose,
            evidence=evidence,
            caption=caption,
        )

    def with_embedding(self, vector: np.ndarray, model: str) -> Memory:
        if not model:
            raise ValidationError("model name must not be empty")
        return replace(self, embedding=as_vector(vector), model=model)

    def effective_confidence(self, now: float, half_life_s: float) -> float:
        """Confidence after exponential decay since the memory was last reinforced."""
        if half_life_s <= 0:
            raise ValidationError("half_life_s must be positive")
        age = max(0.0, now - self.last_seen)
        return float(self.confidence * 0.5 ** (age / half_life_s))


def as_vector(vector: np.ndarray) -> np.ndarray:
    """Validate and normalise a single embedding to a finite 1-D float32 array."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValidationError(f"embedding must be a non-empty 1-D vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError("embedding contains NaN or infinity")
    arr.setflags(write=False)
    return arr
