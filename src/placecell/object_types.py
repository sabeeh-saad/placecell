"""Object identities and their independent, bounded observation histories."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Literal, Protocol

from placecell.depth import Box, ObjectPosition
from placecell.errors import ValidationError
from placecell.memory import Evidence, Memory


@dataclass(frozen=True)
class Detection:
    label: str
    description: str
    box: Box

    def __post_init__(self) -> None:
        if (
            not self.label.strip()
            or len(self.label) > 100
            or not self.description.strip()
            or len(self.description) > 500
        ):
            raise ValidationError("objects need a short label and visual description")


class ObjectDetector(Protocol):
    def detect(self, image: Evidence) -> list[Detection]: ...

    def absent(self, reference_png: bytes, image: Evidence, region: Box) -> bool:
        """True only for a clearly visible, empty old location; occlusion/uncertainty return False."""
        ...


@dataclass(frozen=True)
class ObjectRecord:
    id: str
    robot_id: str
    camera_id: str
    frame_id: str
    map_id: str
    label: str
    first_seen: float
    last_seen: float
    position: ObjectPosition | None = None
    status: Literal["present", "missing", "ambiguous"] = "present"
    misses: int = 0
    last_miss: float = 0
    revision: int = 1

    def __post_init__(self) -> None:
        if not all((self.id, self.robot_id, self.camera_id, self.frame_id, self.label)):
            raise ValidationError("object identity, scope and label must be nonempty")
        if self.status not in {"present", "missing", "ambiguous"} or self.misses < 0 or self.revision < 1:
            raise ValidationError("invalid object state")
        if not all(math.isfinite(t) and t >= 0 for t in (self.first_seen, self.last_seen, self.last_miss)):
            raise ValidationError("invalid object timestamps")
        if self.last_seen < self.first_seen:
            raise ValidationError("object last_seen precedes first_seen")


@dataclass(frozen=True)
class ObjectView:
    object_id: str
    memory: Memory
    """Capture pose, scope and crop vectors. Evidence references the original full frame."""
    box: Box
    crop_png: bytes = field(repr=False, compare=False)

    def image_url(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.crop_png).decode("ascii")


@dataclass(frozen=True)
class ObjectHit:
    object: ObjectRecord
    view: ObjectView
    similarity: float


@dataclass(frozen=True)
class ObjectEvent:
    timestamp: float
    kind: str
    position: ObjectPosition | None
