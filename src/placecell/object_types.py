"""Object identities and their independent, bounded observation histories."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from placecell.depth import Box, ObjectPosition
from placecell.errors import ValidationError
from placecell.memory import Evidence, Memory

if TYPE_CHECKING:
    from placecell.verification import SceneVerdict


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


@runtime_checkable
class ObjectComparator(Protocol):
    def compare(self, references: tuple[bytes, ...], candidate: bytes) -> SceneVerdict: ...


@dataclass(frozen=True)
class ArrivalComparison:
    selected: int
    identity: SceneVerdict
    destination: SceneVerdict

    def __post_init__(self) -> None:
        if (
            type(self.selected) is not int
            or not -1 <= self.selected < 64
        ):
            raise ValidationError("invalid arrival comparison selection")
        for verdict in (self.identity, self.destination):
            if (
                verdict.result not in {"matched", "not_matched", "uncertain"}
                or not isinstance(verdict.reason, str)
                or not verdict.reason.strip()
                or len(verdict.reason) > 1000
                or (self.selected == -1 and verdict.result == "matched")
            ):
                raise ValidationError("invalid arrival comparison verdict")


@runtime_checkable
class ArrivalComparator(Protocol):
    def compare_arrival(
        self, references: tuple[bytes, ...], candidates: tuple[bytes, ...], image: Evidence, target: str
    ) -> ArrivalComparison: ...


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
    position_timestamp: float | None = None

    def __post_init__(self) -> None:
        if not all((self.id, self.robot_id, self.camera_id, self.frame_id, self.label)):
            raise ValidationError("object identity, scope and label must be nonempty")
        if self.status not in {"present", "missing", "ambiguous"} or self.misses < 0 or self.revision < 1:
            raise ValidationError("invalid object state")
        if not all(math.isfinite(t) and t >= 0 for t in (self.first_seen, self.last_seen, self.last_miss)):
            raise ValidationError("invalid object timestamps")
        if self.last_seen < self.first_seen:
            raise ValidationError("object last_seen precedes first_seen")
        if self.position_timestamp is not None and (
            self.position is None
            or not math.isfinite(self.position_timestamp)
            or not 0 <= self.position_timestamp <= self.last_seen
        ):
            raise ValidationError("invalid object position timestamp")


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
