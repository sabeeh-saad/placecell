"""Portable RGB-D sessions for offline replay; exported images have independent ownership."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from placecell.depth import DepthSnapshot
from placecell.errors import ValidationError
from placecell.memory import Evidence, EvidenceKind, Pose, memory_id
from placecell.pipeline import Observation


class RecordingWriter:
    """One serialized camera stream per new session directory. Capture only; no provider calls."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self._last: float | None = None
        self._last_id = ""
        self._scope: tuple[str, str] | None = None

    def append(self, observation: Observation) -> None:
        identity = memory_id(observation.robot_id, observation.camera_id, observation.timestamp)
        scope = observation.robot_id, observation.camera_id
        if not math_valid_timestamp(observation.timestamp, self._last) or identity == self._last_id:
            raise ValidationError("recording timestamps must increase")
        if self._scope is not None and scope != self._scope:
            raise ValidationError("a recording contains one robot and camera")
        if observation.evidence.kind is not EvidenceKind.FRAME:
            raise ValidationError("RGB-D recordings require frames")
        with Path(observation.evidence.uri.removeprefix("file://")).open("rb") as source:
            raw = source.read(8_000_001)
        if len(raw) > 8_000_000 or not raw:
            raise ValidationError("recording image must contain at most 8 MB")
        if raw.startswith(b"\xff\xd8\xff"):
            suffix = ".jpg"
        elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
            suffix = ".png"
        else:
            raise ValidationError("recording image must be PNG or JPEG")
        name = f"frame-{round(observation.timestamp * 1_000_000)}{suffix}"
        with (self.directory / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        obs = replace(observation, evidence=replace(observation.evidence, uri=name, managed=False))
        row = {
            "version": 1,
            "observation_id": identity,
            "observation": asdict(obs),
        }
        with (self.directory / "observations.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._last = observation.timestamp
        self._last_id, self._scope = identity, scope


def read_recording(path: str | Path) -> Iterator[Observation]:
    manifest = Path(path).expanduser().resolve()
    last: float | None = None
    last_id = ""
    scope: tuple[str, str] | None = None
    with manifest.open() as stream:
        while line := stream.readline(2_000_001):
            if len(line) > 2_000_000:
                raise ValidationError("recording row exceeds size limit")
            try:
                row = json.loads(line)
                if row["version"] != 1:
                    raise ValidationError("unsupported recording version")
                data: dict[str, Any] = row["observation"]
                uri = Path(data["evidence"]["uri"])
                source = (manifest.parent / uri).resolve()
                if uri.is_absolute() or not source.is_relative_to(manifest.parent):
                    raise ValidationError("recording images must remain within the session directory")
                data["evidence"] = Evidence(
                    EvidenceKind(data["evidence"]["kind"]),
                    str(source),
                    data["evidence"].get("digest", ""),
                    managed=False,
                )
                data["pose"] = Pose(**data["pose"])
                if data.get("depth") is not None:
                    data["depth"]["map_from_camera"] = tuple(data["depth"]["map_from_camera"])
                    data["depth"] = DepthSnapshot(**data["depth"])
                observation = Observation(**data)
                identity = memory_id(observation.robot_id, observation.camera_id, observation.timestamp)
                current_scope = observation.robot_id, observation.camera_id
                if (
                    identity != row["observation_id"]
                    or identity == last_id
                    or not math_valid_timestamp(observation.timestamp, last)
                    or (scope is not None and current_scope != scope)
                    or observation.evidence.kind is not EvidenceKind.FRAME
                ):
                    raise ValidationError("recording identities or timestamp order are invalid")
            except (KeyError, TypeError, ValueError) as e:
                raise ValidationError(f"invalid recording row: {e}") from e
            last = observation.timestamp
            last_id, scope = identity, current_scope
            yield observation


def math_valid_timestamp(timestamp: float, previous: float | None) -> bool:
    import math

    return math.isfinite(timestamp) and timestamp >= 0 and (previous is None or timestamp > previous)
