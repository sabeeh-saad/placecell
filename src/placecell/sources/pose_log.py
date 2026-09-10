"""A time-indexed pose track, interpolated so any evidence timestamp gets a pose."""

from __future__ import annotations

import bisect
import csv
import math
from collections.abc import Iterable
from pathlib import Path

from placecell.errors import ValidationError
from placecell.memory import Pose


class PoseTrack:
    """Poses sorted by time. Lookups between samples interpolate position and heading."""

    def __init__(self, samples: Iterable[tuple[float, Pose]], tolerance_s: float = 0.5) -> None:
        rows = sorted(samples, key=lambda s: s[0])
        if not rows:
            raise ValidationError("a pose track needs at least one sample")
        for i in range(1, len(rows)):
            if not rows[0][1].same_frame(rows[i][1]):
                raise ValidationError("all poses of a track must share frame and map")
        self._times = [t for t, _ in rows]
        self._poses = [p for _, p in rows]
        self._tolerance_s = tolerance_s

    @classmethod
    def from_csv(cls, path: str | Path, tolerance_s: float = 0.5) -> PoseTrack:
        """Columns: timestamp, x, y, yaw, optional frame_id and map_id. Header required."""
        with Path(path).open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"timestamp", "x", "y", "yaw"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValidationError(f"pose csv needs columns {sorted(required)}")
            samples = [
                (
                    float(r["timestamp"]),
                    Pose(
                        float(r["x"]),
                        float(r["y"]),
                        float(r["yaw"]),
                        r.get("frame_id") or "map",
                        r.get("map_id") or "",
                    ),
                )
                for r in reader
            ]
        return cls(samples, tolerance_s)

    def __len__(self) -> int:
        return len(self._times)

    @property
    def span(self) -> tuple[float, float]:
        return self._times[0], self._times[-1]

    def at(self, timestamp: float) -> Pose:
        """Pose at the time. Slightly outside the track the nearest end is used; further out is an error."""
        first, last = self.span
        if timestamp < first - self._tolerance_s or timestamp > last + self._tolerance_s:
            raise ValidationError(f"time {timestamp} is outside the pose track [{first}, {last}]")
        if timestamp <= first:
            return self._poses[0]
        if timestamp >= last:
            return self._poses[-1]
        i = bisect.bisect_right(self._times, timestamp)
        t0, t1 = self._times[i - 1], self._times[i]
        a, b = self._poses[i - 1], self._poses[i]
        if t1 == t0:
            return b
        w = (timestamp - t0) / (t1 - t0)
        dyaw = (b.yaw - a.yaw + math.pi) % (2 * math.pi) - math.pi
        return Pose(a.x + w * (b.x - a.x), a.y + w * (b.y - a.y), a.yaw + w * dyaw, a.frame_id, a.map_id)
