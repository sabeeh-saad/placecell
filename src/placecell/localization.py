"""Bounded, timestamped localization quality checks without ROS dependencies."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from placecell.errors import ValidationError
from placecell.memory import Pose


@dataclass(frozen=True)
class LocalizationPolicy:
    max_age_s: float = 5.0
    max_position_std_m: float = 0.3
    max_yaw_std_rad: float = 0.35
    max_pose_difference_m: float = 0.5
    max_heading_difference_rad: float = 0.5

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValidationError("localization limits must be finite and positive")


class LocalizationGate:
    """Require a recent valid covariance estimate in the configured map.

    Invalid estimates revoke readiness immediately. Receipt age also uses monotonic
    time so a paused simulation or replayed message cannot keep a pose trusted.
    """

    def __init__(
        self,
        frame_id: str,
        map_id: str,
        policy: LocalizationPolicy | None = None,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._frame, self._map = frame_id, map_id
        self._policy, self._clock, self._monotonic = policy or LocalizationPolicy(), clock, monotonic
        self._lock = threading.Lock()
        self._sample: tuple[float, float, Pose] | None = None
        self._last_stamp = -math.inf

    def update(self, timestamp: float, pose: Pose, covariance: Sequence[float]) -> bool:
        with self._lock:
            now = self._clock()
            if self._last_stamp > now:
                self._sample = None
                self._last_stamp = -math.inf
            if math.isfinite(timestamp) and timestamp <= self._last_stamp:
                return False
            self._sample = None
            if not math.isfinite(timestamp) or not 0 <= now - timestamp <= self._policy.max_age_s:
                return False
            self._last_stamp = timestamp
            if pose.frame_id != self._frame or pose.map_id != self._map or len(covariance) != 36:
                return False
            matrix = np.asarray(covariance, dtype=float).reshape(6, 6)[np.ix_([0, 1, 5], [0, 1, 5])]
            if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-9):
                return False
            if np.linalg.eigvalsh(matrix).min() < -1e-9 or np.any(np.diag(matrix) <= 0):
                return False
            p = self._policy
            if (
                np.linalg.eigvalsh(matrix[:2, :2]).max() > p.max_position_std_m**2
                or matrix[2, 2] > p.max_yaw_std_rad**2
            ):
                return False
            self._sample = timestamp, self._monotonic(), pose
            return True

    def invalidate(self) -> None:
        with self._lock:
            self._sample = None

    def ready(self) -> bool:
        with self._lock:
            return self._ready()

    def _ready(self) -> bool:
        if self._sample is None:
            return False
        stamp, received, _pose = self._sample
        return (
            0 <= self._clock() - stamp <= self._policy.max_age_s
            and 0 <= self._monotonic() - received <= self._policy.max_age_s
        )

    def accepts(self, pose: Pose, timestamp: float) -> bool:
        with self._lock:
            if not self._ready() or self._sample is None:
                return False
            stamp, _received, estimate = self._sample
            p = self._policy
            return (
                math.isfinite(timestamp)
                and 0 <= self._clock() - timestamp <= p.max_age_s
                and abs(timestamp - stamp) <= p.max_age_s
                and pose.same_frame(estimate)
                and pose.distance_to(estimate) <= p.max_pose_difference_m
                and pose.heading_difference(estimate) <= p.max_heading_difference_rad
            )
