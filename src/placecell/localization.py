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
        self._std = (0.0, 0.0)
        self._last_stamp = -math.inf
        self._last_clock = -math.inf
        self._generation = 0

    def _invalidate(self) -> None:
        if self._sample is not None:
            self._generation += 1
        self._sample = None

    def _now(self) -> float:
        now = self._clock()
        if not math.isfinite(now) or now < self._last_clock:
            self._invalidate()
            self._last_stamp = -math.inf
        self._last_clock = now if math.isfinite(now) else math.inf
        return now

    @property
    def generation(self) -> int:
        """Changes when trust is lost, even if a valid estimate recovers before polling."""
        with self._lock:
            self._ready()
            return self._generation

    def update(self, timestamp: float, pose: Pose, covariance: Sequence[float]) -> bool:
        with self._lock:
            self._ready()
            now = self._now()
            if math.isfinite(timestamp) and timestamp <= self._last_stamp:
                return False
            if not math.isfinite(timestamp) or timestamp <= 0 or not 0 <= now - timestamp <= self._policy.max_age_s:
                self._invalidate()
                return False
            self._last_stamp = timestamp
            if pose.frame_id != self._frame or pose.map_id != self._map or len(covariance) != 36:
                self._invalidate()
                return False
            try:
                matrix = np.asarray(covariance, dtype=float).reshape(6, 6)[np.ix_([0, 1, 5], [0, 1, 5])]
            except (TypeError, ValueError, OverflowError):
                self._invalidate()
                return False
            if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-9):
                self._invalidate()
                return False
            if np.linalg.eigvalsh(matrix).min() < -1e-9 or np.any(np.diag(matrix) <= 0):
                self._invalidate()
                return False
            p = self._policy
            if (
                np.linalg.eigvalsh(matrix[:2, :2]).max() > p.max_position_std_m**2
                or matrix[2, 2] > p.max_yaw_std_rad**2
            ):
                self._invalidate()
                return False
            self._sample = timestamp, self._monotonic(), pose
            self._std = (float(np.sqrt(np.linalg.eigvalsh(matrix[:2, :2]).max())), math.sqrt(matrix[2, 2]))
            return True

    def invalidate(self) -> None:
        with self._lock:
            self._invalidate()

    def ready(self) -> bool:
        with self._lock:
            return self._ready()

    def uncertainty_at(self, timestamp: float) -> tuple[float, float] | None:
        """Accepted planar position/yaw standard deviations for a recent capture."""
        with self._lock:
            if not self._ready() or self._sample is None or not math.isfinite(timestamp):
                return None
            if not 0 <= self._clock() - timestamp <= self._policy.max_age_s:
                return None
            if abs(timestamp - self._sample[0]) > self._policy.max_age_s:
                return None
            return self._std

    def _ready(self) -> bool:
        now = self._now()
        if self._sample is None:
            return False
        stamp, received, _pose = self._sample
        ready = (
            0 <= now - stamp <= self._policy.max_age_s and 0 <= self._monotonic() - received <= self._policy.max_age_s
        )
        if not ready:
            self._invalidate()
        return ready

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
