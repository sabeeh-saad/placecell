"""Live sensor provenance, independent of ingestion sampling and source-clock pauses."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

from placecell.errors import ValidationError


class SensorHealth:
    """Require advancing, validated RGB/TF and aligned depth within a bounded age.

    Clock resets latch a fault: timestamp-based memory identities cannot be reused
    safely in another simulation epoch. Recovery requires a fresh run/collection.
    The ROS jump callback only sets an event, never waits for a controller lock.
    """

    def __init__(
        self,
        max_age_s: float = 5.0,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValidationError("sensor maximum age must be finite and positive")
        self._age, self._clock, self._monotonic = max_age_s, clock, monotonic
        self._lock = threading.Lock()
        self.clock_changed = threading.Event()
        self._last_clock = -math.inf
        self._last_stamp = -math.inf
        self._camera: tuple[float, float] | None = None
        self._depth: tuple[float, float] | None = None
        self._camera_generation = self._depth_generation = 0

    def _revoke(self, *, camera: bool = True) -> None:
        if camera and self._camera is not None:
            self._camera_generation += 1
            self._camera = None
        if self._depth is not None:
            self._depth_generation += 1
            self._depth = None

    def _refresh(self) -> float:
        now = self._clock()
        if not math.isfinite(now) or now < self._last_clock:
            self.clock_changed.set()
        self._last_clock = now
        mono = self._monotonic()
        if self.clock_changed.is_set():
            self._revoke()
        else:
            for camera, sample in ((True, self._camera), (False, self._depth)):
                if sample is not None and not (
                    0 <= now - sample[0] <= self._age and 0 <= mono - sample[1] <= self._age
                ):
                    self._revoke(camera=camera)
        return now

    def observe(self, stamp: float, *, camera: bool, depth: bool) -> bool:
        """Offer a validated capture; repeats never refresh receipt age."""
        with self._lock:
            now = self._refresh()
            if self.clock_changed.is_set():
                return False
            if not math.isfinite(stamp) or not 0 < stamp <= now or now - stamp > self._age:
                self._revoke()
                return False
            if stamp <= self._last_stamp:
                return False
            self._last_stamp = stamp
            if not camera:
                self._revoke()
                return False
            sample = (stamp, self._monotonic())
            self._camera = sample
            if depth:
                self._depth = sample
            # A single unpaired RGB capture does not invalidate a still-fresh depth
            # sample. It also never refreshes depth age: a sustained missing stream
            # expires in _refresh(), preserving trust-loss history across recovery.
            return True

    def ready(self, *, camera: bool = False, depth: bool = False) -> bool:
        with self._lock:
            self._refresh()
            return (
                not self.clock_changed.is_set()
                and (not camera or self._camera is not None)
                and (not depth or self._depth is not None)
            )

    def generation(self, *, depth: bool = False) -> tuple[int, int]:
        with self._lock:
            self._refresh()
            return self._camera_generation, self._depth_generation if depth else 0
