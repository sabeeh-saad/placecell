"""The node's background workers: one ordered ingest writer and small bounded task pools.

They hold no rclpy objects, so admission, retry and shutdown are tested without ROS.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from placecell.errors import ProviderError, ValidationError
from placecell.pipeline import Ingester, Observation


class IngestWorker:
    """One ordered writer consuming durable jobs. Provider calls never hold a shared lock."""

    def __init__(
        self,
        ingester: Ingester,
        lock: threading.Lock | None,
        batch_size: int,
        max_queue: int,
        log: Any,
        *,
        max_attempts: int = 5,
        retry_delay_s: float = 1,
    ) -> None:
        if (
            any(type(v) is not int or v < 1 for v in (batch_size, max_queue, max_attempts))
            or max_attempts > 32
            or not math.isfinite(retry_delay_s)
            or retry_delay_s < 0
        ):
            raise ValidationError("invalid worker limits")
        self._ingester, self._batch_size, self._max_queue = ingester, batch_size, max_queue
        self._log, self._max_attempts, self._retry_delay_s = log, max_attempts, retry_delay_s
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="placecell-ingest", daemon=True)
        self._submit_lock = threading.Lock()
        self.dropped = 0
        self._last_drop_log = -math.inf

    def reject(self) -> None:
        """Count pre-encoding camera drops without retaining image data or flooding logs."""
        with self._submit_lock:
            self.dropped += 1
            now = time.monotonic()
            log = now - self._last_drop_log >= 5
            if log:
                self._last_drop_log = now
            dropped = self.dropped
        if log:
            self._log.warning(f"ingest queue full or stopped, dropped {dropped} observations so far")

    def health(self) -> dict[str, float | int | bool]:
        result: dict[str, float | int | bool] = dict(self._ingester.jobs.stats())
        with self._submit_lock:
            result.update(capacity=self._max_queue, dropped=self.dropped, stopped=self._stop.is_set())
        return result

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10) -> bool:
        with self._submit_lock:
            self._stop.set()
        self._wake.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def has_capacity(self) -> bool:
        return not self._stop.is_set() and self._ingester.jobs.stats()["queued"] < self._max_queue

    def submit(self, observation: Observation) -> bool:
        with self._submit_lock:
            if not self._stop.is_set() and self._ingester.jobs.enqueue(observation, self._max_queue):
                self._wake.set()
                return True
        self.reject()
        # The journal pins all accepted evidence, including failed jobs and duplicate submissions.
        self._ingester.discard([observation])
        return False

    def _run(self) -> None:
        try:
            self._work_loop()
        except Exception as e:
            self._log.error(f"ingest worker stopped; queued work retained: {e}")
        finally:
            self._stop.set()

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            jobs = self._ingester.jobs.pending(self._batch_size)
            if not jobs:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            try:
                report = self._ingester.ingest([job.observation for job in jobs], preselected=True)
            except Exception as e:
                completed = [job.id for job in jobs if self._ingester.persisted(job.observation)]
                self._ingester.jobs.complete(completed)
                self._ingester.jobs.fail(
                    (job.id for job in jobs if job.id not in completed),
                    str(e),
                    max_attempts=self._max_attempts,
                    retry_delay_s=self._retry_delay_s,
                    defer_queue=True,
                    retry_after_s=e.retry_after_s if isinstance(e, ProviderError) else 0,
                )
                self._log.error(f"ingest failed; work retained for retry: {e}")
            else:
                self._ingester.jobs.complete(job.id for job in jobs)
                self._log.info(
                    f"ingested {report.accepted}/{report.received}: {report.inserted} new, {report.merged} reinforced"
                    + (f", {report.unsupported} unsupported" if report.unsupported else "")
                    + (
                        f", {report.objects_skipped_ambiguous} ambiguous object detections skipped"
                        if report.objects_skipped_ambiguous
                        else ""
                    )
                )
                if report.object_errors:
                    self._log.warning(
                        f"object tracking failed for {len(report.object_errors)} observations, "
                        f"scene memories kept: {report.object_errors[-1]}"
                    )
                if report.objects_skipped_capacity:
                    self._log.warning(
                        f"object capacity reached, {report.objects_skipped_capacity} new objects not stored; "
                        "prune old objects or raise object_max_records"
                    )
            try:
                self._ingester.discard([])  # drain cleanup intents after job ownership is released
            except OSError as e:
                self._log.error(f"evidence cleanup deferred: {e}")


class BoundedTasks:
    """Fixed workers, atomic admission/shutdown, and observable bounded waiting work."""

    def __init__(self, workers: int, capacity: int, log: Any) -> None:
        if any(type(v) is not int or v < 1 for v in (workers, capacity)):
            raise ValidationError("task limits must be positive")
        self._queue: queue.Queue[tuple[Callable[..., None], tuple[Any, ...], str]] = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._keys: set[str] = set()
        self._active = 0
        self._counts = {
            "accepted": 0,
            "completed": 0,
            "failed": 0,
            "discarded": 0,
            "rejected_full": 0,
            "rejected_stopped": 0,
            "coalesced": 0,
            "high_water": 0,
        }
        self._log = log
        self._threads = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for thread in self._threads:
            thread.start()

    def submit(self, function: Callable[..., None], *args: Any, key: str = "") -> bool:
        with self._condition:
            if self._stop.is_set():
                self._counts["rejected_stopped"] += 1
                return False
            if key and key in self._keys:
                self._counts["coalesced"] += 1
                return True
            try:
                self._queue.put_nowait((function, args, key))
            except queue.Full:
                self._counts["rejected_full"] += 1
                return False
            if key:
                self._keys.add(key)
            self._counts["accepted"] += 1
            self._counts["high_water"] = max(self._counts["high_water"], self._queue.qsize())
            self._condition.notify()
            return True

    def health(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                **self._counts,
                "active": self._active,
                "queued": self._queue.qsize(),
                "capacity": self._queue.maxsize,
                "workers": len(self._threads),
                "stopped": self._stop.is_set(),
            }

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stop.is_set() or not self._queue.empty())
                if self._stop.is_set():
                    return
                function, args, key = self._queue.get_nowait()
                self._active += 1
            outcome = "completed"
            try:
                function(*args)
            except Exception as e:
                outcome = "failed"
                self._log.error(f"background task failed: {e}")
            finally:
                with self._condition:
                    self._active -= 1
                    self._counts[outcome] += 1
                    self._keys.discard(key)
                    self._queue.task_done()
                # Do not retain the last request's images/context while this worker is idle.
                del function, args

    def stop(self, timeout: float = 10) -> bool:
        with self._condition:
            self._stop.set()
            while not self._queue.empty():
                _, _, key = self._queue.get_nowait()
                self._keys.discard(key)
                self._queue.task_done()
                self._counts["discarded"] += 1
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout=timeout / len(self._threads))
        return all(not thread.is_alive() for thread in self._threads)
