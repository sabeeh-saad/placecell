"""Operator corrections and refinement requests, maintenance passes and periodic diagnostics.

Nothing here imports rclpy. The node's timers submit the passes to its maintenance worker
through the node, so a replaced `node._run_refiner` or `node._run_curator` is what runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from placecell.corrections import correction_now
from placecell.errors import PlacecellError, ValidationError
from placecell.lifecycle import remove_local_file
from placecell.mission_context import MissionContext
from placecell.ros2.components import Components
from placecell.ros2.depth import PendingImages
from placecell.ros2.workers import BoundedTasks
from placecell.store.base import EVERYTHING
from placecell.tracing import TraceStore


class Housekeeping:
    """What the `~/correct` and `~/refine` topics and the maintenance and diagnostics timers run."""

    def __init__(
        self,
        parts: Components,
        *,
        pending: PendingImages,
        mission_context: MissionContext | None,
        mission_traces: TraceStore | None,
        command_tasks: BoundedTasks | None,
        clock: Callable[[], float],
        log: Any,
    ) -> None:
        self._store, self._corrections, self._object_policy = parts.store, parts.corrections, parts.object_policy
        self._refiner, self._curator, self._consolidator = parts.refiner, parts.curator, parts.consolidator
        self._worker, self._questions, self._maintenance = parts.worker, parts.questions, parts.maintenance
        self._sensors, self._pending_images = parts.sensors, pending
        self._mission_context, self._mission_traces = mission_context, mission_traces
        self._command_tasks, self._memory_time, self._log = command_tasks, clock, log

    def on_correct(self, msg: Any) -> None:
        """JSON: {"memory_id": ..., "verdict": "right"|"wrong", "question": ..., "note": ...}."""
        try:
            data = json.loads(msg.data)
            correction = correction_now(
                str(data["memory_id"]),
                str(data["verdict"]),
                str(data.get("question", "")),
                str(data.get("note", "")),
            )
            if self._store.get(correction.memory_id) is None:
                raise ValidationError("correction refers to an unavailable scene memory")
            self._corrections.record(correction)
            if correction.verdict == "wrong":
                accepted = self._store.refinements.request(correction.memory_id, "operator correction")
                if not accepted:
                    self._log.warning("Correction saved; recheck queue full or memory is ineligible.")
        except (ValueError, KeyError, TypeError, OSError, PlacecellError) as e:
            self._log.warning(f"ignored correction: {e}")
            return

    def on_refine(self, msg: Any) -> None:
        """JSON: {"memory_id": ..., "action": "recheck"|"rollback"}."""
        try:
            data = json.loads(msg.data)
            identity, action = str(data["memory_id"]), data.get("action", "recheck")
            if action == "recheck":
                accepted = self._store.refinements.request(identity)
            elif action == "rollback" and self._refiner is not None:
                accepted = self._refiner.rollback(identity)
            else:
                raise ValidationError("unknown refinement action or refinement is disabled")
            self._log.info(f"refinement {action} for {identity}: {'accepted' if accepted else 'skipped'}")
        except (ValueError, KeyError, TypeError, PlacecellError) as e:
            self._log.warning(f"ignored refinement request: {e}")

    def run_refiner(self) -> None:
        if self._refiner is not None:
            report = self._refiner.run()
            if report.attempted:
                self._log.info(f"memory refinement: {report}")

    def reference_available(self, data: dict[str, Any]) -> bool:
        """Whether a mission context event still points at a present object or current memory."""
        if data.get("object_id"):
            record = self._store.objects.get(data["object_id"])
            return record is not None and record.status == "present"
        if data.get("memory_id"):
            memory = self._store.get(data["memory_id"])
            return memory is not None and not memory.superseded
        return True

    def run_curator(self) -> None:
        self._store.drain_cleanup(remove_local_file)
        before = self._memory_time() - self._object_policy.retention_s
        for _ in range(128):
            if not self._store.objects.prune(before, limit=1):
                break
            self._store.drain_cleanup(remove_local_file)
        report = self._curator.run()
        self._corrections.prune(m.id for batch in self._store.iter_query(EVERYTHING) for m in batch)
        if self._mission_context is not None:
            self._mission_context.prune()
        maintain = getattr(self._store, "maintain", None)
        if maintain is not None:
            maintain()
        if report.removed or report.discredited or report.history_pruned:
            self._log.info(
                f"curator removed {report.removed} memories, discredited {report.discredited}, "
                f"pruned {report.history_pruned} sightings"
            )

    def run_consolidator(self) -> None:
        if self._consolidator is None:  # pragma: no cover - timer only exists with a consolidator
            return
        try:
            report = self._consolidator.run()
        except PlacecellError as e:
            self._log.error(f"consolidation failed: {e}")
            return
        if report.summaries:
            self._log.info(f"consolidated {report.folded} memories into {report.summaries} summaries")

    def diagnostics(self) -> None:
        stats = self._worker.health()
        self._log.info(f"ingestion: {stats}, dropped={self._worker.dropped}, objects={self._store.objects.count()}")
        self._log.info(
            f"queues: questions={self._questions.health()}, maintenance={self._maintenance.health()}, "
            f"commands={self._command_tasks.health() if self._command_tasks is not None else None}, "
            f"images={self._pending_images.health()}, sensors={self._sensors.health()}"
        )
        if self._mission_traces is not None:
            health = self._mission_traces.health()
            self._log.info(f"mission traces: {health}")
            if health["dropped_events"] or health["write_errors"] or not health["writer_alive"]:
                self._log.warning("Mission trace capture is incomplete; inspect trace health counters.")
