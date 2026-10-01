"""Status bookkeeping for the navigation controller: numbering, the snapshot, traces and context rows.

`NavigationCommands` decides each status and sends it under its lock. `StatusPublisher` only
shapes and records it, never locks and never calls the publish callback, so the controller
stays the one place that orders status delivery.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING

from placecell.navigation_state import EXECUTION_FAILURES
from placecell.tracing import TraceContext, current_trace

if TYPE_CHECKING:
    from placecell.mission_context import MissionContext
    from placecell.navigation import Destination, NavigationUpdate


def trace_destination(destination: Destination | None) -> dict[str, object] | None:
    if destination is None:
        return None
    return {
        "label": destination.label,
        "target": destination.target,
        "source": destination.source,
        "pose": asdict(destination.pose),
        "memory_id": destination.memory.id if destination.memory else None,
        "object_id": destination.object_id,
        "object_revision": destination.object_revision,
    }


class StatusPublisher:
    """One controller instance's status sequence, snapshot and conversation-context rows."""

    def __init__(self, instance_id: str, snapshot: NavigationUpdate) -> None:
        self.instance_id = instance_id
        self.sequence = 0
        self.snapshot = snapshot
        self.context_ok = True
        self._last_context_state: tuple[str, str] | None = None

    def number(self, update: NavigationUpdate, *, state_update: bool) -> NavigationUpdate:
        """Stamp the next sequence number; a state update also becomes the snapshot."""
        self.sequence += 1
        update = replace(update, instance_id=self.instance_id, sequence=self.sequence)
        # A (str, Enum) member formats differently on Python 3.10 and 3.11+; publish plain strings.
        assert type(update.state) is str, f"status state must be a plain str, not {type(update.state).__name__}"
        if state_update:
            self.snapshot = update
        return update

    @staticmethod
    def trace(update: NavigationUpdate, context: TraceContext | None, trip_trace: TraceContext | None) -> None:
        """Record the status in the given trace, else the active one, else the trip's for the same request."""
        context = context or current_trace()
        if context is None and trip_trace and update.request_id == trip_trace.request_id:
            context = trip_trace
        if context:
            context.emit(
                "status",
                state=update.state,
                # An exception's text can hold a provider response body; traces keep its type only.
                message="" if update.error_type else update.message,
                error_type=update.error_type,
                destination=trace_destination(update.destination),
                choices=[trace_destination(choice) for choice in update.choices],
                distance_remaining=update.distance_remaining,
                object_result=update.object_result,
                search_attempt=update.search_attempt,
                mission_destinations=update.mission_destinations,
                failure_stage=update.failure_stage,
            )

    @staticmethod
    def trace_for(update: NavigationUpdate, trip_trace: TraceContext | None) -> TraceContext | None:
        """The trip's trace, unless the caller already runs in this request's own trace."""
        active_trace = current_trace()
        if active_trace and update.request_id == active_trace.request_id:
            return active_trace
        return trip_trace

    @staticmethod
    def attribute(
        update: NavigationUpdate, mission_id: str, mission_step: int, mission_destinations: tuple[str, ...]
    ) -> NavigationUpdate:
        """Attribute an unstaged failure to execution and name the mission it belongs to."""
        if not update.failure_stage and update.state in EXECUTION_FAILURES:
            update = replace(update, failure_stage="execution")
        if mission_id:
            update = replace(
                update,
                mission_id=mission_id,
                mission_step=mission_step,
                mission_destinations=mission_destinations,
            )
        return update

    def record(self, update: NavigationUpdate, context: MissionContext | None) -> NavigationUpdate:
        """Save each new (request, state) outcome; a failed write is appended to the message."""
        key = (update.request_id, update.state)
        if context is not None and key != self._last_context_state:
            destination = update.destination
            try:
                context.record(
                    update.request_id,
                    "status",
                    # Structured outcome only: messages can quote model or provider text.
                    {
                        "state": update.state,
                        "mission_id": update.mission_id,
                        "step": update.mission_step,
                        "destinations": update.mission_destinations,
                        "target": destination.target if destination else "",
                        "memory_id": destination.memory.id if destination and destination.memory else "",
                        "object_id": destination.object_id if destination else "",
                        "failure_stage": update.failure_stage,
                        "object_result": update.object_result,
                        "destination_source": destination.source if destination else "",
                        "place": destination.label if destination and destination.source == "named_place" else "",
                        "error_type": update.error_type,
                    },
                )
                self._last_context_state = key
            except Exception as e:
                self.context_ok = False
                update = replace(update, message=f"{update.message} Context persistence failed: {e}")
        return update

    def record_instruction(self, context: MissionContext | None, request_id: str, text: str) -> Exception | None:
        """Save the user's words; the controller reports a returned error."""
        if context is None:
            return None
        try:
            context.record(request_id, "instruction", {"text": text})
            self.context_ok = True
            return None
        except Exception as e:
            self.context_ok = False
            return e
