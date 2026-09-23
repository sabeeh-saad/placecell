"""Non-blocking Nav2 action adapter, with injectable transport for offline tests."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from placecell.errors import ValidationError
from placecell.memory import Pose
from placecell.navigation import Destination, NavigationCommands, NavigationEvent
from placecell.tracing import TraceContext, current_trace


@dataclass
class _Trip:
    request_id: str
    callback: Callable[[NavigationEvent], None]
    started: float
    handle: Any = None
    cancel_requested: bool = False
    cancel_pending: bool = False
    cancel_started: float | None = None
    response_reported: bool = False
    response_received: bool = False
    deadline_reported: bool = False
    cancel_reported: bool = False
    trace: TraceContext | None = None
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace_started: float = field(default_factory=time.perf_counter)
    last_feedback: float = -math.inf

    def record(self, stage: str, *, kind: str = "event", **data: Any) -> None:
        if self.trace:
            self.trace.emit(stage, kind=kind, transport_id=self.request_id, **data)


class Nav2Navigator:
    def __init__(
        self,
        client: Any,
        make_goal: Callable[[Pose], Any],
        *,
        response_timeout_s: float = 10.0,
        trip_timeout_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
        feedback_interval_s: float = 0.2,
    ) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in (response_timeout_s, trip_timeout_s, feedback_interval_s)):
            raise ValidationError("navigation timeouts must be finite and positive")
        self._client, self._make_goal, self._clock = client, make_goal, clock
        self._response_timeout, self._trip_timeout = response_timeout_s, trip_timeout_s
        self._feedback_interval = feedback_interval_s
        self._lock = threading.RLock()
        self._trip: _Trip | None = None

    def send(self, request_id: str, destination: Destination, callback: Callable[[NavigationEvent], None]) -> None:
        with self._lock:
            if self._trip is not None:
                raise ValidationError("a Nav2 goal is already pending")
            trip = _Trip(request_id, callback, self._clock(), trace=current_trace())
            self._trip = trip
        trip.record(
            "navigation",
            kind="start",
            span_id=trip.span_id,
            pose=asdict(destination.pose),
            source=destination.source,
            memory_id=destination.memory.id if destination.memory else None,
            object_id=destination.object_id,
        )
        try:
            ready = self._client.server_is_ready()
            goal = self._make_goal(destination.pose)
        except Exception as e:
            self._finish(trip, "unavailable", f"Nav2 could not prepare a goal: {e}")
            return
        if not ready:
            self._finish(trip, "unavailable", "Nav2 action server is not ready. No goal was sent.")
            return
        try:
            trip.record("nav2.dispatch", pose=asdict(destination.pose))
            future = self._client.send_goal_async(goal, feedback_callback=lambda m: self._feedback(trip, m))
            future.add_done_callback(lambda f: self._accepted(trip, f))
        except Exception as e:
            self._uncertain(trip, f"Nav2 goal submission could not be confirmed: {e}")

    def _accepted(self, trip: _Trip, future: Any) -> None:
        with self._lock:
            if trip.response_received:
                return
            trip.response_received = True
        try:
            handle = future.result()
            trip.record("nav2.goal_response", accepted=bool(handle.accepted))
            if not handle.accepted:
                self._finish(trip, "rejected", "Nav2 rejected the destination.")
                return
            with self._lock:
                current = self._trip is trip
                if current:
                    trip.handle = handle
                    if self._clock() - trip.started >= self._response_timeout:
                        trip.cancel_requested = True
                    cancel = trip.cancel_requested
            if not current:
                trip.record("nav2.late_acceptance", action="cancel")
                handle.cancel_goal_async()
                return
            result = handle.get_result_async()
            result.add_done_callback(lambda f: self._result(trip, f))
            if cancel:
                self._cancel(trip)
            else:
                self._emit(trip, NavigationEvent("navigating", "Nav2 accepted the goal."))
        except Exception as e:
            self._uncertain(trip, f"Nav2 goal response could not be read: {e}")
            self._cancel(trip)

    def _result(self, trip: _Trip, future: Any) -> None:
        try:
            response = future.result()
            status = int(response.status)
            # action_msgs/GoalStatus values are shared by Humble and Jazzy.
            error_code = getattr(response.result, "error_code", 0)
            trip.record("nav2.result", status=status, error_code=error_code)
            if status == 4 and not error_code:
                self._finish(
                    trip, "succeeded", "Reached the navigation goal. Camera observations continue updating memory."
                )
            elif status == 5:
                self._finish(trip, "canceled", "Nav2 confirmed cancellation.")
            elif status in (4, 6):
                detail = getattr(response.result, "error_msg", "")
                self._finish(trip, "failed", f"Nav2 could not reach the destination. {detail}".strip())
            else:
                self._cancel(trip)
                self._uncertain(trip, f"Nav2 returned an unexpected goal status: {status}")
        except Exception as e:
            self._cancel(trip)
            self._uncertain(trip, f"Nav2 result could not be confirmed: {e}")

    def _feedback(self, trip: _Trip, message: Any) -> None:
        try:
            distance = float(message.feedback.distance_remaining)
        except (AttributeError, TypeError, ValueError):
            return
        if not math.isfinite(distance) or distance < 0:
            return
        with self._lock:
            now = self._clock()
            if self._trip is not trip or now - trip.last_feedback < self._feedback_interval:
                return
            trip.last_feedback = now
            canceling = trip.cancel_requested
        self._emit(trip, NavigationEvent("canceling" if canceling else "navigating", distance_remaining=distance))

    def cancel(self, request_id: str) -> None:
        with self._lock:
            trip = self._trip
            if trip is None or trip.request_id != request_id:
                return
        self._cancel(trip)

    def _cancel(self, trip: _Trip) -> None:
        """Async callbacks use object identity, never a possibly reused request ID."""
        with self._lock:
            if self._trip is not trip:
                return
            trip.cancel_requested = True
            if trip.handle is None or trip.cancel_pending:
                return
            trip.cancel_pending = True
            trip.cancel_started = self._clock()
            trip.cancel_reported = False
            handle = trip.handle
        trip.record("nav2.cancel_requested")
        try:
            future = handle.cancel_goal_async()
            self._emit(trip, NavigationEvent("canceling", "Waiting for Nav2 to cancel the goal."))
            future.add_done_callback(lambda f: self._canceled(trip, f))
        except Exception as e:
            with self._lock:
                trip.cancel_pending = False
            self._uncertain(trip, f"Nav2 cancellation request failed: {e}")

    def _canceled(self, trip: _Trip, future: Any) -> None:
        try:
            response = future.result()
            trip.record("nav2.cancel_acknowledgement", accepted=bool(response.goals_canceling))
            if not response.goals_canceling:
                with self._lock:
                    trip.cancel_pending = False
                self._emit(
                    trip, NavigationEvent("cancel_failed", "Nav2 did not accept cancellation. The trip remains active.")
                )
            # Acceptance is not completion. Only the action result may report 'canceled'.
        except Exception as e:
            with self._lock:
                trip.cancel_pending = False
            self._uncertain(trip, f"Nav2 cancellation response could not be confirmed: {e}")

    def poll(self) -> None:
        """Call from a short ROS timer; timeouts request cancellation and retain uncertain ownership."""
        event = None
        with self._lock:
            trip = self._trip
            if trip is None:
                return
            now = self._clock()
            if trip.handle is None and now - trip.started >= self._response_timeout and not trip.response_reported:
                trip.response_reported = trip.cancel_requested = True
                event = NavigationEvent(
                    "uncertain", "Nav2 has not acknowledged the goal. It will be canceled if accepted late."
                )
            elif now - trip.started >= self._trip_timeout and not trip.deadline_reported:
                trip.deadline_reported = trip.cancel_requested = True
                event = NavigationEvent("canceling", "Navigation time limit reached; requesting cancellation.")
            elif (
                trip.cancel_started is not None
                and now - trip.cancel_started >= self._response_timeout
                and not trip.cancel_reported
            ):
                trip.cancel_reported = True
                event = NavigationEvent(
                    "uncertain", "Nav2 has not confirmed that the robot stopped. Check navigation status."
                )
        if event is not None:
            self._cancel(trip)
            self._emit(trip, event)

    def _uncertain(self, trip: _Trip, message: str) -> None:
        with self._lock:
            if self._trip is not trip:
                return
            trip.cancel_requested = True
        self._emit(trip, NavigationEvent("uncertain", message))

    def _emit(self, trip: _Trip, event: NavigationEvent) -> None:
        with self._lock:
            if self._trip is not trip:
                return
            event = replace(event, cancel_requested=trip.cancel_requested)
            if trip.cancel_requested and event.state == "navigating":
                event = replace(event, state="canceling")
        trip.record(
            "nav2.event",
            state=event.state,
            message=event.message,
            distance_remaining=event.distance_remaining,
            cancel_requested=event.cancel_requested,
        )
        trip.callback(event)  # callbacks never run while holding the transport lock

    def _finish(self, trip: _Trip, state: str, message: str) -> None:
        with self._lock:
            if self._trip is not trip:
                return
            if self._clock() - trip.started >= self._trip_timeout:
                trip.cancel_requested = True
            # Carry intent with the terminal result even if its earlier event is still
            # waiting for the controller lock. Clearing ownership linearizes here.
            event = NavigationEvent(state, message, cancel_requested=trip.cancel_requested)
            self._trip = None
        trip.record(
            "navigation",
            kind="end",
            span_id=trip.span_id,
            state=state,
            message=message,
            cancel_requested=event.cancel_requested,
            duration_ms=(time.perf_counter() - trip.trace_started) * 1000,
        )
        trip.callback(event)


def create_navigator(node: Any, action_name: str, response_timeout_s: float, trip_timeout_s: float) -> Nav2Navigator:
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup

    def make_goal(pose: Pose) -> Any:
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = pose.frame_id
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(pose.x), float(pose.y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(pose.yaw / 2), math.cos(pose.yaw / 2)
        return goal

    return Nav2Navigator(
        ActionClient(node, NavigateToPose, action_name, callback_group=ReentrantCallbackGroup()),
        make_goal,
        response_timeout_s=response_timeout_s,
        trip_timeout_s=trip_timeout_s,
    )


def create_navigation_timers(node: Any, navigator: Nav2Navigator, commands: NavigationCommands) -> None:
    """Wall-clock deadlines must progress with paused ROS time and slow image callbacks."""
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.clock import Clock, ClockType

    for poll in (navigator.poll, commands.poll):
        node.create_timer(
            0.1, poll, clock=Clock(clock_type=ClockType.STEADY_TIME), callback_group=MutuallyExclusiveCallbackGroup()
        )
