"""Bounded, goal-specific reconciliation over the public ROS action services."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from placecell.navigation_ownership import NavigationOwnership


class GoalRecovery:
    def __init__(
        self,
        ownership: NavigationOwnership,
        result_client: Any,
        cancel_client: Any,
        result_request: Callable[[str], Any],
        cancel_request: Callable[[str], Any],
        *,
        timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ownership = ownership
        self._clients = {"result": result_client, "cancel": cancel_client}
        self._requests = {"result": result_request, "cancel": cancel_request}
        self._timeout, self._clock = timeout_s, clock
        self._lock = threading.RLock()
        self._pending: dict[str, tuple[Any, float]] = {}
        self._next: dict[str, float] = {"result": 0, "cancel": 0}
        state = ownership.snapshot()
        self._goal_id = state["goal_id"] if state["state"] == "pending" else ""
        self._blocked = state["state"] != "clean"
        self._message = (
            f"Recovering previous Nav2 goal {self._goal_id}; waiting for a confirmed terminal result."
            if self._goal_id
            else "Nav2 startup ownership is unknown; independently stop/reset Nav2 and attest clean ownership."
        )

    @property
    def block_reason(self) -> str:
        with self._lock:
            return self._message if self._blocked else ""

    def poll(self) -> None:
        with self._lock:
            if not self._blocked or not self._goal_id:
                return
            now = self._clock()
            for kind, client in self._clients.items():
                pending = self._pending.get(kind)
                if pending is not None:
                    future, started = pending
                    if future.done():
                        del self._pending[kind]
                        self._next[kind] = now + self._timeout
                        try:
                            response = future.result()
                            # Cancel acknowledgement is never evidence of termination.
                            if kind == "result" and response.status in (4, 5, 6):
                                self._ownership.terminal(
                                    self._goal_id, {4: "succeeded", 5: "canceled", 6: "failed"}[response.status]
                                )
                                self._blocked = False
                                self._abandon_all()
                                return
                        except Exception:
                            self._message = "Nav2 recovery or its durable journal failed; ownership remains uncertain."
                    elif now - started >= self._timeout:
                        client.remove_pending_request(future)
                        del self._pending[kind]
                        self._next[kind] = now + self._timeout
                    continue
                if now >= self._next[kind] and client.service_is_ready():
                    self._next[kind] = now + self._timeout
                    try:
                        future = client.call_async(self._requests[kind](self._goal_id))
                        self._pending[kind] = (future, now)
                    except Exception:
                        self._message = "Nav2 recovery transport failed; ownership remains uncertain."

    def _abandon_all(self) -> None:
        for kind, (future, _) in self._pending.items():
            self._clients[kind].remove_pending_request(future)
        self._pending.clear()

    def close(self) -> None:
        with self._lock:
            self._abandon_all()


def create_recovery(node: Any, action: str, ownership: NavigationOwnership, timeout_s: float) -> GoalRecovery:
    from action_msgs.srv import CancelGoal
    from nav2_msgs.action import NavigateToPose
    from rclpy.callback_groups import ReentrantCallbackGroup
    from unique_identifier_msgs.msg import UUID

    # Action remapping applies to the base name, not to separately created child services.
    action = node.resolve_topic_name(action)

    def result_request(identity: str) -> Any:
        request = NavigateToPose.Impl.GetResultService.Request()
        request.goal_id = UUID(uuid=list(bytes.fromhex(identity)))
        return request

    def cancel_request(identity: str) -> Any:
        request = CancelGoal.Request()
        request.goal_info.goal_id = UUID(uuid=list(bytes.fromhex(identity)))
        # The zero timestamp plus NONZERO exact UUID selects only this goal.
        return request

    return GoalRecovery(
        ownership,
        node.create_client(
            NavigateToPose.Impl.GetResultService,
            action + "/_action/get_result",
            callback_group=ReentrantCallbackGroup(),
        ),
        node.create_client(CancelGoal, action + "/_action/cancel_goal", callback_group=ReentrantCallbackGroup()),
        result_request,
        cancel_request,
        timeout_s=timeout_s,
    )
