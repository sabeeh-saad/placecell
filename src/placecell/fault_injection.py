"""Offline fault contracts using the production mission controller and Nav2 adapter.

Providers and action transport are scripted; clocks advance without sleeping. This
checks software behavior, not model quality, ROS delivery, physics or stopping time.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from placecell.chat import ChatMessage, ChatReply, ToolCall
from placecell.errors import ValidationError
from placecell.localization import LocalizationGate
from placecell.memory import Evidence, EvidenceKind, Memory, Pose
from placecell.mission_context import MissionContext
from placecell.missions import MissionPlanner, PlanReviewAgent
from placecell.navigation import DestinationResolver, NavigationCommands, NavigationUpdate
from placecell.pipeline import Observation
from placecell.providers._http import RetryPolicy
from placecell.providers.chat import OpenAICompatibleChat
from placecell.providers.hashing import HashingEmbedder
from placecell.retrieval import Recall
from placecell.ros2.depth import PendingImages
from placecell.ros2.navigation import Nav2Navigator
from placecell.ros2.node import navigation_payload
from placecell.sensors import SensorHealth
from placecell.store.base import CollectionInfo
from placecell.store.in_memory import InMemoryStore
from placecell.tracing import TraceStore, read_trace
from placecell.verification import SceneVerdict, VisionVerifier


def _reply(review: bool = False) -> ChatReply:
    arguments: dict[str, Any] = {"decision": "approve" if review else "ready", "message": "Scripted decision."}
    if not review:
        arguments["destinations"] = ["printer", "cupboard"]
    return ChatReply(
        None, (ToolCall("fixture", "review_navigation_plan" if review else "propose_navigation_plan", arguments),)
    )


class _Model:
    def __init__(self, reply: ChatReply) -> None:
        self.reply = reply
        self.error: Exception | None = None
        self.before: Callable[[], None] = lambda: None
        self.calls = 0

    def complete(self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]) -> ChatReply:
        self.calls += 1
        self.before()
        if self.error:
            raise self.error
        return self.reply


class _Handle:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.result: Future[Any] = Future()
        self.ack: Future[Any] = Future()
        self.cancel_calls = 0

    def get_result_async(self) -> Future[Any]:
        return self.result

    def cancel_goal_async(self) -> Future[Any]:
        self.cancel_calls += 1
        return self.ack


class _Client:
    def __init__(self) -> None:
        self.ready = True
        self.error: Exception | None = None
        self.goals: list[Pose] = []
        self.responses: list[Future[Any]] = []
        self.feedback: list[Callable[[Any], None]] = []
        self.handles: list[_Handle] = []

    def server_is_ready(self) -> bool:
        return self.ready

    def send_goal_async(self, goal: Pose, feedback_callback: Callable[[Any], None]) -> Future[Any]:
        self.goals.append(goal)  # Record attempts even if delivery cannot be confirmed.
        self.feedback.append(feedback_callback)
        if self.error:
            raise self.error
        future: Future[Any] = Future()
        self.responses.append(future)
        return future

    def accept(self, accepted: bool = True) -> _Handle:
        handle = _Handle(accepted)
        self.handles.append(handle)
        self.responses[-1].set_result(handle)
        return handle


class _Context(MissionContext):
    def __init__(self, path: Path) -> None:
        super().__init__(path, scope="fault-robot:office:operator")
        self.fail_kind = self.fail_state = ""
        self.fail_read = False
        self.faults = 0

    def record(self, request_id: str, kind: str, payload: dict[str, Any]) -> None:
        if kind == self.fail_kind and (not self.fail_state or payload.get("state") == self.fail_state):
            self.faults += 1
            raise sqlite3.OperationalError("injected context write failure")
        super().record(request_id, kind, payload)

    def recent(self, *, exclude_request_id: str = "", limit: int = 20, max_chars: int = 16000) -> list[dict[str, Any]]:
        if self.fail_read:
            self.faults += 1
            raise sqlite3.OperationalError("injected context read failure")
        return super().recent(exclude_request_id=exclude_request_id, limit=limit, max_chars=max_chars)


class _Verifier:
    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None
        self.before: Callable[[], None] = lambda: None
        self.result = "matched"

    def verify(self, target: str, image_url: str) -> SceneVerdict:
        self.calls += 1
        self.before()
        if self.error:
            raise self.error
        return SceneVerdict(self.result, "Scripted verdict; no visual model was called.")  # type: ignore[arg-type]


class _Rig:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.elapsed, self.stamp = 0.0, 1000.0
        self.gate = LocalizationGate("map", "office", clock=lambda: self.stamp, monotonic=lambda: self.elapsed)
        self.pose = Pose(1, 2, map_id="office")
        self.refresh_localization()
        self.model, self.reviewer = _Model(_reply()), _Model(_reply(True))
        self.verifier = _Verifier()
        self.embedder = HashingEmbedder(64)
        self.store = InMemoryStore(CollectionInfo("faults", self.embedder.model_name, self.embedder.dimension))
        self.context = _Context(directory / "context.sqlite3")
        self.traces = TraceStore(directory / "traces.sqlite3", queue_size=1024)
        self.client = _Client()
        self.tasks: list[Callable[[], None]] = []
        self.events: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.queue_ready = True
        self.navigator = Nav2Navigator(
            self.client, lambda pose: pose, response_timeout_s=2, trip_timeout_s=10, clock=lambda: self.elapsed
        )
        self.resolver = DestinationResolver(
            self.store,
            Recall(self.store, self.embedder, clock=lambda: self.stamp),
            robot_id="fault-robot",
            camera_id="front",
            map_id="office",
            places={"printer": self.pose, "cupboard": Pose(8, 2, map_id="office")},
            verifier=self.verifier,
            clock=lambda: self.stamp,
        )
        self.commands = self.new_controller()

    def new_controller(self) -> NavigationCommands:
        return NavigationCommands(
            self.resolver,
            self.navigator,
            self.submit,
            self.publish,
            mission_planner=MissionPlanner(self.model, PlanReviewAgent(self.reviewer)),
            mission_context=self.context,
            trace_store=self.traces,
            clock=lambda: self.elapsed,
            observation_clock=lambda: self.stamp,
            localization_ready=self.gate.ready,
            localization_generation=lambda: self.gate.generation,
            request_timeout_s=5,
            arrival_timeout_s=3,
        )

    def refresh_localization(self) -> None:
        covariance = [0.0] * 36
        for index in (0, 7, 35):
            covariance[index] = 0.01
        self.gate.update(self.stamp, self.pose, covariance)

    def advance(self, seconds: float, *, localization: bool = True, source_clock: bool = True) -> None:
        self.elapsed += seconds
        if source_clock:
            self.stamp += seconds
        if localization:
            self.refresh_localization()

    def submit(self, task: Callable[[], None]) -> bool:
        if not self.queue_ready:
            return False
        self.tasks.append(task)
        return True

    def drain(self) -> None:
        for _ in range(20):
            if not self.tasks:
                return
            self.tasks.pop(0)()
        raise RuntimeError("fault runner task bound exceeded")

    def publish(self, update: NavigationUpdate) -> None:
        self.events.append({"elapsed_s": self.elapsed, **json.loads(navigation_payload(update))})

    def check(self, name: str, actual: Any, expected: Any) -> None:
        self.checks.append({"name": name, "actual": actual, "expected": expected, "passed": actual == expected})

    @property
    def state(self) -> str:
        return str(self.events[-1]["state"]) if self.events else "idle"

    def checkpoint(self, name: str, state: str, goals: int, busy: bool) -> None:
        self.check(f"{name}: status", self.state, state)
        self.check(f"{name}: dispatch attempts", len(self.client.goals), goals)
        self.check(f"{name}: owned mission", self.commands.busy, busy)

    def start(self) -> None:
        self.commands.handle("First visit the printer, then the cupboard")
        self.drain()

    def finish(self, status: int = 4) -> None:
        self.client.handles[-1].result.set_result(SimpleNamespace(status=status, result=SimpleNamespace(error_code=0)))

    def memory_destination(self) -> Evidence:
        # A valid one-pixel PNG for exercising file transport, not perception accuracy.
        path = self.directory / "view.png"
        path.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII="
            )
        )
        evidence = Evidence(EvidenceKind.FRAME, str(path))
        memory = Memory.create("fault-robot", "front", self.stamp, self.pose, evidence, "printer")
        memory = replace(memory, confidence=1.0, localization_checked=True)
        self.store.upsert(
            [memory.with_embedding(self.embedder.embed_text(["printer"])[0], self.embedder.model_name, kind="caption")]
        )
        self.resolver = DestinationResolver(
            self.store,
            Recall(self.store, self.embedder, clock=lambda: self.stamp),
            robot_id="fault-robot",
            camera_id="front",
            map_id="office",
            verifier=self.verifier,
            clock=lambda: self.stamp,
        )
        self.commands = self.new_controller()
        return evidence

    def close(self) -> None:
        # Pending fake goals are evidence, not real robot jobs. Do not alter reported state during teardown.
        self.context.close()
        self.store.close()
        self.traces.close()


class _ContractTransport:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self.body, self.status, self.calls = body, status, 0

    def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, str], Any]:
        self.calls += 1
        return self.status, {}, self.body


def _provider_contract_fault(rig: _Rig, fault: str) -> None:
    visual = fault.startswith("visual_")
    reviewing = fault.startswith("review_")
    arguments: dict[str, Any] = (
        {"result": "matched", "reason": "Scripted pixels."}
        if visual
        else dict(_reply(reviewing).tool_calls[0].arguments)
    )
    if fault.endswith("extra_field"):
        arguments["action"] = "drive_without_verification"
    if fault.endswith("oversized"):
        arguments["message"] = "x" * 1001
    raw = json.dumps(arguments)
    if fault.endswith("duplicate"):
        field = '"result":"not_matched",' if visual else '"decision":"reject",'
        raw = "{" + field + raw[1:]
    msg: dict[str, Any] = {"role": "assistant", "content": raw if visual else None}
    if not visual:
        msg["tool_calls"] = [
            {
                "id": "fixture",
                "type": "function",
                "function": {
                    "name": "review_navigation_plan" if reviewing else "propose_navigation_plan",
                    "arguments": raw,
                },
            }
        ]
    if fault.endswith("unknown_tool"):
        msg["tool_calls"][0]["function"]["name"] = "drive"
    if fault.endswith("refusal"):
        msg["refusal"] = "Provider refused the request."
    choice: dict[str, Any] = {"finish_reason": "stop" if visual else "tool_calls", "message": msg}
    if fault.endswith("missing_finish"):
        choice.pop("finish_reason")
    transport = _ContractTransport({"choices": [choice]}, 503 if fault.endswith("http_error") else 200)
    if visual:
        rig.memory_destination()
        rig.resolver._verifier = VisionVerifier("fixture", "http://offline.test", transport=transport)
    else:
        model = OpenAICompatibleChat("fixture", transport=transport, retry=RetryPolicy(attempts=1))
        rig.commands._mission_planner = MissionPlanner(
            rig.model if reviewing else model, PlanReviewAgent(model if reviewing else rig.reviewer)
        )
    rig.start()
    rig.checkpoint("invalid provider output refused", "not_found" if visual else "rejected", 0, False)
    rig.check("one bounded provider call", transport.calls, 1)
    rig.check("visible refusal reason", bool(rig.events[-1]["message"]), True)
    if not visual and not reviewing:
        rig.check("review not called after invalid proposal", rig.reviewer.calls, 0)


def _model_fault(rig: _Rig, fault: str) -> None:
    model = rig.reviewer if fault.startswith("review") else rig.model
    if fault.endswith("timeout"):
        model.error = TimeoutError("injected provider timeout")
    elif fault.endswith("malformed"):
        model.reply = ChatReply("unstructured answer")
    elif fault == "review_reject":
        model.reply = ChatReply(
            None, (ToolCall("fixture", "review_navigation_plan", {"decision": "reject", "message": "Rejected."}),)
        )
    elif fault.endswith("_stop"):
        model.before = lambda: rig.commands.handle("stop")
    else:

        def expire() -> None:
            rig.advance(5)
            rig.commands.poll()

        model.before = expire
    rig.start()
    expected = "canceled" if fault.endswith("_stop") else "not_found" if fault.endswith("late") else "rejected"
    rig.checkpoint("fault handled", expected, 0, False)
    rig.check("faulty provider called", model.calls, 1)
    if fault.startswith("planner"):
        rig.check("review never started", rig.reviewer.calls, 0)


def _cancellation_boundary(rig: _Rig, fault: str) -> None:
    """Force callback interleavings without relying on scheduler timing."""
    visual = fault.startswith("arrival_") or fault in {"lookup_stop", "nav_timeout_visual_result_race"}
    if visual:
        evidence = rig.memory_destination()
    if fault == "lookup_stop":
        rig.verifier.before = lambda: rig.commands.handle("stop")
    rig.start()
    if fault == "lookup_stop":
        rig.checkpoint("stop during candidate verification", "canceled", 0, False)
        return
    if fault.startswith("arrival_"):
        rig.client.accept()
        rig.finish()
        rig.checkpoint("waiting for image", "awaiting_observation", 1, True)
        if fault == "arrival_wait_stop":
            rig.commands.handle("stop")
        else:
            rig.advance(0.1)
            rig.verifier.before = lambda: rig.commands.handle("stop")
            rig.commands.observe(
                Observation("fault-robot", "front", rig.stamp, rig.pose, evidence, localization_checked=True)
            )
            rig.drain()
        rig.checkpoint("arrival canceled", "canceled", 1, False)
    elif fault == "nav_stop_before_acceptance":
        rig.commands.handle("stop")
        rig.checkpoint("stop with no handle", "canceling", 1, True)
        rig.commands.handle("go to cupboard")
        rig.checkpoint("replacement refused", "busy", 1, True)
        handle = rig.client.accept()
        rig.check("late handle canceled", handle.cancel_calls, 1)
        handle.ack.set_result(SimpleNamespace(goals_canceling=[1]))
        rig.checkpoint("ack is not terminal", "canceling", 1, True)
        rig.finish(5)
        rig.checkpoint("terminal releases ownership", "canceled", 1, False)
    elif fault == "nav_unpolled_late_acceptance":
        rig.advance(3)
        handle = rig.client.accept()
        rig.check("deadline enforced at acceptance without poll", handle.cancel_calls, 1)
        rig.finish()
        rig.checkpoint("late success does not advance", "canceled", 1, False)
    else:
        response_race = fault == "nav_response_result_race"
        if not response_race:
            rig.client.accept()
        rig.advance(11)
        if fault == "nav_unpolled_late_success":
            rig.finish()
        else:
            emit = rig.navigator._emit
            interleaved = False

            def result_first(trip: Any, event: Any) -> None:
                nonlocal interleaved
                if not interleaved and event.state in {"canceling", "uncertain"}:
                    interleaved = True
                    if response_race:
                        handle = _Handle(True)
                        handle.result.set_result(SimpleNamespace(status=4, result=SimpleNamespace(error_code=0)))
                        rig.client.handles.append(handle)
                        rig.client.responses[-1].set_result(handle)
                    else:
                        rig.finish()
                emit(trip, event)

            rig.navigator._emit = result_first  # type: ignore[method-assign]
            rig.navigator.poll()
            rig.check("result raced ahead of cancellation event", interleaved, True)
        rig.checkpoint("terminal carries intent", "destination_unverified" if visual else "canceled", 1, False)
    rig.check("no queued next destination", len(rig.tasks), 0)
    rig.check(
        "no mission or step success", any(e["state"] in {"succeeded", "step_succeeded"} for e in rig.events), False
    )


def _navigation_fault(rig: _Rig, fault: str) -> None:
    visual = fault == "nav_timeout_visual_success"
    late_acceptance = fault in {"nav_late_acceptance", "nav_late_acceptance_success"}
    late_success = fault in {"nav_timeout_late_success", "nav_timeout_visual_success", "nav_late_acceptance_success"}
    if visual:
        rig.memory_destination()
    if fault == "nav_unavailable":
        rig.client.ready = False
    if fault == "nav_send_error":
        rig.client.error = OSError("injected lost send response")
    rig.start()
    if fault in {"nav_unavailable", "nav_send_error"}:
        uncertain = fault == "nav_send_error"
        rig.checkpoint("submission", "uncertain" if uncertain else "unavailable", int(uncertain), uncertain)
        return
    if fault == "nav_rejected":
        rig.client.accept(False)
        rig.checkpoint("rejection", "rejected", 1, False)
        return
    if late_acceptance:
        rig.advance(2)
        rig.navigator.poll()
        rig.checkpoint("response deadline", "uncertain", 1, True)
    handle = rig.client.accept()
    if fault in {"nav_lost_result", "nav_timeout_late_success", "nav_timeout_visual_success"}:
        rig.advance(10)
        rig.navigator.poll()
    elif not late_acceptance:
        rig.commands.handle("stop")
    rig.check("cancellation requested", handle.cancel_calls, 1)
    rig.checkpoint("waiting for cancellation", "canceling", 1, True)
    if fault == "nav_cancel_rejected":
        handle.ack.set_result(SimpleNamespace(goals_canceling=[]))
        rig.checkpoint("cancel rejection", "cancel_failed", 1, True)
    elif not late_acceptance:
        rig.advance(2)
        rig.navigator.poll()
        rig.checkpoint("cancel result deadline", "uncertain", 1, True)
        handle.ack.set_result(SimpleNamespace(goals_canceling=["fixture"]))
        rig.check("cancel acknowledgement retains ownership", rig.commands.busy, True)
    rig.commands.handle("Visit the cupboard")
    rig.checkpoint("new request blocked", "busy", 1, True)
    if fault == "nav_lost_result":
        return  # No terminal result: intentional unresolved ownership, not a timeout success.
    rig.finish(4 if late_success else 5)
    rig.drain()
    rig.checkpoint("late terminal result", "destination_unverified" if visual else "canceled", 1, False)
    rig.check(
        "late result cannot complete a mission step",
        any(e["state"] in {"step_succeeded", "succeeded"} for e in rig.events),
        False,
    )
    previous = len(rig.events)
    rig.client.feedback[0](SimpleNamespace(feedback=SimpleNamespace(distance_remaining=0.0)))
    rig.check("late feedback ignored", len(rig.events), previous)


def _localization_fault(rig: _Rig, fault: str) -> None:
    before = fault == "localization_stale_before"
    if not before:
        rig.start()
        rig.client.accept()
    rig.advance(6, localization=False, source_clock=fault != "localization_paused_clock")
    rig.check("localization revoked", rig.gate.ready(), False)
    if before:
        rig.start()
        rig.checkpoint("admission blocked", "unavailable", 0, False)
    else:
        rig.commands.poll()
        rig.checkpoint("localization lost", "canceling", 1, True)
        rig.check("cancel sent", rig.client.handles[0].cancel_calls, 1)
        rig.finish(5)
        rig.checkpoint("stop confirmed", "canceled", 1, False)


def _arrival_fault(rig: _Rig, fault: str) -> None:
    evidence = rig.memory_destination()
    if fault == "visual_lookup_timeout":
        rig.verifier.error = TimeoutError("injected visual lookup timeout")
    rig.start()
    if fault == "visual_lookup_timeout":
        rig.checkpoint("candidate check failed", "not_found", 0, False)
        rig.check("candidate image checked", rig.verifier.calls, 1)
        return
    rig.client.accept()
    rig.finish()
    rig.checkpoint("pose reached", "awaiting_observation", 1, True)
    rig.advance(0.1)
    observation = Observation("fault-robot", "front", rig.stamp, rig.pose, evidence, localization_checked=True)
    if fault == "arrival_stale_camera":
        rig.commands.observe(replace(observation, timestamp=1000.0))
    elif fault == "arrival_untrusted_pose":
        # Exercise the capture-pose admission check used by the ROS node. A transform
        # inconsistent with current localization cannot supply trusted arrival evidence.
        accepted = rig.gate.accepts(Pose(9, 2, map_id="office"), rig.stamp)
        rig.check("inconsistent capture transform rejected", accepted, False)
        rig.commands.observe(replace(observation, localization_checked=accepted))
    elif fault == "arrival_verifier_timeout":
        rig.verifier.error = TimeoutError("injected arrival verifier timeout")
        rig.commands.observe(observation)
        rig.drain()
    if fault != "arrival_verifier_timeout":
        rig.checkpoint("no trusted fresh image", "awaiting_observation", 1, True)
        rig.advance(3)
        rig.commands.poll()
    rig.checkpoint("arrival not confirmed", "destination_unverified", 1, False)
    rig.check("no mission success", any(e["state"] in {"succeeded", "step_succeeded"} for e in rig.events), False)


def _target_fault(rig: _Rig, fault: str) -> None:
    evidence = rig.memory_destination()
    # Isolate the capture freshness limit from the longer time allowed to obtain a view.
    rig.commands._arrival_timeout = 30

    def delete_selected() -> None:
        destination = rig.commands._destination
        assert destination is not None and destination.memory is not None
        rig.store.delete([destination.memory.id])

    if fault == "target_changed_before_send":
        publish = rig.commands._publish_callback

        def changed(update: NavigationUpdate) -> None:
            publish(update)
            if update.state == "submitting":
                delete_selected()

        rig.commands._publish_callback = changed
    rig.start()
    if fault == "target_changed_before_send":
        rig.checkpoint("changed reference blocks dispatch", "not_found", 0, False)
        rig.check("failure stage", rig.events[-1]["failure_stage"], "retrieval")
        return
    rig.client.accept()
    rig.finish(6 if fault == "target_navigation_abort" else 4)
    if fault == "target_navigation_abort":
        rig.checkpoint("transport failure", "failed", 1, False)
        rig.check("failure stage", rig.events[-1]["failure_stage"], "execution")
        return
    rig.advance(0.1)
    observation = Observation("fault-robot", "front", rig.stamp, rig.pose, evidence, localization_checked=True)
    stage = "geometry"

    def age_image() -> None:
        for _ in range(6):
            rig.advance(1)  # Keep localization healthy while the capture itself expires.

    if fault == "target_deleted_at_arrival":
        delete_selected()
        stage = "retrieval"
    elif fault == "target_deleted_during_verdict":
        rig.verifier.before = delete_selected
        stage = "retrieval"
    elif fault == "target_late_verdict":
        rig.verifier.before = age_image
    elif fault == "target_post_arrival_stale_image":
        age_image()
    elif fault in {"target_visual_mismatch", "target_visual_uncertain"}:
        rig.verifier.result = "not_matched" if fault.endswith("mismatch") else "uncertain"
        stage = "identity"
    calls = rig.verifier.calls
    rig.commands.observe(observation)
    if fault == "target_queued_image_expired":
        age_image()
    rig.drain()
    if fault == "target_post_arrival_stale_image":
        rig.checkpoint("old capture refused", "awaiting_observation", 1, True)
        rig.commands._arrival_deadline = rig.elapsed
        rig.commands.poll()
    if fault in {"target_post_arrival_stale_image", "target_queued_image_expired", "target_deleted_at_arrival"}:
        rig.check("no model call on unusable evidence", rig.verifier.calls, calls)
    rig.checkpoint("arrival remains unverified", "destination_unverified", 1, False)
    rig.check("failure stage", rig.events[-1]["failure_stage"], stage)
    rig.check("no next destination", bool(rig.tasks), False)
    rig.check("no false success", any(e["state"] in {"succeeded", "step_succeeded"} for e in rig.events), False)


def _storage_fault(rig: _Rig, fault: str) -> None:
    if fault == "storage_read":
        rig.context.fail_read = True
    else:
        rig.context.fail_kind = "instruction" if fault == "storage_instruction" else "status"
        rig.context.fail_state = {
            "storage_dispatch": "submitting",
            "storage_step": "step_succeeded",
            "storage_motion": "navigating",
        }.get(fault, "")
    rig.start()
    if fault in {"storage_instruction", "storage_read", "storage_dispatch"}:
        rig.checkpoint("storage blocked mission", "rejected" if fault == "storage_read" else "unavailable", 0, False)
    else:
        rig.client.accept()
        if fault == "storage_motion":
            rig.commands.poll()
            rig.checkpoint("write failure during motion", "canceling", 1, True)
            rig.finish(5)
            rig.checkpoint("stop confirmed", "canceled", 1, False)
        else:
            rig.finish()
            rig.drain()
            rig.checkpoint("next step blocked", "unavailable", 1, False)
    rig.check("storage fault exercised", rig.context.faults > 0, True)


def _depth_fault(rig: _Rig, fault: str) -> None:
    pending = PendingImages()
    message = SimpleNamespace(header=SimpleNamespace(frame_id="front", stamp=SimpleNamespace(sec=1000, nanosec=0)))
    pending.add(message, False, 0)
    if fault == "depth_stale":
        pending.depth.append(
            SimpleNamespace(header=SimpleNamespace(frame_id="front", stamp=SimpleNamespace(sec=999, nanosec=0)))
        )
        pending.info.append(message)
    rig.check("wait for aligned depth", pending.pop(0.1) is None, True)
    rig.check("bounded scene-only fallback", pending.pop(0.3) == (message, False), True)
    rig.check("RGB delivered only once", pending.pop(0.4) is None, True)
    rig.checkpoint("synchronizer only; no mission requested", "idle", 0, False)


def _provenance_fault(rig: _Rig, fault: str) -> None:
    sensor = SensorHealth(clock=lambda: rig.stamp, monotonic=lambda: rig.elapsed)
    camera = "camera" in fault or "depth" in fault
    if camera:
        rig.memory_destination()
        rig.commands._sensor_ready = lambda _: sensor.ready(camera=True, depth="depth" in fault)
        rig.commands._sensor_generation = lambda _: sensor.generation(depth="depth" in fault)
    rig.commands._localization_check = lambda: sensor.ready() and rig.gate.ready()
    if "before" in fault:
        if not camera:
            rig.gate.invalidate()
        rig.start()
        rig.checkpoint("invalid provenance blocks dispatch", "unavailable", 0, False)
        return
    sensor.observe(rig.stamp, camera=True, depth=True)
    rig.start()
    rig.client.accept()
    if "clock_reset" in fault:
        sensor.clock_changed.set()
        rig.stamp -= 900
        rig.refresh_localization()
    elif "paused" in fault:
        rig.elapsed += 6
    elif "forward" in fault:
        rig.stamp += 6
    elif camera:
        if "depth" in fault:
            for _ in range(6):
                rig.advance(1)
                sensor.observe(rig.stamp, camera=True, depth=False)
        else:
            rig.advance(0.1)
        sensor.observe(rig.stamp, camera="depth" in fault, depth=False)
        if "recovered" in fault:
            rig.advance(0.1)
            sensor.observe(rig.stamp, camera=True, depth=True)
    else:
        rig.gate.invalidate()
        rig.advance(0.1)
    if "result" not in fault:
        rig.commands.poll()
        rig.checkpoint("provenance loss requests cancellation", "canceling", 1, True)
        rig.check("one cancellation request", rig.client.handles[0].cancel_calls, 1)
    rig.finish()
    rig.checkpoint(
        "late success cannot confirm or continue", "destination_unverified" if camera else "canceled", 1, False
    )
    rig.check("no next mission work", len(rig.tasks), 0)
    rig.check("no successful step", any(e["state"] in {"succeeded", "step_succeeded"} for e in rig.events), False)


# A child exits without connection cleanup midway through a real SQLite transaction.
# No parent process or robot process is killed. This is not a power-loss test.
_INTERRUPTED_WRITE = """
import os, sqlite3, sys
from placecell.mission_context import MissionContext
context = MissionContext(sys.argv[1], scope='fault-robot:office:operator')
context.record('old', 'instruction', {'text': 'Visit printer then cupboard'})
context.record('old', 'status', {'state': 'navigating', 'step': 1})
connection = sqlite3.connect(sys.argv[1])
connection.execute('BEGIN IMMEDIATE')
connection.execute("INSERT INTO mission_events(scope, request_id, kind, payload, timestamp) VALUES (?, ?, ?, ?, ?)",
                   ('fault-robot:office:operator', 'uncommitted', 'status', '{"state":"succeeded"}', 1000))
os._exit(17)
"""


def _interrupted_storage(rig: _Rig, fault: str) -> None:
    rig.context.close()
    result = subprocess.run(  # noqa: S603 -- fixed child code, current interpreter, isolated temporary database
        [sys.executable, "-c", _INTERRUPTED_WRITE, str(rig.directory / "context.sqlite3")],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    rig.context = _Context(rig.directory / "context.sqlite3")
    rig.check("child reached interruption", result.returncode, 17)
    history = rig.context.recent()
    rig.check("committed history survives", [event["data"].get("state") for event in history], [None, "navigating"])
    rig.check("uncommitted success absent", any(event["request_id"] == "uncommitted" for event in history), False)
    rig.commands = rig.new_controller()
    rig.commands.poll()
    rig.drain()
    rig.checkpoint("restart does not replay motion", "idle", 0, False)
    rig.check("restart makes no model call", rig.model.calls + rig.reviewer.calls, 0)


def _control(rig: _Rig, fault: str) -> None:
    if fault == "control_visual_arrival":
        evidence = rig.memory_destination()
        rig.model.reply = ChatReply(
            None,
            (
                ToolCall(
                    "fixture",
                    "propose_navigation_plan",
                    {
                        "decision": "ready",
                        "destinations": ["printer"],
                        "message": "Scripted single visual destination.",
                    },
                ),
            ),
        )
        rig.start()
        rig.client.accept()
        rig.finish()
        rig.checkpoint("wait for camera", "awaiting_observation", 1, True)
        rig.advance(0.1)
        rig.commands.observe(
            Observation("fault-robot", "front", rig.stamp, rig.pose, evidence, localization_checked=True)
        )
        rig.drain()
        rig.checkpoint("fresh image checked", "succeeded", 1, False)
        rig.check("candidate and arrival checks", rig.verifier.calls, 2)
        return
    rig.start()
    for _ in range(2):
        rig.client.accept()
        rig.finish()
        rig.drain()
    rig.checkpoint("normal ordered completion", "succeeded", 2, False)
    rig.check("requested goal order", [pose.x for pose in rig.client.goals], [1, 8])
    rig.check("planner and review called", [rig.model.calls, rig.reviewer.calls], [1, 1])


@dataclass(frozen=True)
class FaultCase:
    id: str
    scope: str
    fault: str
    exercise: Callable[[_Rig, str], None]


CASES = (
    FaultCase("control_ordered_mission", "mission", "No fault: two named places complete in order", _control),
    FaultCase(
        "control_visual_arrival", "mission", "No fault: scripted candidate and fresh arrival verification", _control
    ),
    *(
        FaultCase(name, "provider_contract", description, _provider_contract_fault)
        for name, description in (
            ("planner_duplicate", "Conflicting duplicate decision fields in planner tool JSON"),
            ("review_duplicate", "Conflicting duplicate decision fields in reviewer tool JSON"),
            ("planner_extra_field", "Planner adds an unsupported action field"),
            ("planner_unknown_tool", "Planner attempts a direct drive tool"),
            ("planner_oversized", "Planner exceeds the explanation character limit"),
            ("review_oversized", "Reviewer exceeds the explanation character limit"),
            ("planner_missing_finish", "Planner omits completion evidence"),
            ("planner_refusal", "Provider refusal accompanies an executable-looking plan"),
            ("review_http_error", "Reviewer HTTP 503 is not retried or converted into approval"),
            ("visual_duplicate", "Conflicting duplicate visual verdicts"),
            ("visual_extra_field", "Visual verdict includes an unsupported action"),
            ("visual_missing_finish", "Visual verification omits completion evidence"),
            ("visual_refusal", "Visual provider refusal accompanies matched JSON"),
            ("visual_http_error", "Visual HTTP 503 cannot authorize a destination"),
        )
    ),
    *(
        FaultCase(name, "mission", description, _model_fault)
        for name, description in (
            ("planner_timeout", "Planning provider raises TimeoutError"),
            ("planner_malformed", "Planning provider returns no structured tool call"),
            ("planner_late", "Planning reply arrives after the lookup deadline is polled"),
            ("planner_stop", "Stop during planning; late model reply cannot dispatch"),
            ("review_timeout", "Review provider raises TimeoutError"),
            ("review_late", "Review reply arrives after the lookup deadline is polled"),
            ("review_malformed", "Review provider returns no structured tool call"),
            ("review_reject", "Independent reviewer rejects a valid plan"),
            ("review_stop", "Stop arrives during plan review"),
        )
    ),
    *(
        FaultCase(name, "mission_callback_boundary", description, _cancellation_boundary)
        for name, description in (
            ("lookup_stop", "Stop during candidate image verification"),
            ("arrival_wait_stop", "Stop while waiting for a fresh arrival image"),
            ("arrival_verification_stop", "Stop during arrival verification; late verdict is discarded"),
            ("nav_stop_before_acceptance", "Stop before a goal handle exists; late acceptance is canceled"),
            ("nav_unpolled_late_acceptance", "Goal response expires while poll is delayed"),
            ("nav_unpolled_late_success", "Trip deadline expires before a result without an intervening poll"),
            ("nav_timeout_result_race", "Successful result beats delivery of the timeout cancellation event"),
            ("nav_timeout_visual_result_race", "Visual-goal success beats the timeout event; no arrival check starts"),
            ("nav_response_result_race", "Late acceptance has an already-complete success before timeout delivery"),
        )
    ),
    *(
        FaultCase(name, "mission", description, _navigation_fault)
        for name, description in (
            ("nav_unavailable", "Action server is unavailable before dispatch"),
            ("nav_rejected", "Action server rejects the submitted goal"),
            ("nav_send_error", "Transport raises after a goal submission attempt"),
            ("nav_late_acceptance", "Goal is accepted after its response deadline"),
            ("nav_late_acceptance_success", "Late-accepted goal reports success after cancellation was requested"),
            ("nav_lost_result", "Trip expires; cancel acknowledged but terminal result never arrives"),
            ("nav_timeout_late_success", "Trip expires; a successful terminal result arrives after cancellation"),
            (
                "nav_timeout_visual_success",
                "Remembered-pose trip expires, then reports success without arrival evidence",
            ),
            ("nav_cancel_delayed", "Cancel acknowledgement is delayed past the response deadline"),
            ("nav_cancel_rejected", "Action server refuses cancellation"),
        )
    ),
    *(
        FaultCase(name, "mission", description, _localization_fault)
        for name, description in (
            ("localization_stale_before", "Localization ages out before instruction admission"),
            ("localization_stale_motion", "Localization ages out during navigation"),
            ("localization_paused_clock", "Source time pauses while localization receipt age expires"),
        )
    ),
    *(
        FaultCase(name, "mission", description, _arrival_fault)
        for name, description in (
            ("visual_lookup_timeout", "Candidate image verifier raises TimeoutError"),
            ("arrival_missing_camera", "No camera image arrives after reaching a remembered pose"),
            ("arrival_stale_camera", "Only a pre-arrival camera image arrives"),
            ("arrival_untrusted_pose", "Fresh image has no validated capture pose (TF/localization boundary)"),
            ("arrival_verifier_timeout", "Fresh arrival image verifier raises TimeoutError"),
        )
    ),
    *(
        FaultCase(name, "mission", description, _storage_fault)
        for name, description in (
            ("storage_instruction", "Context instruction write fails"),
            ("storage_read", "Context history read fails before planning"),
            ("storage_dispatch", "Context status write fails immediately before dispatch"),
            ("storage_motion", "Context status write fails when navigation is accepted"),
            ("storage_step", "Context status write fails after the first completed step"),
        )
    ),
    *(
        FaultCase(name, "target_freshness", description, _target_fault)
        for name, description in (
            ("target_changed_before_send", "Selected reference changes during submission status persistence"),
            ("target_deleted_at_arrival", "Selected memory is removed before the arrival image"),
            ("target_deleted_during_verdict", "Selected memory is removed during visual verification"),
            ("target_late_verdict", "Positive verdict arrives after the capture freshness bound"),
            ("target_post_arrival_stale_image", "Post-arrival image is already too old when delivered"),
            ("target_queued_image_expired", "Accepted image expires while waiting for the worker"),
            ("target_visual_mismatch", "Fresh image does not match the requested target"),
            ("target_visual_uncertain", "Fresh target identity remains uncertain"),
            ("target_navigation_abort", "Transport aborts a trip to the selected target"),
        )
    ),
    FaultCase("depth_missing", "sensor_boundary", "RGB arrives without depth or calibration", _depth_fault),
    FaultCase("depth_stale", "sensor_boundary", "Only out-of-skew depth accompanies RGB", _depth_fault),
    *(
        FaultCase(name, "mission", description, _provenance_fault)
        for name, description in (
            ("provenance_localization_before", "Missing localization blocks command admission"),
            ("provenance_camera_before", "Missing live camera blocks a remembered destination"),
            ("provenance_depth_before", "Missing aligned depth blocks depth-dependent dispatch"),
            ("provenance_localization_recovered", "Localization recovers before polling; old mission still cancels"),
            ("provenance_localization_result", "Named-goal success overtakes polling after localization recovers"),
            ("provenance_clock_reset", "Reset clock and new localization cannot resume a mission"),
            ("provenance_clock_reset_result", "Success overtakes polling after a clock reset"),
            ("provenance_paused_result", "Success arrives after receipt age expires on a paused clock"),
            ("provenance_forward_result", "Source clock jumps beyond localization freshness before success"),
            ("provenance_camera_motion", "Camera failure during motion requests cancellation"),
            ("provenance_camera_recovered_result", "Camera recovers before success; old trust generation cannot pass"),
            ("provenance_depth_motion", "Aligned depth fails during motion and requests cancellation"),
            ("provenance_depth_recovered_result", "Depth recovers before success; old trust generation cannot pass"),
        )
    ),
    FaultCase(
        "storage_interrupted",
        "persistence_restart",
        "Child exits with an uncommitted SQLite write",
        _interrupted_storage,
    ),
)


def run_faults(*, cases: Sequence[str] = (), repeat: int = 1) -> dict[str, Any]:
    """Run selected contracts; report every repetition including failed scenarios."""
    known = {case.id for case in CASES}
    if type(repeat) is not int or not 1 <= repeat <= 1000 or set(cases) - known or len(set(cases)) != len(cases):
        raise ValidationError("select unique known case IDs and a repeat count within 1..1000")
    selected = [case for case in CASES if not cases or case.id in cases]
    results = []
    started = time.perf_counter()
    for iteration in range(1, repeat + 1):
        for case in selected:
            tick = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="placecell-fault-") as directory:
                rig = _Rig(Path(directory))
                error = None
                try:
                    case.exercise(rig, case.id)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                try:
                    flushed = rig.traces.flush()
                    trace = read_trace(rig.traces.path)
                    if rig.events:
                        rig.check("trace writer flushed", flushed, True)
                        rig.check("trace events captured", bool(trace["events"]), True)
                        rig.check("trace reports last status", trace["summary"]["last_status"]["state"], rig.state)
                        rig.check(
                            "trace preserves every published status",
                            [event["data"]["state"] for event in trace["events"] if event["stage"] == "status"],
                            [event["state"] for event in rig.events],
                        )
                        rig.check(
                            "trace preserves failure attribution",
                            [event["data"]["failure_stage"] for event in trace["events"] if event["stage"] == "status"],
                            [event["failure_stage"] for event in rig.events],
                        )
                        rig.check(
                            "trace dispatch attempts",
                            sum(event["stage"] == "nav2.dispatch" for event in trace["events"]),
                            len(rig.client.goals),
                        )
                        rig.check(
                            "trace cancel requests",
                            sum(event["stage"] == "nav2.cancel_requested" for event in trace["events"]),
                            sum(handle.cancel_calls for handle in rig.client.handles),
                        )
                        rig.check(
                            "trace event loss",
                            sum(
                                rig.traces.health()[key]
                                for key in ("dropped_events", "write_errors", "trimmed_events", "truncated_events")
                            ),
                            0,
                        )
                    result = {
                        "case_id": case.id,
                        "iteration": iteration,
                        "scope": case.scope,
                        "fault": case.fault,
                        "passed": error is None and bool(rig.checks) and all(check["passed"] for check in rig.checks),
                        "error": error,
                        "checks": rig.checks,
                        "events": rig.events,
                        "final_state": rig.state,
                        "failure_stage": rig.events[-1]["failure_stage"] if rig.events else "",
                        "mission_owned": rig.commands.busy,
                        "queued_tasks": len(rig.tasks),
                        "dispatch_attempts": [asdict(pose) for pose in rig.client.goals],
                        "cancel_requests": sum(handle.cancel_calls for handle in rig.client.handles),
                        "scripted_model_calls": rig.model.calls + rig.reviewer.calls,
                        "scripted_visual_calls": rig.verifier.calls,
                        "injected_storage_errors": rig.context.faults,
                        "trace": trace,
                        "simulated_elapsed_s": rig.elapsed,
                    }
                finally:
                    rig.close()
            result["wall_duration_ms"] = (time.perf_counter() - tick) * 1000
            results.append(result)
    sources = {
        str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.rglob("*.py"))
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": "placecell-offline-faults-v1",
        "runner": "scripted_fault_injection",
        "python_version": sys.version.split()[0],
        "sources_sha256": sources,
        "configuration": {
            "repeat": repeat,
            "case_ids": [case.id for case in selected],
            "lookup_timeout_s": 5,
            "arrival_timeout_s": 3,
            "target_case_arrival_timeout_s": 30,
            "max_observation_age_s": 5,
            "nav_response_timeout_s": 2,
            "nav_trip_timeout_s": 10,
        },
        "limitations": [
            "Scripted providers and action transport; no live ROS, Gazebo or physical robot",
            "Virtual deadlines are not physical stop latency or wall-clock provider timeouts",
            "Depth tests cover synchronization only; TF tests cover capture-trust admission, not a TF buffer",
            "Process interruption covers SQLite rollback and no automatic replay, not robot restart reconciliation",
            "Synthetic fixtures do not measure model or visual recognition accuracy",
        ],
        "paid_api_calls": 0,
        "cost_usd": 0,
        "summary": {
            "runs": len(results),
            "passed": sum(result["passed"] for result in results),
            "failed": sum(not result["passed"] for result in results),
            "final_failure_stages": {
                stage: sum(result["failure_stage"] == stage for result in results)
                for stage in ("", "retrieval", "identity", "geometry", "execution")
            },
            "wall_duration_ms": (time.perf_counter() - started) * 1000,
        },
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path (never overwritten)")
    parser.add_argument("--case", action="append", default=[], choices=[case.id for case in CASES])
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.repeat <= 1000:
        parser.error("--repeat must be within 1..1000")
    if len(set(args.case)) != len(args.case):
        parser.error("--case IDs must be unique")
    if args.output.exists():
        parser.error("output already exists; select a new report path")
    report = run_faults(cases=args.case, repeat=args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    sys.stdout.write(json.dumps(report["summary"]) + "\n")
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
