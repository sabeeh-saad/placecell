"""Deterministic mission executions for command admission, chains and retained context.

Only transport, provider replies and the receipt publisher are controlled fixtures.
The operator routing, journal, planner/reviewer, controller and Nav2 adapter are real.
"""

from __future__ import annotations

import json
from concurrent.futures import Future
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from placecell.chat import ChatReply, ToolCall
from placecell.command_identity import CommandJournal, CommandScope
from placecell.pipeline import Observation
from placecell.ros2.operator import OperatorInterface

if TYPE_CHECKING:
    from placecell.fault_injection import _Rig


def _plan(rig: _Rig, destinations: list[str]) -> None:
    rig.model.reply = ChatReply(
        None,
        (
            ToolCall(
                "execution",
                "propose_navigation_plan",
                {"decision": "ready", "destinations": destinations, "message": "Scripted execution plan."},
            ),
        ),
    )


def _chain(rig: _Rig, case: str) -> None:
    _plan(rig, ["printer", "cupboard", "printer"])
    rig.start()
    rig.client.accept()
    if case == "chain_next_worker_full":
        rig.queue_ready = False
    rig.finish()
    if case == "chain_stop_between_steps":
        rig.commands.handle("stop")
        rig.drain()
        rig.checkpoint("queued second step discarded", "canceled", 1, False)
    elif case == "chain_next_worker_full":
        rig.checkpoint("queue refusal stops chain", "failed", 1, False)
    else:
        rig.drain()
        rig.client.accept(case != "chain_middle_rejected")
        if case == "chain_middle_rejected":
            rig.checkpoint("rejected middle step stops chain", "rejected", 2, False)
        elif case == "chain_middle_aborted":
            rig.finish(6)
            rig.drain()
            rig.checkpoint("aborted middle step stops chain", "failed", 2, False)
        else:
            rig.finish()
            rig.drain()
            rig.client.accept()
            rig.finish()
            rig.checkpoint("all repeated visits complete", "succeeded", 3, False)
            rig.check("order includes deliberate return", [p.x for p in rig.client.goals], [1, 8, 1])
    rig.check("review runs exactly once for the chain", rig.reviewer.calls, 1)
    rig.check("no abandoned next-step work", len(rig.tasks), 0)


def _work_boundary(rig: _Rig, case: str) -> None:
    if case == "admission_worker_full":
        rig.queue_ready = False
        rig.start()
        rig.checkpoint("full worker refuses admission", "busy", 0, False)
        rig.check("no planning while worker full", rig.model.calls, 0)
        return
    if case == "admission_busy_planning":
        rig.commands.handle("Visit printer then cupboard")
        rig.commands.handle("Visit cupboard instead")
        rig.checkpoint("second instruction refused during planning", "busy", 0, True)
        rig.drain()
        rig.checkpoint("original instruction dispatched once", "submitting", 1, True)
        rig.check("only original instruction planned", rig.model.calls, 1)
        return
    _plan(rig, ["printer"])
    rig.start()
    rig.client.accept()
    old_trip = rig.navigator._trip
    old_feedback = rig.client.feedback[-1]
    rig.finish()
    _plan(rig, ["cupboard"])
    rig.start()
    rig.client.accept()
    count = len(rig.events)
    if case == "old_feedback_new_mission":
        old_feedback(SimpleNamespace(feedback=SimpleNamespace(distance_remaining=0.0)))
    else:
        failed: Future[Any] = Future()
        failed.set_exception(OSError("late old result callback"))
        assert old_trip is not None
        rig.navigator._result(old_trip, failed)
    rig.check("old callback publishes no new status", len(rig.events), count)
    rig.checkpoint("replacement retains ownership", "navigating", 2, True)
    rig.check("replacement not canceled by old callback", rig.client.handles[-1].cancel_calls, 0)
    rig.finish()
    rig.checkpoint("replacement succeeds normally", "succeeded", 2, False)


def _mixed_chain(rig: _Rig, case: str) -> None:
    evidence = rig.memory_destination()
    rig.resolver._places["cupboard"] = replace(rig.pose, x=8)
    _plan(rig, ["cupboard", "printer", "cupboard"])
    rig.start()
    rig.client.accept()
    rig.finish()
    rig.drain()
    rig.client.accept()
    rig.finish()
    rig.checkpoint("middle memory waits for visual evidence", "awaiting_observation", 2, True)
    rig.advance(0.1)
    rig.commands.observe(Observation("fault-robot", "front", rig.stamp, rig.pose, evidence, localization_checked=True))
    rig.drain()
    rig.checkpoint("verified middle step permits final leg", "submitting", 3, True)
    rig.client.accept()
    rig.finish()
    rig.checkpoint("mixed completion", "succeeded", 3, False)
    rig.check("named-memory-named dispatch order", [p.x for p in rig.client.goals], [8, 1, 8])
    rig.check("candidate and fresh arrival verified", rig.verifier.calls, 2)


def _choice(rig: _Rig, case: str) -> None:
    evidence = rig.memory_destination()
    first = rig.store.query()[0]
    rig.store.upsert([replace(first, id="second-printer", pose=replace(first.pose, x=5))])
    rig.resolver._places["cupboard"] = replace(rig.pose, x=8)
    rig.start()
    rig.checkpoint("ambiguity prevents dispatch", "ambiguous", 0, True)
    selected = rig.commands.snapshot().status.choices[0]
    if case == "choice_stop":
        rig.commands.handle("stop")
        rig.commands.handle("option one")
        rig.drain()
        rig.checkpoint("stopped choice cannot restart", "invalid", 0, False)
        return
    if case == "choice_expired":
        for _ in range(12):
            rig.advance(0.5)  # Keep localization fresh while only the choice deadline expires.
        rig.commands.handle("option one")
        rig.drain()
        rig.checkpoint("expired choice refused", "invalid", 0, True)
        rig.commands.handle("stop")
        rig.checkpoint("operator can release expired choice", "canceled", 0, False)
        return
    if case == "choice_deleted":
        assert selected.memory is not None
        rig.store.delete([selected.memory.id])
    rig.commands.handle("option one")
    rig.drain()
    if case == "choice_deleted":
        rig.checkpoint("deleted choice cannot dispatch", "not_found", 0, False)
        return
    rig.client.accept()
    rig.finish()
    rig.checkpoint("selected memory requires arrival", "awaiting_observation", 1, True)
    rig.pose = selected.pose
    rig.advance(0.1)
    rig.commands.observe(Observation("fault-robot", "front", rig.stamp, rig.pose, evidence, localization_checked=True))
    rig.drain()
    rig.client.accept()
    rig.finish()
    rig.checkpoint("choice continues reviewed chain", "succeeded", 2, False)
    rig.check("selected target precedes cupboard", [p.x for p in rig.client.goals], [selected.pose.x, 8])
    rig.check("choice does not replan the mission", rig.model.calls, 1)


def _identity(rig: _Rig, case: str) -> None:
    scope = CommandScope("fault-robot", "office", "operator")
    path = rig.directory / "commands.sqlite3"
    journal = CommandJournal(path, scope, clock=lambda: rig.stamp)
    # Run production JSON parsing/admission/routing; only ROS construction and receipt delivery are stubbed.
    bridge = OperatorInterface.__new__(OperatorInterface)
    bridge._commands, bridge._journal = rig.commands, journal
    bridge._message = SimpleNamespace
    receipts: list[dict[str, Any]] = []
    bridge._receipts = SimpleNamespace(publish=lambda msg: receipts.append(json.loads(msg.data)))
    envelope = {
        "schema_version": 2,
        "command_id": "original",
        "scope": asdict(scope),
        "issued_at_unix_s": rig.stamp,
        "command": "instruction",
        "text": "Visit printer",
    }

    def send(data: dict[str, Any]) -> None:
        bridge._on_json(SimpleNamespace(data=json.dumps(data)))
        rig.drain()

    try:
        _plan(rig, ["printer"])
        if case == "stop_during_admission":
            claim = journal.claim

            def interrupted(command: Any) -> Any:
                receipt = claim(command)
                rig.commands.handle("stop")
                return receipt

            journal.claim = interrupted  # type: ignore[method-assign]
            send(envelope)
            rig.checkpoint("stop invalidates pending admission", "stale_command", 0, False)
            rig.check("reservation committed once", receipts[-1]["disposition"], "recorded")
            return
        send(envelope)
        rig.client.accept()
        first_request = receipts[0]["request_id"]
        if case in {"retry_completed", "retry_after_reopen"}:
            rig.finish()
        if case == "retry_after_reopen":
            journal.close()
            journal = CommandJournal(path, scope, clock=lambda: rig.stamp)
            bridge._journal = journal
        repeated = dict(envelope)
        disposition = "duplicate"
        if case == "conflicting_payload":
            repeated["text"] = "Visit cupboard"
            disposition = "conflict"
        elif case in {"scope_stale_stop", "targeted_stale_stop"}:
            repeated = {k: v for k, v in envelope.items() if k != "text"}
            repeated.update(command_id="stop", command="stop", target_request_id=first_request)
            if case == "scope_stale_stop":
                repeated["scope"] = asdict(replace(scope, map_id="previous-map"))
                disposition = "wrong_scope"
            else:
                repeated["target_request_id"] = "f" * 32
                disposition = "recorded"
        send(repeated)
        rig.check("admission receipt", receipts[-1]["disposition"], disposition)
        rig.check("exactly one dispatch after retry/refusal", len(rig.client.goals), 1)
        rig.check("no unintended cancel", rig.client.handles[0].cancel_calls, 0)
        rig.check("no duplicate planning", rig.model.calls, 1)
        if disposition == "duplicate":
            rig.check("retry refers to original request", receipts[-1]["request_id"], first_request)
        if case in {"retry_completed", "retry_after_reopen"}:
            rig.checkpoint("completed command not replayed", "succeeded", 1, False)
        else:
            rig.check("original mission still owned", rig.commands.busy, True)
            rig.finish()
            rig.checkpoint("original completes after refused command", "succeeded", 1, False)
    finally:
        journal.close()


def _context_boundary(rig: _Rig, case: str) -> None:
    if case == "context_capacity_admission":
        rig.context._max_events = 1
        rig.start()
        rig.checkpoint("context cannot retain instruction plus status", "unavailable", 0, False)
        rig.check("bounded context despite write refusal", rig.context.stats()["events"], 1)
        return
    rig.context._references_available = lambda data: data.get("memory_id") != "deleted"
    for identity in ("retained-printer", "deleted"):
        rig.context.record(identity, "instruction", {"text": "Visit " + identity})
        rig.context.record(identity, "status", {"state": "succeeded", "memory_id": identity})
    rig.model.reply = ChatReply(
        None,
        (
            ToolCall(
                "context",
                "propose_navigation_plan",
                {
                    "decision": "clarify",
                    "destinations": [],
                    "message": "Which destination? Earlier context is unavailable.",
                },
            ),
        ),
    )
    rig.commands.handle("Go there again")
    rig.drain()
    rig.checkpoint("scripted clarification cannot dispatch", "clarification_required", 0, False)
    task = json.loads(rig.model.inputs[-1][-1].content or "{}")
    context = task["recent_context"]
    rig.check("planner receives explicit boundary", context[0]["kind"], "retention_boundary")
    rig.check("older target excluded from planner input", "retained-printer" in json.dumps(context), False)


EXECUTION_CASES = (
    ("control_repeated_chain", "Three ordered visits include a deliberate return to the first place", _chain),
    ("chain_middle_rejected", "A rejected second leg prevents the third destination", _chain),
    ("chain_middle_aborted", "An aborted second leg prevents the third destination", _chain),
    ("chain_stop_between_steps", "Stop discards a queued next leg before it can resolve", _chain),
    ("chain_next_worker_full", "A full worker queue stops advancement after the first completed leg", _chain),
    ("admission_worker_full", "A full planning queue cannot create executable work", _work_boundary),
    ("admission_busy_planning", "A new instruction cannot replace one waiting for planning", _work_boundary),
    ("old_feedback_new_mission", "Feedback from a completed mission cannot alter its replacement", _work_boundary),
    ("old_result_new_mission", "A late old result exception cannot cancel a new mission", _work_boundary),
    ("control_mixed_chain", "A named-memory-named chain advances only after visual arrival", _mixed_chain),
    ("control_choice_chain", "A selected ambiguous memory completes before the next reviewed destination", _choice),
    ("choice_expired", "An expired ambiguity choice cannot dispatch and remains stoppable", _choice),
    ("choice_deleted", "A selected reference deleted during clarification cannot dispatch", _choice),
    ("choice_stop", "A choice after stop cannot restart the previous mission", _choice),
    ("retry_active", "An identical JSON command retry cannot duplicate an active goal", _identity),
    ("retry_completed", "An identical JSON command retry cannot replay a completed mission", _identity),
    ("conflicting_payload", "Reusing a command ID with another destination preserves the original mission", _identity),
    ("scope_stale_stop", "A stop from an old map scope cannot cancel the current mission", _identity),
    ("targeted_stale_stop", "A stop for a different request cannot cancel the current mission", _identity),
    ("retry_after_reopen", "Reopening the durable journal still suppresses a completed command retry", _identity),
    ("stop_during_admission", "Stop between reservation and routing prevents the reserved instruction", _identity),
    ("context_capacity_admission", "Context capacity refusal prevents a new goal", _context_boundary),
    (
        "context_deleted_followup",
        "Deleted latest context exposes a boundary instead of an older destination",
        _context_boundary,
    ),
)
