"""ROS operator topics and read-only snapshot service."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict
from typing import Any

from placecell.command_identity import CommandJournal, CommandReceipt, IdentifiedCommand
from placecell.errors import ValidationError
from placecell.navigation import NavigationCommands, NavigationSnapshot, NavigationUpdate
from placecell.operator import navigation_payload, parse_operator_command, snapshot_payload


class OperatorInterface:
    """One node instance, with volatile commands and a retained, periodically refreshed snapshot."""

    def __init__(
        self, node: Any, commands: NavigationCommands | None, *, journal: CommandJournal | None = None
    ) -> None:
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.clock import Clock, ClockType
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self._message = String
        self._commands = commands
        self._journal = journal
        self._disabled_lock = threading.RLock()
        self._disabled_sequence = 0
        self._disabled_status = NavigationUpdate(
            "", "disabled", "Set navigation_enabled to use movement commands.", instance_id=uuid.uuid4().hex
        )
        self._status = node.create_publisher(String, "~/navigation_status", 10)
        self._receipts = node.create_publisher(String, "~/command_receipt", 10)
        retained = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
        )
        volatile = QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE, reliability=ReliabilityPolicy.RELIABLE)
        self._snapshot = node.create_publisher(String, "~/mission_snapshot", retained)
        self._command_group = MutuallyExclusiveCallbackGroup()
        # Keep the legacy stop path schedulable during a slow command-journal write.
        self._json_group = MutuallyExclusiveCallbackGroup()
        self._text_subscription = node.create_subscription(
            String, "~/command", self._on_text, volatile, callback_group=self._command_group
        )
        self._json_subscription = node.create_subscription(
            String, "~/command_json", self._on_json, volatile, callback_group=self._json_group
        )
        self._service = node.create_service(Trigger, "~/get_mission_snapshot", self._on_snapshot)
        # Refresh even while simulated ROS time is paused. Snapshot reads do not poll the controller.
        self._timer = node.create_timer(0.2, self.publish_snapshot, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.publish_snapshot()

    def publish(self, update: NavigationUpdate) -> None:
        self._status.publish(self._message(data=navigation_payload(update)))

    def _disabled_event(self, state: str, message: str) -> None:
        with self._disabled_lock:
            self._disabled_sequence += 1
            self.publish(
                NavigationUpdate(
                    uuid.uuid4().hex,
                    state,
                    message,
                    instance_id=self._disabled_status.instance_id,
                    sequence=self._disabled_sequence,
                )
            )

    def _on_text(self, msg: Any) -> None:
        if self._commands is None:
            self._disabled_event("disabled", self._disabled_status.message)
        else:
            self._commands.handle(msg.data)

    def _on_json(self, msg: Any) -> None:
        try:
            text = parse_operator_command(msg.data)
        except ValidationError as e:
            if self._commands is None:
                self._disabled_event("invalid", str(e))
            else:
                self._commands.reject_command(str(e))
            return
        if isinstance(text, IdentifiedCommand):
            self._on_identified(text)
        else:
            self._on_text(self._message(data=text))

    def _on_identified(self, command: IdentifiedCommand) -> None:
        epoch = self._commands.admission_epoch if self._commands is not None else None
        if self._commands is None:
            receipt = CommandReceipt("disabled")
        elif self._journal is None:
            receipt = CommandReceipt("unavailable")
        else:
            try:
                receipt = self._journal.claim(command)
            except (sqlite3.Error, OSError, ValidationError):
                receipt = CommandReceipt("unavailable")
        # Publish the committed reservation before routing. Receipt loss is recovered by
        # resending the identical envelope; a crash here must never replay the instruction.
        self._receipts.publish(
            self._message(
                data=json.dumps(
                    {
                        "schema_version": 1,
                        "type": "command_receipt",
                        "command_id": command.command_id,
                        "scope": asdict(command.scope),
                        "request_id": receipt.request_id or None,
                        "instance_id": (
                            self._commands.snapshot().status.instance_id
                            if self._commands is not None
                            else self._disabled_status.instance_id
                        ),
                        "disposition": receipt.disposition,
                    },
                    allow_nan=False,
                )
            )
        )
        if receipt.disposition == "recorded" and self._commands is not None:
            self._commands.handle(
                command.text,
                request_id=receipt.request_id,
                target_request_id=command.target_request_id,
                admission_epoch=epoch,
            )

    def _snapshot_payload(self) -> str:
        if self._commands is not None:
            payload = snapshot_payload(self._commands.snapshot())
        else:
            with self._disabled_lock:
                payload = snapshot_payload(
                    NavigationSnapshot(self._disabled_status, self._disabled_sequence, False, False, None, None),
                    navigation_enabled=False,
                )
        value = json.loads(payload)
        value["command_identity"] = (
            {
                "schema_version": 2,
                "scope": asdict(self._journal.scope),
                "retry_window_s": self._journal.retry_window_s,
                "max_records": self._journal.max_records,
                "durable": self._journal.durable,
            }
            if self._journal is not None
            else None
        )
        return json.dumps(value, allow_nan=False)

    def publish_snapshot(self) -> None:
        self._snapshot.publish(self._message(data=self._snapshot_payload()))

    def _on_snapshot(self, request: Any, response: Any) -> Any:
        response.success = True
        response.message = self._snapshot_payload()
        return response
