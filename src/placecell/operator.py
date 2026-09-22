"""Versioned operator wire contracts, independent of ROS message classes."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from placecell.command_identity import CommandScope, IdentifiedCommand
from placecell.errors import ValidationError
from placecell.navigation import Destination, NavigationSnapshot, NavigationUpdate, parse_movement

OPERATOR_SCHEMA_VERSION = 1


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValidationError("Command JSON must not contain duplicate fields.")
        result[name] = value
    return result


def parse_operator_command(payload: str) -> str | IdentifiedCommand:
    """Validate an envelope before routing it through the existing command controller."""
    if not isinstance(payload, str) or not 1 <= len(payload) <= 16384:
        raise ValidationError("Command JSON must contain 1..16384 characters.")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_fields)
    except (ValueError, RecursionError) as e:
        raise ValidationError("Command must be a valid JSON object with unique fields.") from e
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] not in (1, 2)
    ):
        raise ValidationError("Command schema_version must be the integer 1 or 2.")
    if value["schema_version"] == 2:
        return _identified(value)
    return _command_text(value)


def _command_text(value: dict[str, Any]) -> str:
    command = value.get("command")
    if command == "instruction" and set(value) == {"schema_version", "command", "text"}:
        text = value["text"]
        if isinstance(text, str) and text.strip() and len(text) <= 2000:
            return text
        raise ValidationError("Command text must contain 1..2000 characters and not be blank.")
    if command == "stop" and set(value) == {"schema_version", "command"}:
        return "stop"
    if command == "choose" and set(value) == {"schema_version", "command", "option"}:
        option = value["option"]
        if type(option) is int and 1 <= option <= 3:
            return f"option {option}"
        raise ValidationError("Command option must be an integer within 1..3.")
    raise ValidationError("Unknown command or fields. Use instruction/text, stop, or choose/option.")


def _identified(value: dict[str, Any]) -> IdentifiedCommand:
    value = value.copy()
    try:
        command_id = value.pop("command_id")
        scope = value.pop("scope")
        issued = value.pop("issued_at_unix_s")
    except KeyError as e:
        raise ValidationError("Version 2 requires command_id, scope and issued_at_unix_s.") from e
    if not isinstance(scope, dict) or set(scope) != {"robot_id", "map_id", "conversation_id"}:
        raise ValidationError("scope requires exactly robot_id, map_id and conversation_id.")
    target = value.pop("target_request_id", "") if value.get("command") in ("stop", "choose") else ""
    text = _command_text(value)
    if value["command"] == "instruction":
        try:
            movement = parse_movement(text)
        except ValidationError:
            pass  # Natural-language missions are validated by the planner/controller.
        else:
            if movement.kind in {"cancel", "choose"}:
                raise ValidationError("Version 2 stop/choice phrases require explicit stop/choose and a target.")
    return IdentifiedCommand(command_id, CommandScope(**scope), issued, value["command"], text, target)


def _describe(destination: Destination) -> dict[str, Any]:
    p = destination.pose
    return {
        "label": destination.label,
        "source": destination.source,
        "memory_id": destination.memory.id if destination.memory else None,
        "object_id": destination.object_id,
        "goal_kind": (
            "object_search"
            if destination.approach and destination.approach.region
            else "object_approach"
            if destination.approach
            else "destination"
        ),
        "target": destination.target,
        "x": p.x,
        "y": p.y,
        "yaw": p.yaw,
        "frame_id": p.frame_id,
        "map_id": p.map_id,
    }


def navigation_data(update: NavigationUpdate) -> dict[str, Any]:
    distance = update.distance_remaining
    return {
        "schema_version": OPERATOR_SCHEMA_VERSION,
        "type": "navigation_status",
        "instance_id": update.instance_id,
        "sequence": update.sequence,
        "request_id": update.request_id,
        "state": update.state,
        "message": update.message,
        "destination": _describe(update.destination) if update.destination else None,
        "choices": [{"option": i, **_describe(d)} for i, d in enumerate(update.choices, 1)],
        "distance_remaining": distance if distance is not None and math.isfinite(distance) else None,
        "object_result": update.object_result,
        "search_attempt": update.search_attempt,
        "mission_id": update.mission_id,
        "mission_step": update.mission_step,
        "mission_destinations": list(update.mission_destinations),
    }


def navigation_payload(update: NavigationUpdate) -> str:
    """Keep all legacy status fields, adding explicit version and event ordering."""
    return json.dumps(navigation_data(update), allow_nan=False)


def snapshot_payload(snapshot: NavigationSnapshot, *, navigation_enabled: bool = True) -> str:
    return json.dumps(
        {
            "schema_version": OPERATOR_SCHEMA_VERSION,
            "type": "mission_snapshot",
            "captured_at_unix_s": time.time(),
            "instance_id": snapshot.status.instance_id,
            "sequence": snapshot.sequence,
            "navigation_enabled": navigation_enabled,
            "busy": snapshot.busy,
            "closed": snapshot.closed,
            "active_request_id": snapshot.active_request_id,
            "awaiting_choice": snapshot.choice_remaining_s is not None and snapshot.choice_remaining_s > 0,
            "choice_remaining_s": snapshot.choice_remaining_s,
            "status": navigation_data(snapshot.status),
        },
        allow_nan=False,
    )
