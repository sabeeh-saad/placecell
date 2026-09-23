"""Bounded JSON and completion validation shared by model decision adapters."""

from __future__ import annotations

import json
import math
from typing import Any

from placecell.errors import ProviderError


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise ValueError("non-finite JSON number")


def _depth(value: Any, remaining: int = 32) -> None:
    if remaining < 0:
        raise ValueError("JSON nesting exceeds 32 levels")
    if isinstance(value, dict):
        for item in value.values():
            _depth(item, remaining - 1)
    elif isinstance(value, list):
        for item in value:
            _depth(item, remaining - 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")


def strict_json(text: str, *, max_chars: int = 65536) -> Any:
    """Raise ValueError for duplicate keys, non-finite values, deep or oversized input."""
    if not isinstance(text, str) or not 1 <= len(text) <= max_chars:
        raise ValueError("JSON text exceeds its size limit or is empty")
    try:
        value = json.loads(text, object_pairs_hook=_unique, parse_constant=_constant)
        _depth(value)
        return value
    except RecursionError as e:
        raise ValueError("JSON nesting exceeds its limit") from e


def bounded_json(value: Any, *, max_chars: int) -> str:
    """Validate injected/provider objects and bound serialized model task data."""
    try:
        _depth(value)
        text = json.dumps(value, allow_nan=False)
    except (TypeError, RecursionError) as e:
        raise ValueError("invalid JSON value") from e
    if len(text) > max_chars:
        raise ValueError("JSON value exceeds its size limit")
    return text


def completion_message(body: Any, *, tools: bool = False) -> dict[str, Any]:
    """One completed assistant response; refusals and unsupported actions fail closed."""
    try:
        choices = body["choices"]
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("expected exactly one completion")
        choice = choices[0]
        allowed = ("stop", "tool_calls") if tools else ("stop",)
        if choice.get("finish_reason") not in allowed:
            raise ProviderError("chat completion was interrupted, truncated or missing its finish reason")
        msg = choice["message"]
        if not isinstance(msg, dict) or msg.get("role", "assistant") != "assistant":
            raise ValueError("expected an assistant message")
        if msg.get("refusal") not in (None, "") or msg.get("function_call") is not None:
            raise ValueError("refused response or unsupported legacy function call")
        if not tools and msg.get("tool_calls") not in (None, []):
            raise ValueError("unexpected tool calls in a text decision")
        return msg
    except (KeyError, TypeError, ValueError) as e:
        raise ProviderError(f"malformed chat completion: {e}") from e


def completion_text(msg: dict[str, Any], *, max_chars: int = 65536, nullable: bool = False) -> str | None:
    content = msg.get("content")
    if nullable and content is None:
        return None
    if isinstance(content, list):
        if len(content) > 64 or any(
            not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str) for p in content
        ):
            raise ProviderError("malformed chat completion content parts")
        if sum(len(p["text"]) for p in content) > max_chars:
            raise ProviderError("chat completion content exceeds its size limit")
        content = " ".join(p["text"] for p in content)
    if not isinstance(content, str) or len(content) > max_chars:
        raise ProviderError("malformed or oversized chat completion content")
    return content
