"""Chat types shared by the reasoning agent and the chat backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn of a chat. `tool_calls` on assistant turns, `tool_call_id` on tool turns."""

    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ChatReply:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()


@runtime_checkable
class ChatModel(Protocol):
    """Contract for the reasoning backend: messages and tool schemas in, one reply out."""

    def complete(self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]) -> ChatReply: ...
