"""Chat types shared by the reasoning agent and the chat backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from placecell.errors import ProviderError


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


class MalformedReplyError(ProviderError):
    """The model answered, but its tool call could not be decoded. Asking again may help."""


@runtime_checkable
class ChatModel(Protocol):
    """Contract for the reasoning backend: messages and tool schemas in, one reply out.

    `tool_choice` names the one offered tool the reply must call; None lets the model decide.
    """

    def complete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]], *, tool_choice: str | None = None
    ) -> ChatReply: ...
