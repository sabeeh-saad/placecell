"""Chat completions with function calling from any OpenAI-compatible server."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from placecell.chat import ChatMessage, ChatModel, ChatReply, ToolCall
from placecell.errors import ProviderError, ValidationError
from placecell.providers._contracts import bounded_json, completion_message, completion_text, strict_json
from placecell.providers._http import Endpoint, RetryPolicy, Transport


class OpenAICompatibleChat(ChatModel):
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
        timeout_s: float = 60.0,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not model:
            raise ValidationError("model must not be empty")
        if type(max_tokens) is not int or max_tokens < 1 or not math.isfinite(temperature) or temperature < 0:
            raise ValidationError("max_tokens must be positive and temperature non-negative")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._endpoint = Endpoint.build(
            base_url, "/chat/completions", api_key, timeout_s, transport, retry, sleep, extra_headers
        )

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]) -> ChatReply:
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [_encode(m) for m in messages],
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        return _decode(self._endpoint.post(payload))


def _encode(m: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in m.tool_calls
        ]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out


def _decode(body: Any) -> ChatReply:
    msg = completion_message(body, tools=True)
    content = completion_text(msg, nullable=True)
    try:
        raw_calls = msg.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list) or len(raw_calls) > 8:
            raise ValueError("tool calls must be an array of at most eight entries")
        calls = []
        ids: set[str] = set()
        for c in raw_calls:
            if not isinstance(c, dict) or c.get("type") != "function" or not isinstance(c.get("function"), dict):
                raise ValueError("unsupported tool call type")
            name, call_id = c["function"]["name"], c["id"]
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name)
                or not isinstance(call_id, str)
                or not call_id.strip()
                or len(call_id) > 128
                or call_id in ids
            ):
                raise ValueError("invalid or duplicate tool call identity")
            ids.add(call_id)
            arguments = c["function"]["arguments"]
            parsed = strict_json(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(parsed, dict):
                raise ValueError("tool arguments must be an object")
            bounded_json(parsed, max_chars=65536)
            calls.append(ToolCall(call_id, name, parsed))
    except (KeyError, TypeError, ValueError) as e:
        raise ProviderError(f"malformed chat completion: {e}") from e
    return ChatReply(content, tuple(calls))
