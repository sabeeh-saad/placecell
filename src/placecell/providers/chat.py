"""Chat completions with function calling from any OpenAI-compatible server."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from placecell.chat import ChatMessage, ChatModel, ChatReply, ToolCall
from placecell.errors import ProviderError, ValidationError
from placecell.providers._http import Endpoint, RetryPolicy, Transport, message


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
        if max_tokens < 1 or temperature < 0:
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
    try:
        if body["choices"][0].get("finish_reason", "stop") not in {"stop", "tool_calls"}:
            raise ProviderError("chat completion was interrupted or truncated")
        msg = body["choices"][0]["message"]
        content = msg.get("content")
        raw_calls = msg.get("tool_calls") or []
        calls = []
        for c in raw_calls:
            arguments = c["function"].get("arguments") or "{}"
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(parsed, dict):
                raise ProviderError(f"tool arguments must be an object, got {type(parsed).__name__}")
            calls.append(ToolCall(str(c["id"]), str(c["function"]["name"]), parsed))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        raise ProviderError(f"malformed chat completion: {message(body)}") from e
    if isinstance(content, list):
        content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ChatReply(content if isinstance(content, str) else None, tuple(calls))
