"""Captions from a vision model behind the OpenAI chat completions API.

One request per frame: the image goes in as a data URL, the model answers with a short
description. Clips are not sent; they get an empty caption and the pipeline reports them
if nothing else can embed them. Works with OpenAI, OpenRouter, Gemini's compatibility
endpoint and local servers hosting a vision-language model.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from placecell.errors import ProviderError, ValidationError
from placecell.memory import Evidence, EvidenceKind
from placecell.providers._http import Endpoint, RetryPolicy, Transport, message

DEFAULT_PROMPT = (
    "You are the eyes of a mobile robot. Describe what is in this image in one or two plain "
    "sentences: the objects, where they are relative to each other, and anything a person "
    "might later ask the robot to find. No preamble."
)


class OpenAICompatibleCaptioner:
    """Describes frames with a vision-language model through `/chat/completions`."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 120,
        detail: str = "low",
        timeout_s: float = 60.0,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not model or not prompt.strip():
            raise ValidationError("model and prompt must not be empty")
        if max_tokens < 1:
            raise ValidationError("max_tokens must be positive")
        self._model = model
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._detail = detail
        self._endpoint = Endpoint.build(
            base_url, "/chat/completions", api_key, timeout_s, transport, retry, sleep, extra_headers
        )

    @property
    def model_name(self) -> str:
        return self._model

    def caption(self, items: Sequence[Evidence]) -> list[str]:
        return [self._caption_one(item) if item.kind is EvidenceKind.FRAME else "" for item in items]

    def _caption_one(self, item: Evidence) -> str:
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": {"url": data_url(item.uri), "detail": self._detail}},
                    ],
                }
            ],
        }
        return parse_text(self._endpoint.post(payload))


def data_url(uri: str) -> str:
    """Local image file as a base64 data URL. Only files are supported; remote URIs are not fetched."""
    path = Path(uri.removeprefix("file://"))
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        raise ValidationError(f"{uri} is not an image file")
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ProviderError(f"cannot read {uri}: {e}") from e
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def parse_text(body: Any) -> str:
    """The assistant text of a chat completion; content may be a string or a list of parts."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"malformed chat completion: {message(body)}") from e
    if isinstance(content, list):
        content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise ProviderError(f"malformed chat completion content: {message(body)}")
    return " ".join(content.split())
