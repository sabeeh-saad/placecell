"""Text embeddings from any server speaking the OpenAI embeddings API.

That covers OpenAI itself, OpenRouter (`https://openrouter.ai/api/v1`), Gemini's
compatibility endpoint (`https://generativelanguage.googleapis.com/v1beta/openai`) and local
servers such as vLLM or Ollama. Requests are batched, rate limits and server errors are
retried with backoff, and the transport is injectable so tests never touch the network.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from placecell.errors import ProviderError, RateLimitedError, UnsupportedMediaError, ValidationError
from placecell.memory import Evidence, Matrix
from placecell.providers.base import Capabilities, normalise_rows


class Transport(Protocol):
    """Minimal HTTP surface: post JSON, get status, headers and decoded body back."""

    def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, str], Any]: ...


class UrllibTransport:
    """Standard-library transport. HTTP errors are returned, not raised, so the caller can retry."""

    def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, str], Any]:
        if not url.startswith(("https://", "http://")):
            raise ValidationError(f"only http(s) endpoints are supported, got {url!r}")
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            url, data=body, method="POST", headers={**headers, "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - scheme checked above
                return response.status, dict(response.headers.items()), _decode(response.read())
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers.items()), _decode(e.read())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(f"request to {url} failed: {e}") from e


def _decode(raw: bytes) -> Any:
    try:
        return json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return raw.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0

    def __post_init__(self) -> None:
        if self.attempts < 1 or self.base_delay_s < 0 or self.max_delay_s < self.base_delay_s:
            raise ValidationError("retry policy out of range")

    def delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.max_delay_s)
            except ValueError:
                pass
        return min(self.base_delay_s * 2.0**attempt, self.max_delay_s)


class OpenAICompatibleEmbedder:
    """Embeds text through the `/embeddings` endpoint of an OpenAI-compatible server."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        dimension: int | None = None,
        batch_size: int = 64,
        timeout_s: float = 30.0,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not model:
            raise ValidationError("model must not be empty")
        if not base_url.startswith(("https://", "http://")):
            raise ValidationError("base_url must be an http(s) URL")
        if batch_size < 1 or timeout_s <= 0:
            raise ValidationError("batch_size must be positive and timeout_s greater than zero")
        self._model = model
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {**(extra_headers or {})}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._dimension = dimension
        self._batch_size = batch_size
        self._timeout_s = timeout_s
        self._transport = transport or UrllibTransport()
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._capabilities = Capabilities(text=True, image=False, video=False)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        """Vector size. Probed with one request if it was not given at construction."""
        if self._dimension is None:
            self._dimension = self._request(["placecell"])[0].shape[0]
        return self._dimension

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def embed_text(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        rows = [self._request(texts[i : i + self._batch_size]) for i in range(0, len(texts), self._batch_size)]
        return normalise_rows(np.vstack(rows), len(texts), self.dimension)

    def embed_media(self, items: Sequence[Evidence]) -> Matrix:
        raise UnsupportedMediaError(f"{self._model} via the embeddings endpoint takes text only")

    def _request(self, batch: Sequence[str]) -> Matrix:
        payload = {"model": self._model, "input": list(batch)}
        for attempt in range(self._retry.attempts):
            status, headers, body = self._transport.post_json(self._url, self._headers, payload, self._timeout_s)
            if status == 200:
                return self._parse(body, len(batch))
            if status == 429 or status >= 500:
                if attempt + 1 < self._retry.attempts:
                    self._sleep(self._retry.delay(attempt, _header(headers, "retry-after")))
                    continue
                if status == 429:
                    raise RateLimitedError(f"{self._url}: rate limited after {self._retry.attempts} attempts")
                raise ProviderError(f"{self._url}: server error {status} after {self._retry.attempts} attempts")
            raise ProviderError(f"{self._url}: HTTP {status}: {_message(body)}")
        raise ProviderError("unreachable")  # pragma: no cover

    def _parse(self, body: Any, expected: int) -> Matrix:
        try:
            data = sorted(body["data"], key=lambda d: d["index"])
            matrix = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(f"malformed embeddings response: {_message(body)}") from e
        if matrix.ndim != 2 or matrix.shape[0] != expected:
            raise ProviderError(f"expected {expected} embeddings, got {matrix.shape}")
        if self._dimension is not None and matrix.shape[1] != self._dimension:
            raise ProviderError(f"model returned dimension {matrix.shape[1]}, configured {self._dimension}")
        return matrix


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _message(body: Any) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])
    return str(body)[:200]
