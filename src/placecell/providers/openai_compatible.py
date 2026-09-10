"""Text embeddings from any server speaking the OpenAI embeddings API.

That covers OpenAI itself, OpenRouter (`https://openrouter.ai/api/v1`), Gemini's
compatibility endpoint (`https://generativelanguage.googleapis.com/v1beta/openai`) and local
servers such as vLLM or Ollama. Requests are batched, rate limits and server errors are
retried with backoff, and the transport is injectable so tests never touch the network.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from placecell.errors import ProviderError, UnsupportedMediaError, ValidationError
from placecell.memory import Evidence, Matrix
from placecell.providers._http import Endpoint, RetryPolicy, Transport, UrllibTransport, message
from placecell.providers.base import Capabilities, normalise_rows

__all__ = ["OpenAICompatibleEmbedder", "RetryPolicy", "Transport", "UrllibTransport"]


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
        if batch_size < 1:
            raise ValidationError("batch_size must be positive")
        self._model = model
        self._endpoint = Endpoint.build(
            base_url, "/embeddings", api_key, timeout_s, transport, retry, sleep, extra_headers
        )
        self._dimension = dimension
        self._batch_size = batch_size
        self._capabilities = Capabilities(text=True, image=False, video=False)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        """Vector size. Probed with one request if it was not given at construction."""
        if self._dimension is None:
            self._dimension = int(self._request(["placecell"]).shape[1])
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
        body = self._endpoint.post({"model": self._model, "input": list(batch)})
        return self._parse(body, len(batch))

    def _parse(self, body: Any, expected: int) -> Matrix:
        try:
            data = sorted(body["data"], key=lambda d: d["index"])
            matrix = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(f"malformed embeddings response: {message(body)}") from e
        if matrix.ndim != 2 or matrix.shape[0] != expected:
            raise ProviderError(f"expected {expected} embeddings, got {matrix.shape}")
        if self._dimension is not None and matrix.shape[1] != self._dimension:
            raise ProviderError(f"model returned dimension {matrix.shape[1]}, configured {self._dimension}")
        return matrix
