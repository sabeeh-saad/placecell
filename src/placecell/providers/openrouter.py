"""Gemini image/text embeddings through OpenRouter's multimodal embeddings endpoint."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from placecell.errors import ProviderError, ValidationError
from placecell.memory import Matrix
from placecell.providers._http import Endpoint, RetryPolicy, Transport
from placecell.providers.base import normalise_rows
from placecell.providers.gemini import GeminiEmbedder

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterGeminiEmbedder(GeminiEmbedder):
    """Uses the same retrieval prefixes and bounded images as the native Gemini adapter.

    Collections have a separate identity because routing can change provider behavior.
    Each input block produces one vector; image and caption are never pooled together.
    """

    def __init__(
        self,
        model: str = "google/gemini-embedding-2",
        *,
        api_key: str | None = None,
        dimension: int = 768,
        base_url: str = OPENROUTER_BASE_URL,
        batch_size: int = 16,
        timeout_s: float = 60.0,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        if not model.startswith("google/"):
            raise ValidationError("OpenRouter Gemini model must have the google/ prefix")
        super().__init__(model.removeprefix("google/"), api_key=api_key, dimension=dimension, batch_size=batch_size)
        self._router_model = model
        self._endpoint = Endpoint.build(base_url, "/embeddings", api_key, timeout_s, transport, retry, time.sleep, None)

    @property
    def model_name(self) -> str:
        return f"openrouter:{self._router_model}:{self.dimension}:retrieval-v1"

    def _request(self, requests: Sequence[dict[str, Any]]) -> Matrix:
        inputs = []
        for request in requests:
            part = request["content"]["parts"][0]
            if "text" in part:
                content = {"type": "text", "text": part["text"]}
            else:
                data = part["inline_data"]
                content = {"type": "image_url", "image_url": {"url": f"data:{data['mime_type']};base64,{data['data']}"}}
            inputs.append({"content": [content]})
        body = self._endpoint.post(
            {
                "model": self._router_model,
                "input": inputs,
                "dimensions": self.dimension,
                "encoding_format": "float",
            }
        )
        try:
            data = sorted(body["data"], key=lambda row: row["index"])
            if [row["index"] for row in data] != list(range(len(inputs))):
                raise ValueError("invalid embedding indices")
            result = normalise_rows([row["embedding"] for row in data], len(inputs), self.dimension)
            if np.any(np.linalg.norm(result, axis=1) == 0):
                raise ValueError("empty embedding signal")
            return result
        except (KeyError, TypeError, ValueError, ValidationError) as e:
            raise ProviderError("OpenRouter returned invalid embeddings (indices, count, dimension or values)") from e
