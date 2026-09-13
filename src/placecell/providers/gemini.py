"""Gemini Embedding 2 through its native multimodal REST endpoint; no model download."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from placecell.errors import ProviderError, UnsupportedMediaError, ValidationError
from placecell.memory import Evidence, EvidenceKind, Matrix
from placecell.providers._http import Endpoint, RetryPolicy, Transport
from placecell.providers.base import Capabilities, normalise_rows

DEFAULT_GEMINI_MODEL = "gemini-embedding-2"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiEmbedder:
    """Independent image/document embeddings and asymmetric retrieval-query embeddings.

    Uses batchEmbedContents with one input per request, avoiding Embedding 2's aggregation
    of multiple parts into one vector. Local PNG/JPEG bytes are uploaded inline. Text
    follows the model's documented search-query/document prefixes, not the older taskType
    field. Model, dimension and formatting version are part of the collection identity.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        api_key: str | None = None,
        dimension: int = 768,
        base_url: str = GEMINI_BASE_URL,
        batch_size: int = 16,
        timeout_s: float = 60.0,
        max_image_bytes: int = 8_000_000,
        max_batch_bytes: int = 12_000_000,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if model not in {DEFAULT_GEMINI_MODEL, "gemini-embedding-2-preview"}:
            raise ValidationError("GeminiEmbedder requires gemini-embedding-2 or gemini-embedding-2-preview")
        if dimension not in {768, 1536, 3072}:
            raise ValidationError("Gemini embedding dimension must be 768, 1536 or 3072")
        if not 1 <= batch_size <= 100 or max_image_bytes < 1 or max_batch_bytes < 1:
            raise ValidationError("Gemini batch_size must be 1-100 and byte limits must be positive")
        if not api_key or not api_key.strip():
            raise ValidationError("GeminiEmbedder requires an API key (for example from GEMINI_API_KEY)")
        self._model, self._dimension, self._batch_size = model, dimension, batch_size
        self._max_image_bytes, self._max_batch_bytes = max_image_bytes, max_batch_bytes
        self._endpoint = Endpoint.build(
            base_url,
            f"/models/{model}:batchEmbedContents",
            None,
            timeout_s,
            transport,
            retry,
            sleep,
            {"x-goog-api-key": api_key},
        )

    @property
    def model_name(self) -> str:
        return f"gemini:{self._model}:{self.dimension}:retrieval-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(text=True, image=True)

    def embed_text(self, texts: Sequence[str]) -> Matrix:
        """Encode stored captions/documents. Use embed_queries for retrieval queries."""
        return self._embed({"text": f"title: none | text: {text}"} for text in texts)

    def embed_queries(self, texts: Sequence[str]) -> Matrix:
        return self._embed({"text": f"task: search result | query: {text}"} for text in texts)

    def embed_media(self, items: Sequence[Evidence]) -> Matrix:
        if any(item.kind is not EvidenceKind.FRAME for item in items):
            raise UnsupportedMediaError("GeminiEmbedder currently accepts frames; sample video into frames first")
        return self._embed(self._image_part(item) for item in items)

    def _image_part(self, evidence: Evidence) -> dict[str, Any]:
        if "://" in evidence.uri and not evidence.uri.startswith("file://"):
            raise UnsupportedMediaError("GeminiEmbedder requires local PNG or JPEG files")
        path = Path(evidence.uri.removeprefix("file://")).expanduser()
        try:
            with path.open("rb") as stream:
                raw = stream.read(self._max_image_bytes + 1)
        except OSError as e:
            raise ProviderError(f"cannot read image {path}: {e}") from e
        if len(raw) > self._max_image_bytes:
            raise ValidationError("image exceeds GeminiEmbedder max_image_bytes")
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif raw.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        else:
            raise UnsupportedMediaError("GeminiEmbedder requires PNG or JPEG image bytes")
        return {"inline_data": {"mime_type": mime, "data": base64.b64encode(raw).decode("ascii")}}

    def _embed(self, parts: Iterable[dict[str, Any]]) -> Matrix:
        matrices = []
        pending: list[dict[str, Any]] = []
        size = 16
        for part in parts:
            request = {
                "model": f"models/{self._model}",
                "content": {"parts": [part]},
                "outputDimensionality": self.dimension,
            }
            request_size = len(json.dumps(request).encode("utf-8")) + 2
            if request_size + 16 > self._max_batch_bytes:
                raise ValidationError("input exceeds GeminiEmbedder max_batch_bytes")
            if pending and (len(pending) == self._batch_size or size + request_size > self._max_batch_bytes):
                matrices.append(self._request(pending))
                pending, size = [], 16
            pending.append(request)
            size += request_size
        if pending:
            matrices.append(self._request(pending))
        return np.concatenate(matrices) if matrices else np.empty((0, self.dimension), dtype=np.float32)

    def _request(self, requests: Sequence[dict[str, Any]]) -> Matrix:
        body = self._endpoint.post({"requests": list(requests)})
        try:
            vectors = [embedding["values"] for embedding in body["embeddings"]]
            result = normalise_rows(vectors, len(requests), self.dimension)
            if np.any(np.linalg.norm(result, axis=1) == 0):
                raise ValidationError("empty embedding signal")
            return result
        except (KeyError, TypeError, ValueError, ValidationError) as e:
            raise ProviderError("Gemini returned invalid embeddings (count, dimension or vector values)") from e
