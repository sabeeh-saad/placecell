"""A network-free text embedder for tests, demos and smoke runs.

Words are hashed into buckets with a stable hash, so the same text always yields the same
vector on every machine. Similar wording gives similar vectors; nothing more is promised.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from placecell.errors import UnsupportedMediaError, ValidationError
from placecell.memory import Evidence, Matrix
from placecell.providers.base import Capabilities, normalise_rows

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """Deterministic bag-of-hashed-words embedder. Text only."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValidationError("dimension must be at least 8")
        self._dimension = dimension
        self._capabilities = Capabilities(text=True, image=False, video=False)

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def embed_text(self, texts: Sequence[str]) -> Matrix:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                bucket = value % self._dimension
                sign = 1.0 if (value >> 63) & 1 else -1.0
                matrix[row, bucket] += sign
        return normalise_rows(matrix, len(texts), self._dimension)

    def embed_media(self, items: Sequence[Evidence]) -> Matrix:
        raise UnsupportedMediaError(f"{self.model_name} embeds text only")
