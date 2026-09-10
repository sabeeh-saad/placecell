"""Provider contract.

A provider turns text or media into unit-length float32 vectors of a fixed dimension.
It declares which media kinds it accepts, so the ingestion pipeline can route clips only
to backends that understand video. Batching, retries and rate limits are the adapter's
job; callers hand over whole batches and get one matrix back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from placecell.errors import ValidationError
from placecell.memory import Evidence, EvidenceKind


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Which inputs a provider can embed."""

    text: bool = True
    image: bool = False
    video: bool = False

    def supports(self, evidence: Evidence) -> bool:
        if evidence.kind is EvidenceKind.FRAME:
            return self.image
        return self.video


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract every embedding backend fulfils.

    Both embed methods return an array of shape (len(inputs), dimension), dtype float32,
    with every row scaled to unit length, in input order.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def capabilities(self) -> Capabilities: ...

    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_media(self, items: Sequence[Evidence]) -> np.ndarray: ...


@runtime_checkable
class Captioner(Protocol):
    """Describes media in words, one caption per item, in input order."""

    def caption(self, items: Sequence[Evidence]) -> list[str]: ...


def normalise_rows(vectors: np.ndarray, expected_rows: int, dimension: int) -> np.ndarray:
    """Coerce a backend's output into the contract shape and scale rows to unit length.

    All-zero rows are left as zeros rather than becoming NaN, so an empty input still
    yields a valid, if useless, vector.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape != (expected_rows, dimension):
        raise ValidationError(f"expected an embedding matrix of shape {(expected_rows, dimension)}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError("embedding matrix contains NaN or infinity")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(arr / norms, dtype=np.float32)
