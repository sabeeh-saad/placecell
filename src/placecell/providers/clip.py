"""Local CLIP image/text embeddings through the optional Sentence Transformers extra."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from placecell.errors import PlacecellError, ProviderError, UnsupportedMediaError, ValidationError
from placecell.memory import Evidence, EvidenceKind, Matrix
from placecell.providers.base import Capabilities, normalise_rows

DEFAULT_CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"
_DIMENSIONS = {
    DEFAULT_CLIP_MODEL: 512,
    "sentence-transformers/clip-ViT-B-16": 512,
    "sentence-transformers/clip-ViT-L-14": 768,
}


class ClipEmbedder:
    """Embed local frames and text in one space; weights load on the first nonempty batch.

    Install placecell[clip]. Pin a model revision for repeatable deployments; the revision
    is part of the collection identity. Calls are serialized so camera, query and maintenance
    workers share one model instance. Clips must first be sampled into frames.
    """

    def __init__(
        self,
        model: str = DEFAULT_CLIP_MODEL,
        *,
        revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        cache_folder: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        if model not in _DIMENSIONS:
            raise ValidationError(f"unsupported CLIP checkpoint; choose one of {sorted(_DIMENSIONS)}")
        if batch_size < 1 or not device.strip() or (revision is not None and not revision.strip()):
            raise ValidationError("CLIP needs a positive batch size and nonempty device/revision")
        self._model, self._revision = model, revision
        self._device, self._batch_size = device, batch_size
        self._cache_folder, self._local_files_only = cache_folder, local_files_only
        self._encoder: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return f"clip:{self._model}@{self._revision or 'default'}"

    @property
    def dimension(self) -> int:
        return _DIMENSIONS[self._model]

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(text=True, image=True)

    def _load(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise PlacecellError("ClipEmbedder needs the optional extra: pip install 'placecell[clip]'") from e
            self._encoder = SentenceTransformer(
                self._model,
                revision=self._revision,
                device=self._device,
                cache_folder=self._cache_folder,
                local_files_only=self._local_files_only,
                trust_remote_code=False,
            )
        return self._encoder

    def _encode(self, items: Sequence[Any]) -> Matrix:
        vectors = self._load().encode(
            list(items),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        try:
            result = normalise_rows(vectors, len(items), self.dimension)
        except ValidationError as e:
            raise ProviderError(f"CLIP returned invalid embeddings: {e}") from e
        if np.any(np.linalg.norm(result, axis=1) == 0):
            raise ProviderError("CLIP returned an embedding with no signal")
        return result

    def embed_text(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            with self._lock:
                return np.concatenate(
                    [self._encode(texts[i : i + self._batch_size]) for i in range(0, len(texts), self._batch_size)]
                )
        except PlacecellError:
            raise
        except Exception as e:
            raise ProviderError(f"CLIP text encoding failed: {e}") from e

    def embed_media(self, items: Sequence[Evidence]) -> Matrix:
        if any(e.kind is not EvidenceKind.FRAME for e in items):
            raise UnsupportedMediaError("CLIP accepts frames; sample video clips into frames first")
        if any("://" in e.uri and not e.uri.startswith("file://") for e in items):
            raise UnsupportedMediaError("CLIP requires local image files")
        if not items:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            from PIL import Image, ImageOps

            batches = []
            with self._lock:
                for offset in range(0, len(items), self._batch_size):
                    images = []
                    try:
                        for evidence in items[offset : offset + self._batch_size]:
                            with Image.open(Path(evidence.uri.removeprefix("file://")).expanduser()) as source:
                                images.append(ImageOps.exif_transpose(source).convert("RGB"))
                        batches.append(self._encode(images))
                    finally:
                        for image in images:
                            image.close()
            return np.concatenate(batches)
        except PlacecellError:
            raise
        except ImportError as e:
            raise PlacecellError("ClipEmbedder needs the optional extra: pip install 'placecell[clip]'") from e
        except Exception as e:
            raise ProviderError(f"CLIP image encoding failed: {e}") from e
