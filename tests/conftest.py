from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from placecell import CollectionInfo, Evidence, EvidenceKind, InMemoryStore, Memory, Pose, VectorStore
from placecell.providers import Capabilities, HashingEmbedder, normalise_rows

DIM = 64


class FakeMediaEmbedder:
    """Embeds text like HashingEmbedder and frames by hashing their digest. Declares image support."""

    def __init__(self, dimension: int = DIM, video: bool = False) -> None:
        self._text = HashingEmbedder(dimension)
        self._dimension = dimension
        self.capabilities = Capabilities(text=True, image=True, video=video)
        self.media_calls: list[list[Evidence]] = []
        self.text_calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return f"fake-media-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        self.text_calls.append(list(texts))
        return self._text.embed_text(texts)

    def embed_media(self, items: Sequence[Evidence]) -> np.ndarray:
        self.media_calls.append(list(items))
        matrix = np.zeros((len(items), self._dimension), dtype=np.float32)
        for row, item in enumerate(items):
            seed = int.from_bytes(hashlib.blake2b((item.digest or item.uri).encode(), digest_size=8).digest(), "big")
            matrix[row] = np.random.default_rng(seed).standard_normal(self._dimension)
        return normalise_rows(matrix, len(items), self._dimension)


class FakeCaptioner:
    def __init__(self, text: str = "a {kind} at {uri}") -> None:
        self._text = text
        self.calls: list[list[Evidence]] = []

    def caption(self, items: Sequence[Evidence]) -> list[str]:
        self.calls.append(list(items))
        return [self._text.format(kind=i.kind.value, uri=i.uri) for i in items]


class FakeTransport:
    """Scripted HTTP responses: each call pops the next (status, headers, body)."""

    def __init__(self, responses: list[tuple[int, Mapping[str, str], Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def post_json(self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float):
        self.requests.append({"url": url, "headers": dict(headers), "payload": dict(payload), "timeout": timeout_s})
        if not self.responses:
            raise AssertionError("transport called more often than scripted")
        return self.responses.pop(0)


def frame(uri: str = "frames/a.jpg", digest: str = "") -> Evidence:
    return Evidence(EvidenceKind.FRAME, uri, digest)


def embedded(
    embedder: HashingEmbedder | FakeMediaEmbedder,
    caption: str,
    t: float = 1000.0,
    x: float = 0.0,
    y: float = 0.0,
    robot: str = "r1",
    camera: str = "front",
    **fields: Any,
) -> Memory:
    m = Memory.create(robot, camera, t, Pose(x, y), frame(f"frames/{camera}_{round(t)}.jpg"), caption)
    m = m.with_embedding(embedder.embed_text([caption])[0], embedder.model_name)
    if fields:
        from dataclasses import replace

        m = replace(m, **fields)
    return m


@pytest.fixture
def hashing() -> HashingEmbedder:
    return HashingEmbedder(DIM)


def _backends() -> list[str]:
    try:
        import lancedb  # noqa: F401
    except ImportError:  # pragma: no cover
        return ["memory"]
    return ["memory", "lancedb"]


@pytest.fixture(params=_backends())
def store(request: pytest.FixtureRequest, hashing: HashingEmbedder, tmp_path: Path) -> Iterator[VectorStore]:
    """A fresh collection on each backend, so every store test doubles as a contract test."""
    info = CollectionInfo("test", hashing.model_name, DIM)
    if request.param == "memory":
        s: VectorStore = InMemoryStore(info)
    else:
        from placecell.store.lancedb_store import LanceDBStore

        s = LanceDBStore(tmp_path / "db", info)
    yield s
    s.close()


@pytest.fixture
def media_embedder() -> FakeMediaEmbedder:
    return FakeMediaEmbedder()


@pytest.fixture
def media_store(media_embedder: FakeMediaEmbedder) -> InMemoryStore:
    return InMemoryStore(CollectionInfo("media", media_embedder.model_name, DIM))
