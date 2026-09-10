from __future__ import annotations

import numpy as np
import pytest

from placecell.errors import ProviderError, RateLimitedError, UnsupportedMediaError, ValidationError
from placecell.providers import Capabilities, Captioner, EmbeddingProvider, HashingEmbedder, normalise_rows
from placecell.providers.openai_compatible import OpenAICompatibleEmbedder, RetryPolicy
from tests.conftest import FakeCaptioner, FakeMediaEmbedder, FakeTransport, frame


def test_hashing_embedder_is_deterministic_unit_length_and_text_only() -> None:
    e = HashingEmbedder(32)
    a = e.embed_text(["A red Chair", "a red chair", "blue door"])
    assert a.shape == (3, 32) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a[0], a[1])
    assert float(a[0] @ a[2]) < float(a[0] @ a[1])
    assert np.allclose(e.embed_text(["a red chair"])[0], HashingEmbedder(32).embed_text(["a red chair"])[0])
    assert e.embed_text([""]).shape == (1, 32) and not np.any(e.embed_text([""]))
    assert isinstance(e, EmbeddingProvider)
    with pytest.raises(UnsupportedMediaError):
        e.embed_media([frame()])
    with pytest.raises(ValidationError):
        HashingEmbedder(4)


def test_capabilities_route_by_evidence_kind() -> None:
    from placecell import Evidence, EvidenceKind

    caps = Capabilities(text=True, image=True, video=False)
    assert caps.supports(frame())
    assert not caps.supports(Evidence(EvidenceKind.CLIP, "a.mp4", duration_s=2))


def test_normalise_rows_validates_shape_and_handles_zero_rows() -> None:
    out = normalise_rows(np.array([[3.0, 4.0], [0.0, 0.0]]), 2, 2)
    assert np.allclose(out[0], [0.6, 0.8]) and not np.any(out[1])
    with pytest.raises(ValidationError):
        normalise_rows(np.ones((2, 3)), 2, 2)
    with pytest.raises(ValidationError):
        normalise_rows(np.array([[1.0, float("nan")]]), 1, 2)


def test_fakes_fulfil_the_protocols() -> None:
    assert isinstance(FakeMediaEmbedder(), EmbeddingProvider)
    assert isinstance(FakeCaptioner(), Captioner)


def _ok(vectors: list[list[float]], shuffle: bool = False) -> tuple[int, dict[str, str], dict]:
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    if shuffle:
        data.reverse()
    return 200, {}, {"data": data, "model": "m"}


def test_openai_embedder_batches_orders_and_normalises() -> None:
    transport = FakeTransport([_ok([[1, 0], [0, 2]], shuffle=True), _ok([[3, 4]])])
    e = OpenAICompatibleEmbedder(
        "m", base_url="https://x/v1/", api_key="k", dimension=2, batch_size=2, transport=transport
    )
    out = e.embed_text(["a", "b", "c"])
    assert out.shape == (3, 2)
    assert np.allclose(out, [[1, 0], [0, 1], [0.6, 0.8]])
    assert [r["payload"]["input"] for r in transport.requests] == [["a", "b"], ["c"]]
    assert transport.requests[0]["url"] == "https://x/v1/embeddings"
    assert transport.requests[0]["headers"]["Authorization"] == "Bearer k"
    assert e.embed_text([]).shape == (0, 2)


def test_openai_embedder_probes_dimension_when_not_given() -> None:
    transport = FakeTransport([_ok([[1, 2, 3]]), _ok([[1, 1, 1]])])
    e = OpenAICompatibleEmbedder("m", transport=transport)
    assert e.dimension == 3
    assert e.embed_text(["x"]).shape == (1, 3)
    assert e.model_name == "m" and not e.capabilities.image
    with pytest.raises(UnsupportedMediaError):
        e.embed_media([frame()])


def test_openai_embedder_retries_rate_limits_then_gives_up() -> None:
    sleeps: list[float] = []
    retry = RetryPolicy(attempts=3, base_delay_s=1.0, max_delay_s=4.0)
    transport = FakeTransport(
        [(429, {"Retry-After": "2"}, {"error": {"message": "slow down"}}), (503, {}, "bad gateway"), _ok([[1, 0]])]
    )
    e = OpenAICompatibleEmbedder("m", dimension=2, transport=transport, retry=retry, sleep=sleeps.append)
    assert e.embed_text(["a"]).shape == (1, 2)
    assert sleeps == [2.0, 2.0]  # Retry-After honoured, then exponential 1*2**1
    transport = FakeTransport([(429, {}, None)] * 3)
    e = OpenAICompatibleEmbedder("m", dimension=2, transport=transport, retry=retry, sleep=sleeps.append)
    with pytest.raises(RateLimitedError):
        e.embed_text(["a"])
    assert len(transport.requests) == 3


def test_openai_embedder_reports_client_errors_and_bad_bodies() -> None:
    transport = FakeTransport([(401, {}, {"error": {"message": "no key"}})])
    e = OpenAICompatibleEmbedder("m", dimension=2, transport=transport)
    with pytest.raises(ProviderError, match="no key"):
        e.embed_text(["a"])
    e = OpenAICompatibleEmbedder("m", dimension=2, transport=FakeTransport([(200, {}, {"data": "nope"})]))
    with pytest.raises(ProviderError, match="malformed"):
        e.embed_text(["a"])
    e = OpenAICompatibleEmbedder("m", dimension=2, transport=FakeTransport([_ok([[1, 2, 3]])]))
    with pytest.raises(ProviderError, match="dimension"):
        e.embed_text(["a"])
    e = OpenAICompatibleEmbedder("m", dimension=2, transport=FakeTransport([_ok([[1, 2]])]))
    with pytest.raises(ProviderError, match="expected 2"):
        e.embed_text(["a", "b"])
    with pytest.raises(ValidationError):
        OpenAICompatibleEmbedder("")
    with pytest.raises(ValidationError):
        OpenAICompatibleEmbedder("m", base_url="ftp://x")
    with pytest.raises(ValidationError):
        RetryPolicy(attempts=0)
