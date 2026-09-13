import base64
import json

import numpy as np
import pytest

from placecell import CollectionInfo, Evidence, EvidenceKind, InMemoryStore, Recall
from placecell.errors import ProviderError, RateLimitedError, UnsupportedMediaError, ValidationError
from placecell.evaluation import main
from placecell.providers import GeminiEmbedder, RetryPolicy
from placecell.providers.embedding import embed_memories
from placecell.providers.gemini import GEMINI_BASE_URL
from placecell.ros2.node import build_embedder
from tests.conftest import FakeTransport, embedded, frame
from tests.test_multimodal import dual


def response(count=1, dimension=768):
    return {"embeddings": [{"values": [3, 4, *([0] * (dimension - 2))]} for _ in range(count)]}


def test_native_api_batches_independent_documents_and_queries():
    transport = FakeTransport([(200, {}, response(2)), (200, {}, response()), (200, {}, response())])
    embedder = GeminiEmbedder(api_key="test-key", transport=transport, batch_size=2)
    assert embedder.capabilities.image and embedder.capabilities.text and not embedder.capabilities.video
    assert embedder.dimension == 768 and embedder.model_name == "gemini:gemini-embedding-2:768:retrieval-v1"
    assert (
        embedder.embed_text([]).shape == embedder.embed_queries([]).shape == embedder.embed_media([]).shape == (0, 768)
    )
    assert transport.requests == []
    documents = embedder.embed_text(["printer", "desk", "chair"])
    queries = embedder.embed_queries(["printer"])
    assert documents.shape == (3, 768) and queries.shape == (1, 768)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1)
    calls = transport.requests
    assert calls[0]["url"] == GEMINI_BASE_URL + "/models/gemini-embedding-2:batchEmbedContents"
    assert calls[0]["headers"] == {"x-goog-api-key": "test-key"}
    assert [len(c["payload"]["requests"]) for c in calls] == [2, 1, 1]
    first = calls[0]["payload"]["requests"][0]
    assert first == {
        "model": "models/gemini-embedding-2",
        "content": {"parts": [{"text": "title: none | text: printer"}]},
        "outputDimensionality": 768,
    }
    assert calls[-1]["payload"]["requests"][0]["content"]["parts"] == [{"text": "task: search result | query: printer"}]


def test_images_are_uploaded_separately_with_bounded_payloads(tmp_path):
    png, jpeg = tmp_path / "frame.png", tmp_path / "frame.jpg"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"a" * 100)
    jpeg.write_bytes(b"\xff\xd8\xff" + b"b" * 100)
    transport = FakeTransport([(200, {}, response()), (200, {}, response())])
    embedder = GeminiEmbedder(api_key="key", transport=transport, max_batch_bytes=500)
    vectors = embedder.embed_media([frame(str(png)), frame(f"file://{jpeg}")])
    assert vectors.shape == (2, 768) and len(transport.requests) == 2
    for call, path, mime in zip(transport.requests, (png, jpeg), ("image/png", "image/jpeg"), strict=True):
        requests = call["payload"]["requests"]
        assert len(requests) == 1
        part = requests[0]["content"]["parts"][0]
        assert part["inline_data"]["mime_type"] == mime
        assert base64.b64decode(part["inline_data"]["data"]) == path.read_bytes()
        assert len(json.dumps(call["payload"]).encode()) <= 500


def test_ingestion_keeps_image_and_caption_vectors_and_recall_uses_query_encoder(tmp_path, hashing):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"\xff\xd8\xff" + b"sample")
    transport = FakeTransport([(200, {}, response()), (200, {}, response()), (200, {}, response())])
    embedder = GeminiEmbedder(api_key="key", transport=transport)
    old = embedded(hashing, "desk", evidence=frame(str(image)))
    rows, rejected = embed_memories([old], embedder)
    assert not rejected and rows[0].embedding_kind == "image" and rows[0].caption_embedding is not None
    store = InMemoryStore(CollectionInfo("gemini", embedder.model_name, embedder.dimension))
    store.upsert(rows)
    hits = Recall(store, embedder, clock=lambda: 1000).similar("printer")
    assert hits[0].image_similarity == pytest.approx(1) and hits[0].caption_similarity == pytest.approx(1)
    assert transport.requests[-1]["payload"]["requests"][0]["content"]["parts"] == [
        {"text": "task: search result | query: printer"}
    ]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"embeddings": []},
        {"embeddings": [{"values": [1, 2]}]},
        {"embeddings": [{"values": [0] * 768}]},
        {"embeddings": [{"values": [float("nan")] * 768}]},
    ],
)
def test_malformed_embeddings_are_rejected(body):
    embedder = GeminiEmbedder(api_key="key", transport=FakeTransport([(200, {}, body)]))
    with pytest.raises(ProviderError):
        embedder.embed_queries(["printer"])


def test_gemini_retries_rate_limits_and_reports_auth_errors():
    delays = []
    transport = FakeTransport([(429, {"Retry-After": "0.2"}, {}), (200, {}, response())])
    embedder = GeminiEmbedder(api_key="key", transport=transport, sleep=delays.append)
    assert embedder.embed_text(["printer"]).shape == (1, 768) and delays == [0.2]
    assert transport.requests[0]["payload"] == transport.requests[1]["payload"]
    for status, error in ((401, ProviderError), (429, RateLimitedError)):
        embedder = GeminiEmbedder(
            api_key="key", transport=FakeTransport([(status, {}, {})]), retry=RetryPolicy(attempts=1)
        )
        with pytest.raises(error):
            embedder.embed_text(["printer"])


def test_validation_limits_and_unsupported_media_do_not_send_requests(tmp_path):
    for kwargs in (
        {"api_key": ""},
        {"model": "gemini-embedding-001"},
        {"dimension": 64},
        {"batch_size": 0},
        {"batch_size": 101},
        {"max_image_bytes": 0},
        {"max_batch_bytes": 0},
    ):
        with pytest.raises(ValidationError):
            GeminiEmbedder(**{"api_key": "key", **kwargs})
    transport = FakeTransport([])
    embedder = GeminiEmbedder(api_key="key", transport=transport, max_image_bytes=10, max_batch_bytes=300)
    for evidence in (Evidence(EvidenceKind.CLIP, "clip.mp4", duration_s=1), frame("https://example.com/a.jpg")):
        with pytest.raises(UnsupportedMediaError):
            embedder.embed_media([evidence])
    with pytest.raises(ProviderError):
        embedder.embed_media([frame(str(tmp_path / "missing.jpg"))])
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"invalid")
    with pytest.raises(UnsupportedMediaError):
        embedder.embed_media([frame(str(path))])
    path.write_bytes(b"x" * 11)
    with pytest.raises(ValidationError, match="max_image_bytes"):
        embedder.embed_media([frame(str(path))])
    with pytest.raises(ValidationError, match="max_batch_bytes"):
        embedder.embed_text(["a" * 301])
    assert transport.requests == []


def test_ros_factory_uses_native_gemini_endpoint_and_dimension():
    embedder = build_embedder("", "", "key", 0, backend="gemini")
    assert isinstance(embedder, GeminiEmbedder) and embedder.dimension == 768
    assert embedder._endpoint.url.startswith(GEMINI_BASE_URL)
    preview = build_embedder(
        "https://proxy.example/v1beta", "gemini-embedding-2-preview", "key", 1536, backend="gemini"
    )
    assert "preview" in preview.model_name and preview.dimension == 1536
    assert preview._endpoint.url.startswith("https://proxy.example/v1beta")


def test_evaluation_command_defaults_to_gemini_without_loading_clip(tmp_path, hashing, monkeypatch):
    pytest.importorskip("lancedb")
    from placecell.store.lancedb_store import LanceDBStore

    store = LanceDBStore(tmp_path, CollectionInfo("recording", hashing.model_name, hashing.dimension))
    memory = dual(hashing, "printer", "desk")
    store.upsert([memory])
    store.close()
    labels, output = tmp_path / "queries.json", tmp_path / "report.json"
    labels.write_text(json.dumps([{"query": "printer", "relevant_ids": [memory.id]}]))
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return hashing

    def no_clip(*args, **kwargs):
        raise AssertionError("Gemini evaluation must not initialize CLIP")

    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("placecell.providers.gemini.GeminiEmbedder", factory)
    monkeypatch.setattr("placecell.providers.clip.ClipEmbedder", no_clip)
    main(["--db-path", str(tmp_path), "--collection", "recording", "--queries", str(labels), "--output", str(output)])
    assert calls == [(("gemini-embedding-2",), {"api_key": "fake-test-key", "dimension": 768})]
    assert json.loads(output.read_text())["query_count"] == 1
