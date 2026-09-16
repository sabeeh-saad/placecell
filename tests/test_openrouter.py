import json

import numpy as np
import pytest

from placecell.errors import ProviderError, ValidationError
from placecell.providers.object_detection import ChatObjectDetector
from placecell.providers.openrouter import OpenRouterGeminiEmbedder
from placecell.ros2.node import build_embedder
from tests.conftest import FakeTransport, frame


def response(indices=(0,)):
    return 200, {}, {"data": [{"index": i, "embedding": [3, 4, *([0] * 766)]} for i in indices]}


def test_independent_images_documents_and_queries_share_normalized_space(tmp_path):
    png = tmp_path / "camera.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    transport = FakeTransport([response((1, 0)), response(), response()])
    embedder = OpenRouterGeminiEmbedder(api_key="test", transport=transport)
    vectors = embedder.embed_text(["printer", "table"])
    embedder.embed_queries(["printer"])
    embedder.embed_media([frame(str(png))])
    assert vectors.shape == (2, 768) and np.allclose(np.linalg.norm(vectors, axis=1), 1)
    first, query, image = transport.requests
    assert first["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert first["headers"] == {"Authorization": "Bearer test"}
    assert first["payload"]["dimensions"] == 768
    assert first["payload"]["input"] == [
        {"content": [{"type": "text", "text": "title: none | text: printer"}]},
        {"content": [{"type": "text", "text": "title: none | text: table"}]},
    ]
    assert query["payload"]["input"][0]["content"][0]["text"] == "task: search result | query: printer"
    assert image["payload"]["input"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert embedder.model_name == "openrouter:google/gemini-embedding-2:768:retrieval-v1"
    assert embedder.embed_media([]).shape == (0, 768)
    built = build_embedder("", "", "test", 768, backend="openrouter")
    assert built.model_name == embedder.model_name and built.capabilities.image


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": []},
        response((1,))[2],
        response((0, 0))[2],
        {"data": [{"index": 0, "embedding": [0] * 768}]},
        {"data": [{"index": 0, "embedding": [float("nan")] * 768}]},
        {"data": [{"index": 0, "embedding": [1, 2]}]},
    ],
)
def test_router_rejects_malformed_or_misindexed_embeddings(body):
    embedder = OpenRouterGeminiEmbedder(api_key="test", transport=FakeTransport([(200, {}, body)]))
    with pytest.raises(ProviderError):
        embedder.embed_queries(["printer"])


def test_router_rejects_an_unqualified_or_non_multimodal_model():
    for model in ("gemini-embedding-2", "google/gemini-embedding-001"):
        with pytest.raises(ValidationError):
            OpenRouterGeminiEmbedder(model, api_key="test")


def test_chat_detector_uploads_images_and_preserves_detection_validation(tmp_path):
    png = tmp_path / "frame.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    objects = [{"label": "printer", "description": "white with a blue display", "box_2d": [200, 300, 400, 500]}]
    transport = FakeTransport(
        [(200, {}, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(objects)}}]})]
    )
    detector = ChatObjectDetector(
        "google/gemini-2.5-flash", api_key="test", base_url="https://openrouter.ai/api/v1", transport=transport
    )
    assert detector.detect(frame(str(png)))[0].label == "printer"
    call = transport.requests[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"] == {"Authorization": "Bearer test"}
    assert call["payload"]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert call["payload"]["response_format"]["json_schema"]["schema"]["type"] == "array"


@pytest.mark.parametrize(
    "choice", [{}, {"finish_reason": "length"}, {"finish_reason": "stop", "message": {"content": "bad"}}]
)
def test_incomplete_chat_detection_is_never_an_empty_scene(tmp_path, choice):
    png = tmp_path / "frame.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    detector = ChatObjectDetector(
        "vision",
        api_key="test",
        base_url="http://localhost",
        transport=FakeTransport([(200, {}, {"choices": [choice]})]),
    )
    with pytest.raises(ProviderError):
        detector.detect(frame(str(png)))


def test_chat_detector_requires_configuration():
    with pytest.raises(ValidationError):
        ChatObjectDetector("", api_key="", base_url="http://localhost")
