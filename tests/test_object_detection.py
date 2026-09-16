import json

import pytest
from PIL import Image

from placecell.depth import Box
from placecell.errors import ProviderError, ValidationError
from placecell.memory import Evidence, EvidenceKind
from placecell.providers.object_detection import GeminiObjectDetector
from tests.conftest import FakeTransport


def response(value, finish="STOP"):
    return 200, {}, {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": json.dumps(value)}]}}]}


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (20, 20), "red").save(path)
    return Evidence(EvidenceKind.FRAME, str(path))


def test_native_detection_uploads_pixels_and_validates_bounds(image):
    transport = FakeTransport(
        [response([{"label": "printer", "description": "red printer", "box_2d": [100, 200, 500, 600]}])]
    )
    detector = GeminiObjectDetector("vision-test", api_key="test-key", transport=transport)
    detection = detector.detect(image)[0]
    assert detection.box == Box(0.2, 0.1, 0.6, 0.5)
    request = transport.requests[0]
    assert request["url"].endswith("/models/vision-test:generateContent")
    assert request["headers"]["x-goog-api-key"] == "test-key"
    assert "Authorization" not in request["headers"]
    assert "inlineData" in request["payload"]["contents"][0]["parts"][0]
    assert request["payload"]["generationConfig"]["responseJsonSchema"]["type"] == "array"


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [{"label": 2}],
        [{"label": "x", "description": "x", "box_2d": [0, 0, 0, 100]}],
        [{"label": "x", "description": "x", "box_2d": [0, 0, 1001, 100]}],
        [{"label": "x", "description": "x", "box_2d": [False, 0, 100, 100]}],
    ],
)
def test_malformed_detection_is_an_error_not_an_empty_scene(image, value):
    detector = GeminiObjectDetector("vision-test", api_key="test", transport=FakeTransport([response(value)]))
    with pytest.raises(ProviderError):
        detector.detect(image)


def test_truncated_output_does_not_count_as_absence(image):
    detector = GeminiObjectDetector(
        "vision-test", api_key="test", transport=FakeTransport([response([], "MAX_TOKENS")])
    )
    with pytest.raises(ProviderError):
        detector.detect(image)


@pytest.mark.parametrize(
    "verdict,expected", [("absent", True), ("present", False), ("occluded", False), ("uncertain", False)]
)
def test_absence_requires_explicit_visual_evidence(image, verdict, expected):
    transport = FakeTransport([response({"result": verdict})])
    detector = GeminiObjectDetector("vision-test", api_key="test", transport=transport)
    assert detector.absent(b"crop", image, Box(0.2, 0.2, 0.4, 0.4)) is expected
    assert len(transport.requests[0]["payload"]["contents"][0]["parts"]) == 2


def test_invalid_absence_and_config_are_rejected(image):
    detector = GeminiObjectDetector(
        "vision-test", api_key="test", transport=FakeTransport([response({"result": "yes"})])
    )
    with pytest.raises(ProviderError):
        detector.absent(b"crop", image, Box(0.2, 0.2, 0.4, 0.4))
    with pytest.raises(ValidationError):
        GeminiObjectDetector("../model", api_key="test")


def test_image_type_size_and_missing_file_fail_before_api(image):
    from pathlib import Path

    detector = GeminiObjectDetector("vision-test", api_key="test", transport=FakeTransport([]))
    path = Path(image.uri)
    path.write_bytes(b"not an image")
    with pytest.raises(ValidationError, match="PNG"):
        detector.detect(image)
    path.write_bytes(b"x" * 8_000_001)
    with pytest.raises(ValidationError, match="limit"):
        detector.detect(image)
    path.unlink()
    with pytest.raises(ProviderError, match="cannot read"):
        detector.detect(image)


@pytest.mark.parametrize("verdict", ["matched", "not_matched", "uncertain"])
def test_instance_comparison_sends_saved_crops_before_the_candidate(image, verdict):
    import base64
    from pathlib import Path

    crop = Path(image.uri).read_bytes()
    transport = FakeTransport([response({"result": verdict, "reason": "Visible marks compared."})])
    detector = GeminiObjectDetector("vision-test", api_key="test", transport=transport)
    assert detector.compare((crop, crop), crop).result == verdict
    request = transport.requests[0]["payload"]
    parts = request["contents"][0]["parts"]
    assert len(parts) == 3 and base64.b64decode(parts[-1]["inlineData"]["data"]) == crop
    assert "instance-specific" in request["systemInstruction"]["parts"][0]["text"]


@pytest.mark.parametrize(
    "value", [{"result": "yes", "reason": "x"}, {"result": "matched"}, {"result": "matched", "reason": ""}]
)
def test_bad_instance_comparison_fails_closed(image, value):
    from pathlib import Path

    crop = Path(image.uri).read_bytes()
    detector = GeminiObjectDetector("vision-test", api_key="test", transport=FakeTransport([response(value)]))
    with pytest.raises(ProviderError):
        detector.compare((crop,), crop)
    with pytest.raises(ValidationError):
        detector.compare((), crop)
