from __future__ import annotations

from pathlib import Path

import pytest

from placecell import Evidence, EvidenceKind
from placecell.errors import ProviderError, ValidationError
from placecell.providers import Captioner, OpenAICompatibleCaptioner
from placecell.providers.captioning import data_url, parse_text
from tests.conftest import FakeTransport

JPEG_HEADER = b"\xff\xd8\xff\xe0placeholder"


def _reply(text: object) -> tuple[int, dict[str, str], dict]:
    return 200, {}, {"choices": [{"message": {"role": "assistant", "content": text}}]}


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "front_1000.jpg"
    path.write_bytes(JPEG_HEADER)
    return path


def test_captioner_sends_the_image_as_a_data_url_and_returns_clean_text(image: Path) -> None:
    transport = FakeTransport([_reply("  A grey chair\n next to a window.  ")])
    captioner = OpenAICompatibleCaptioner(
        "vlm", base_url="https://x/v1", api_key="k", detail="high", transport=transport
    )
    assert isinstance(captioner, Captioner) and captioner.model_name == "vlm"
    assert captioner.caption([Evidence(EvidenceKind.FRAME, str(image))]) == ["A grey chair next to a window."]
    request = transport.requests[0]
    assert request["url"] == "https://x/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer k"
    parts = request["payload"]["messages"][0]["content"]
    assert parts[0]["type"] == "text" and "robot" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/")
    assert parts[1]["image_url"]["detail"] == "high"
    assert request["payload"]["temperature"] == 0


def test_captioner_skips_clips_and_keeps_order(image: Path) -> None:
    transport = FakeTransport([_reply("first"), _reply("second")])
    captioner = OpenAICompatibleCaptioner("vlm", transport=transport)
    items = [
        Evidence(EvidenceKind.FRAME, f"file://{image}"),
        Evidence(EvidenceKind.CLIP, "c.mp4", duration_s=2),
        Evidence(EvidenceKind.FRAME, str(image)),
    ]
    assert captioner.caption(items) == ["first", "", "second"]
    assert len(transport.requests) == 2


def test_captioner_handles_content_parts_and_rejects_bad_replies(image: Path) -> None:
    captioner = OpenAICompatibleCaptioner(
        "vlm", transport=FakeTransport([_reply([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])])
    )
    assert captioner.caption([Evidence(EvidenceKind.FRAME, str(image))]) == ["a b"]
    with pytest.raises(ProviderError, match="malformed"):
        parse_text({"choices": []})
    with pytest.raises(ProviderError, match="malformed"):
        parse_text({"choices": [{"message": {"content": 42}}]})
    with pytest.raises(ProviderError, match="quota"):
        OpenAICompatibleCaptioner("vlm", transport=FakeTransport([(402, {}, {"error": {"message": "quota"}})])).caption(
            [Evidence(EvidenceKind.FRAME, str(image))]
        )


def test_data_url_only_reads_local_image_files(tmp_path: Path, image: Path) -> None:
    assert data_url(str(image)).startswith("data:image/jpeg;base64,")
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValidationError):
        data_url(str(tmp_path / "notes.txt"))
    with pytest.raises(ProviderError):
        data_url(str(tmp_path / "missing.png"))


def test_captioner_validates_arguments() -> None:
    with pytest.raises(ValidationError):
        OpenAICompatibleCaptioner("")
    with pytest.raises(ValidationError):
        OpenAICompatibleCaptioner("vlm", prompt="  ")
    with pytest.raises(ValidationError):
        OpenAICompatibleCaptioner("vlm", max_tokens=0)
    with pytest.raises(ValidationError):
        OpenAICompatibleCaptioner("vlm", timeout_s=0)
