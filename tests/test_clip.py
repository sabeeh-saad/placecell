import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from placecell import Evidence, EvidenceKind
from placecell.errors import PlacecellError, ProviderError, UnsupportedMediaError, ValidationError
from placecell.providers import ClipEmbedder
from placecell.providers.clip import DEFAULT_CLIP_MODEL
from placecell.ros2.node import build_embedder
from tests.conftest import frame


@pytest.fixture
def backend(monkeypatch):
    loads, batches = [], []

    class Encoder:
        def __init__(self, *args, **kwargs):
            loads.append((args, kwargs))

        def encode(self, items, **kwargs):
            batches.append((items, kwargs))
            for item in items:
                if isinstance(item, Image.Image):
                    assert item.mode == "RGB" and item.getpixel((0, 0)) is not None
            result = np.zeros((len(items), 512), dtype=np.float32)
            result[:, 0] = 3
            result[:, 1] = 4
            return result

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    return loads, batches, Encoder


def test_clip_loads_once_batches_frames_and_text_in_the_same_space(backend, tmp_path):
    loads, batches, _ = backend
    path = tmp_path / "frame.png"
    Image.new("RGBA", (4, 2), (255, 0, 0, 120)).save(path)
    embedder = ClipEmbedder(revision="pinned-version", cache_folder=str(tmp_path), local_files_only=True, batch_size=2)
    assert embedder.capabilities.text and embedder.capabilities.image and not embedder.capabilities.video
    assert embedder.dimension == 512 and embedder.model_name.endswith("@pinned-version")
    assert embedder.embed_text([]).shape == (0, 512)
    assert embedder.embed_media([]).shape == (0, 512) and not loads
    media = embedder.embed_media([frame(str(path)), frame(f"file://{path}"), frame(str(path))])
    text = embedder.embed_text(["printer", "chair", "desk"])
    assert len(loads) == 1 and [len(items) for items, _ in batches] == [2, 1, 2, 1]
    assert np.allclose(media, text) and np.allclose(np.linalg.norm(media, axis=1), 1)
    assert loads[0] == (
        (DEFAULT_CLIP_MODEL,),
        {
            "revision": "pinned-version",
            "device": "cpu",
            "cache_folder": str(tmp_path),
            "local_files_only": True,
            "trust_remote_code": False,
        },
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: embedder.embed_text(["printer"]), range(8)))
    assert len(loads) == 1 and all(r.shape == (1, 512) for r in results)


def test_clip_config_and_unsupported_inputs_fail_explicitly(backend, tmp_path):
    for kwargs in ({"model": "text-only-model"}, {"batch_size": 0}, {"device": ""}, {"revision": ""}):
        with pytest.raises(ValidationError):
            ClipEmbedder(**kwargs)
    assert ClipEmbedder("sentence-transformers/clip-ViT-L-14").dimension == 768
    assert ClipEmbedder("sentence-transformers/clip-ViT-B-16").dimension == 512
    embedder = ClipEmbedder()
    for evidence in (Evidence(EvidenceKind.CLIP, "clip.mp4", duration_s=2), frame("https://example.com/image.jpg")):
        with pytest.raises(UnsupportedMediaError):
            embedder.embed_media([evidence])
    with pytest.raises(ProviderError, match="image encoding failed"):
        embedder.embed_media([frame(str(tmp_path / "missing.jpg"))])
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ProviderError, match="image encoding failed"):
        embedder.embed_media([frame(str(corrupt))])


@pytest.mark.parametrize("result", [np.zeros((1, 512)), np.ones((1, 5)), np.full((1, 512), float("nan"))])
def test_clip_rejects_malformed_provider_vectors(backend, monkeypatch, result):
    monkeypatch.setattr(backend[2], "encode", lambda *args, **kwargs: result)
    with pytest.raises(ProviderError):
        ClipEmbedder().embed_text(["printer"])


def test_clip_missing_extra_and_runtime_errors_are_actionable(backend, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(PlacecellError, match=r"placecell\[clip\]"):
        ClipEmbedder().embed_text(["printer"])

    def broken(*args, **kwargs):
        raise RuntimeError("unavailable device")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=broken))
    with pytest.raises(ProviderError, match="unavailable device"):
        ClipEmbedder().embed_text(["printer"])
    path = tmp_path / "frame.png"
    Image.new("RGB", (2, 2)).save(path)
    with pytest.raises(ProviderError, match="unavailable device"):
        ClipEmbedder().embed_media([frame(str(path))])
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(PlacecellError, match=r"placecell\[clip\]"):
        ClipEmbedder().embed_media([frame(str(path))])


def test_ros_factory_selects_clip_without_loading_weights():
    embedder = build_embedder("", "", None, 512, backend="clip", revision="v1", device="cpu", local_files_only=True)
    assert isinstance(embedder, ClipEmbedder) and embedder.model_name.endswith("@v1")
    with pytest.raises(ValidationError, match="dimension"):
        build_embedder("", "", None, 64, backend="clip")
    with pytest.raises(ValidationError, match="backend"):
        build_embedder("", "", None, 0, backend="unknown")
