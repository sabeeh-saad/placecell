from __future__ import annotations

import pytest

from placecell import (
    CollectionInfo,
    Evidence,
    EvidenceKind,
    Ingester,
    InMemoryStore,
    Observation,
    Pose,
    Recall,
    SegmentationPolicy,
    Segmenter,
)
from placecell.errors import ModelMismatchError, ValidationError
from placecell.providers import HashingEmbedder
from tests.conftest import DIM, FakeCaptioner, FakeMediaEmbedder, frame


def obs(t: float, x: float = 0.0, yaw: float = 0.0, camera: str = "front", uri: str | None = None) -> Observation:
    evidence = frame(uri or f"frames/{camera}_{round(t * 1000)}.jpg", digest=f"d{t}{camera}")
    return Observation("r1", camera, t, Pose(x, 0, yaw), evidence)


def test_segmenter_needs_time_and_motion() -> None:
    s = Segmenter(SegmentationPolicy(min_interval_s=2, min_travel_m=0.5, min_turn_rad=0.3))
    assert s.accept(obs(0))
    assert not s.accept(obs(1, x=5))  # too soon
    assert not s.accept(obs(3, x=0.2))  # did not move
    assert s.accept(obs(3, x=0.6))  # moved
    assert s.accept(obs(6, x=0.6, yaw=0.4))  # turned
    assert s.accept(obs(6, camera="back"))  # other camera has its own state
    s.reset()
    assert s.accept(obs(0))
    with pytest.raises(ValidationError):
        SegmentationPolicy(min_interval_s=-1)


def test_ingester_captions_embeds_and_reinforces(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    captioner = FakeCaptioner("a printer")
    ingester = Ingester(hashing, store, captioner=captioner, batch_size=2)
    report = ingester.ingest([obs(0, x=0), obs(1, x=9), obs(3, x=0.4), obs(6, x=9), obs(9, x=9.2, yaw=0.5)])
    assert report.received == 5 and report.accepted == 4  # t=1 dropped by the segmenter
    assert report.inserted == 2 and report.merged == 2 and report.unsupported == 0
    assert [len(c) for c in captioner.calls] == [2, 2]  # batches of two
    assert store.count() == 2
    memories = store.query()
    assert {m.observations for m in memories} == {2}
    assert all(m.caption == "a printer" and m.model == hashing.model_name for m in memories)
    hits = Recall(store, hashing, clock=lambda: 10.0).similar("a printer")
    assert len(hits) == 2 and hits[0].similarity == pytest.approx(1.0)


def test_ingester_prefers_media_embeddings_and_falls_back_to_captions(
    media_embedder: FakeMediaEmbedder, media_store: InMemoryStore
) -> None:
    ingester = Ingester(media_embedder, media_store, captioner=FakeCaptioner())
    clip = Observation("r1", "front", 50.0, Pose(9, 0), Evidence(EvidenceKind.CLIP, "c.mp4", duration_s=3))
    report = ingester.ingest([obs(0), clip])
    assert report.inserted == 2 and report.unsupported == 0
    assert len(media_embedder.media_calls) == 1 and media_embedder.media_calls[0][0].kind is EvidenceKind.FRAME
    assert media_embedder.text_calls == [["a clip at c.mp4"]]  # the clip went through its caption


def test_ingester_reports_what_it_cannot_embed(media_embedder: FakeMediaEmbedder, media_store: InMemoryStore) -> None:
    ingester = Ingester(media_embedder, media_store)  # no captioner
    clip = Observation("r1", "front", 50.0, Pose(9, 0), Evidence(EvidenceKind.CLIP, "c.mp4", duration_s=3))
    report = ingester.ingest([obs(0), clip])
    assert report.inserted == 1 and report.unsupported == 1
    assert report.unsupported_ids == ("r1:front:50000",)
    assert media_store.count() == 1


def test_ingester_is_idempotent_across_runs(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    ingester = Ingester(hashing, store, captioner=FakeCaptioner())
    batch = [obs(0, x=0), obs(5, x=3), obs(10, x=6)]
    first = ingester.ingest(batch)
    ingester = Ingester(hashing, store, captioner=FakeCaptioner())  # fresh segmenter, same store
    second = ingester.ingest(batch)
    assert first.inserted == 3 and second.inserted == 0 and second.merged == 3
    assert store.count() == 3 and all(m.observations == 1 for m in store.query())


def test_ingester_validates_its_parts(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    with pytest.raises(ModelMismatchError):
        Ingester(HashingEmbedder(DIM * 2), store)
    with pytest.raises(ModelMismatchError):
        Ingester(hashing, InMemoryStore(CollectionInfo("x", hashing.model_name, DIM + 1)))
    with pytest.raises(ValidationError):
        Ingester(hashing, store, batch_size=0)

    class BadCaptioner:
        def caption(self, items):  # type: ignore[no-untyped-def]
            return []

    with pytest.raises(ValidationError):
        Ingester(hashing, store, captioner=BadCaptioner()).ingest([obs(0)])
