from __future__ import annotations

from pathlib import Path

import pytest

from placecell import (
    CollectionInfo,
    Curator,
    Evidence,
    EvidenceKind,
    Filter,
    Ingester,
    InMemoryStore,
    Observation,
    Pose,
    Recall,
    ReinforcementPolicy,
    Reinforcer,
    SegmentationPolicy,
    Segmenter,
)
from placecell.errors import ModelMismatchError, ProviderError, ValidationError
from placecell.lifecycle import remove_local_file
from placecell.providers import HashingEmbedder
from placecell.ros2.bridge import KeyframeWriter, ObservationBuilder
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


def test_caption_failure_can_be_retried_on_the_same_ingester(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    class FailOnce(FakeCaptioner):
        def caption(self, items):
            if not self.calls:
                self.calls.append(list(items))
                raise ProviderError("temporary failure")
            return super().caption(items)

    ingester = Ingester(hashing, store, FailOnce("printer"))
    observations = [obs(100), obs(200, x=0.4)]
    with pytest.raises(ProviderError):
        ingester.ingest(observations)
    report = ingester.ingest(observations)
    assert report.accepted == 2 and report.inserted == 1 and report.merged == 1
    assert store.query()[0].observations == 2


def test_retry_preserves_completed_batches_and_deduplicates_partial_writes(
    store: InMemoryStore,
    hashing: HashingEmbedder,
) -> None:
    class FailThird(Reinforcer):
        calls = 0

        def reinforce_or_insert(self, memory):
            result = super().reinforce_or_insert(memory)
            self.calls += 1
            if self.calls == 3:
                raise ProviderError("write succeeded before the connection failed")
            return result

    ingester = Ingester(hashing, store, FakeCaptioner("printer"), reinforcer=FailThird(store), batch_size=2)
    observations = [obs(100), obs(200, x=0.4), obs(300), obs(400, x=0.4)]
    with pytest.raises(ProviderError):
        ingester.ingest(observations)
    assert store.query()[0].observations == 3
    ingester.ingest(observations)
    assert store.query()[0].observations == 4
    assert store.query()[0].sighting_times == (100, 200, 300, 400)


def test_source_failure_does_not_advance_unflushed_segmentation(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    def source():
        yield obs(100)
        raise ProviderError("source interrupted")

    ingester = Ingester(hashing, store, FakeCaptioner("printer"))
    with pytest.raises(ProviderError):
        ingester.ingest(source())
    assert ingester.ingest([obs(100)]).inserted == 1


def test_rejected_generated_frames_are_removed_but_pending_frames_survive(
    store: InMemoryStore,
    hashing: HashingEmbedder,
    tmp_path: Path,
) -> None:
    builder = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path))
    observations = [builder.from_compressed(t, "jpeg", b"image", Pose(0, 0)) for t in range(10)]
    ingester = Ingester(hashing, store, FakeCaptioner("printer"), batch_size=2)
    report = ingester.ingest([observations[0], observations[0], *observations[1:]])
    assert report.accepted == 1 and len(list(tmp_path.glob("*.jpg"))) == 1
    stored = store.query()[0]
    assert stored.evidence is not None and Path(stored.evidence.uri).exists()
    Curator(store, remover=remove_local_file).forget(Filter())
    assert not list(tmp_path.glob("*.jpg"))


def test_unsupported_generated_frames_are_cleaned_without_removing_caller_files(
    store: InMemoryStore,
    hashing: HashingEmbedder,
    tmp_path: Path,
) -> None:
    builder = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path))
    managed = builder.from_compressed(100, "jpeg", b"generated", Pose(0, 0))
    original = tmp_path / "original.jpg"
    original.write_bytes(b"original")
    external = obs(200, x=4, uri=str(original))
    report = Ingester(hashing, store).ingest([managed, external])
    assert report.unsupported == 2
    assert not Path(managed.evidence.uri).exists() and original.read_bytes() == b"original"


@pytest.mark.parametrize("keep_newest", [False, True])
def test_reinforcement_removes_only_replaced_generated_keyframes(
    store: InMemoryStore,
    hashing: HashingEmbedder,
    tmp_path: Path,
    keep_newest: bool,
) -> None:
    builder = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path))
    first = builder.from_compressed(100, "jpeg", b"first", Pose(0, 0))
    repeat = builder.from_compressed(200, "jpeg", b"repeat", Pose(0.4, 0))
    reinforcer = Reinforcer(store, ReinforcementPolicy(keep_newest_evidence=keep_newest))
    Ingester(hashing, store, FakeCaptioner("printer"), reinforcer=reinforcer).ingest([first, repeat])
    kept, removed = (repeat, first) if keep_newest else (first, repeat)
    assert Path(kept.evidence.uri).exists() and not Path(removed.evidence.uri).exists()
    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_failed_ingestion_keeps_generated_evidence_for_retry(
    store: InMemoryStore, hashing: HashingEmbedder, tmp_path: Path
) -> None:
    class FailOnce(FakeCaptioner):
        failed = False

        def caption(self, items):
            if not self.failed:
                self.failed = True
                raise ProviderError("temporary failure")
            assert all(Path(item.uri).exists() for item in items)
            return super().caption(items)

    observation = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path)).from_compressed(
        100, "jpeg", b"image", Pose(0, 0)
    )
    ingester = Ingester(hashing, store, FailOnce("printer"))
    with pytest.raises(ProviderError):
        ingester.ingest([observation])
    assert Path(observation.evidence.uri).exists()
    assert ingester.ingest([observation]).inserted == 1


def test_partial_retry_does_not_read_keyframes_already_replaced(
    store: InMemoryStore,
    hashing: HashingEmbedder,
    tmp_path: Path,
) -> None:
    class ReadingCaptioner(FakeCaptioner):
        def caption(self, items):
            for item in items:
                assert Path(item.uri).read_bytes() == b"image"
            return super().caption(items)

    class FailSecond(Reinforcer):
        calls = 0

        def reinforce_or_insert(self, memory):
            result = super().reinforce_or_insert(memory)
            self.calls += 1
            if self.calls == 2:
                raise ProviderError("write succeeded before response failed")
            return result

    builder = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path))
    observations = [builder.from_compressed(100 + i * 100, "jpeg", b"image", Pose((i % 2) * 0.4, 0)) for i in range(3)]
    captioner = ReadingCaptioner("printer")
    ingester = Ingester(hashing, store, captioner, reinforcer=FailSecond(store), batch_size=3)
    with pytest.raises(ProviderError):
        ingester.ingest(observations)
    assert not Path(observations[0].evidence.uri).exists()
    assert ingester.ingest(observations).merged == 3
    assert [len(items) for items in captioner.calls] == [3, 1]
    assert store.query()[0].observations == 3
    assert len(list(tmp_path.glob("*.jpg"))) == 1
