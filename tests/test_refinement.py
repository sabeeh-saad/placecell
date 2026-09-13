from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from placecell import (
    CollectionInfo,
    ConsolidationPolicy,
    Consolidator,
    Evidence,
    EvidenceKind,
    MemoryRefiner,
    RefinementPolicy,
    Reinforcer,
)
from placecell.errors import ModelMismatchError, ProviderError, ValidationError
from placecell.providers import Capabilities, HashingEmbedder
from placecell.store.state import StateStore
from tests.conftest import FakeCaptioner, embedded
from tests.test_consolidation import JoinSummarizer


def test_new_evidence_refines_caption_and_vector_without_creating_a_sighting(store, hashing):
    old = embedded(hashing, "cabinet", confidence=0.4)
    store.upsert([old])
    assert store.refinements.pending() == []  # ingestion already captioned this image
    fresh = embedded(hashing, "cabinet", t=2000)
    before, merged = Reinforcer(store).reinforce_or_insert(fresh)
    assert merged and before.caption == "cabinet" and before.evidence == fresh.evidence
    assert store.refinements.pending()[0]["reason"] == "new evidence"
    captioner = FakeCaptioner("  A red fire equipment cabinet.  ")
    refiner = MemoryRefiner(store, hashing, captioner, clock=lambda: 3000, producer="vision-v2")
    report = refiner.run()
    after = store.get(old.id)
    assert report.attempted == report.updated == 1
    assert after == replace(before, caption="A red fire equipment cabinet.", embedding=after.embedding)
    assert np.allclose(after.embedding, hashing.embed_text([after.caption])[0])
    assert captioner.calls == [[fresh.evidence]]
    assert store.sightings(old.id) == before.sightings
    assert refiner.run().attempted == 0
    revision = store.refinements.history(old.id)[0]
    assert revision.before_caption == "cabinet" and revision.after_caption == after.caption
    assert revision.producer == "vision-v2" and revision.timestamp == 3000
    assert refiner.rollback(old.id)
    restored = store.get(old.id)
    assert restored == before and np.array_equal(restored.embedding, before.embedding)
    assert store.refinements.history(old.id)[0].rolled_back
    assert not refiner.rollback(old.id)
    assert refiner.run().attempted == 0


def test_same_pixels_do_not_request_another_refinement(store, hashing):
    old = embedded(hashing, "cabinet", evidence=Evidence(EvidenceKind.FRAME, "a.jpg", "same-digest"))
    store.upsert([old])
    store.upsert([replace(old, evidence=replace(old.evidence, uri="b.jpg"), last_seen=2000)])
    assert store.refinements.pending() == []
    store.upsert([replace(old, evidence=replace(old.evidence, digest="changed"))])
    assert len(store.refinements.pending()) == 1


def test_captionless_memory_is_queued_but_summaries_and_superseded_rows_are_excluded(store, hashing):
    old = embedded(hashing, "")
    store.upsert([old])
    assert len(store.refinements.pending()) == 1
    store.upsert([replace(old, superseded=True)])
    assert store.refinements.pending() == []
    assert not store.refinements.request(old.id)
    store.upsert([replace(old, role="summary")])
    assert not store.refinements.request(old.id)
    store.upsert([replace(old, evidence=None, view_timestamp=None)])
    assert not store.refinements.request(old.id)
    assert not store.refinements.request("missing")


def test_refinement_keeps_media_embedding_semantics(media_store, media_embedder):
    old = embedded(media_embedder, "cabinet")
    media_store.upsert([old])
    media_store.refinements.request(old.id)
    before_text_calls = len(media_embedder.text_calls)
    refiner = MemoryRefiner(media_store, media_embedder, FakeCaptioner("fire equipment"))
    assert refiner.run().updated == 1
    assert media_embedder.media_calls == [[old.evidence]]
    assert len(media_embedder.text_calls) == before_text_calls + 1
    assert media_embedder.text_calls[-1] == ["fire equipment"]
    assert np.allclose(media_store.get(old.id).caption_embedding, media_embedder.embed_text(["fire equipment"])[0])
    assert np.allclose(media_store.get(old.id).embedding, media_embedder.embed_media([old.evidence])[0])


def test_successful_recheck_with_identical_caption_and_vector_is_acknowledged(store, hashing):
    old = embedded(hashing, "cabinet")
    store.upsert([old])
    store.refinements.request(old.id)
    refiner = MemoryRefiner(store, hashing, FakeCaptioner("cabinet"))
    assert refiner.run().unchanged == 1
    assert store.refinements.history(old.id) == []
    assert refiner.run().attempted == 0


def test_refinement_invalidates_derived_summaries(store, hashing):
    rows = [embedded(hashing, "cabinet", t=t) for t in (1000, 2000)]
    store.upsert(rows)
    Consolidator(store, hashing, JoinSummarizer(), ConsolidationPolicy(min_group=2)).run()
    summary_id = store.get(rows[0].id).consolidated_into
    store.refinements.request(rows[0].id)
    assert MemoryRefiner(store, hashing, FakeCaptioner("fire equipment")).run().updated == 1
    assert store.get(summary_id).superseded
    assert all(not store.get(m.id).consolidated_into for m in rows)


def test_slow_refinement_does_not_block_writes_or_replace_newer_evidence(store, hashing):
    old = embedded(hashing, "cabinet")
    store.upsert([old])
    store.refinements.request(old.id)
    entered, release = Event(), Event()

    class BlockingCaptioner(FakeCaptioner):
        def caption(self, items):
            entered.set()
            assert release.wait(5)
            return super().caption(items)

    refiner = MemoryRefiner(store, hashing, BlockingCaptioner("stale description"), RefinementPolicy(max_memories=1))
    fresh = replace(old, evidence=replace(old.evidence, uri="fresh.jpg"), last_seen=2000)
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(refiner.run)
        try:
            assert entered.wait(5)
            assert refiner.run().attempted == 0
            assert pool.submit(store.upsert, [fresh]).result(timeout=2) == 1
        finally:
            release.set()
        assert running.result(timeout=5).deferred == 1
    assert store.get(old.id) == fresh and store.refinements.history(old.id) == []
    assert MemoryRefiner(store, hashing, FakeCaptioner("fresh description")).run().updated == 1


@pytest.mark.parametrize("change", ["vector", "delete", "request"])
def test_concurrent_changes_cancel_the_old_result(store, hashing, change):
    old = embedded(hashing, "cabinet")
    store.upsert([old])
    store.refinements.request(old.id)

    class ChangingCaptioner(FakeCaptioner):
        def caption(self, items):
            if change == "vector":
                store.upsert([replace(old, embedding=hashing.embed_text(["door"])[0])])
            elif change == "delete":
                store.delete([old.id])
            else:
                store.refinements.request(old.id, "operator correction")
            return ["stale description"]

    report = MemoryRefiner(store, hashing, ChangingCaptioner(), RefinementPolicy(max_memories=1)).run()
    assert report.deferred == 1 and report.updated == 0
    assert store.refinements.history(old.id) == []


def test_failed_providers_back_off_and_exhaust_their_attempt_budget(store, hashing):
    old = embedded(hashing, "cabinet")
    store.upsert([old])
    store.refinements.request(old.id)
    clock = [1000.0]

    class FailingCaptioner(FakeCaptioner):
        def caption(self, items):
            raise ProviderError("unavailable")

    refiner = MemoryRefiner(
        store, hashing, FailingCaptioner(), RefinementPolicy(max_attempts=2, retry_delay_s=5), clock=lambda: clock[0]
    )
    assert refiner.run().failed == 1
    assert refiner.run().attempted == 0
    clock[0] += 6
    assert refiner.run().failed == 1
    clock[0] += 6
    assert refiner.run().attempted == 0
    pending = store.refinements.pending()[0]
    assert pending["attempts"] == 2 and pending["error"] == "unavailable"
    assert store.get(old.id) == old
    assert store.refinements.request(old.id)
    assert MemoryRefiner(store, hashing, FakeCaptioner("updated")).run().updated == 1


@pytest.mark.parametrize("captions", [[], ["", "extra"], [None], ["   "], ["x" * 2001]])
def test_invalid_captions_leave_the_memory_unchanged(store, hashing, captions):
    old = embedded(hashing, "cabinet")
    store.upsert([old])
    store.refinements.request(old.id)

    class InvalidCaptioner(FakeCaptioner):
        def caption(self, items):
            return captions

    assert MemoryRefiner(store, hashing, InvalidCaptioner()).run().failed == 1
    assert store.get(old.id) == old and store.refinements.history(old.id) == []


def test_revision_and_caption_updates_are_one_transaction(store, hashing, monkeypatch):
    old = embedded(hashing, "cabinet")
    store.upsert([old])
    store.refinements.request(old.id)

    def fail_record(*args):
        raise OSError("disk full")

    monkeypatch.setattr(store.refinements, "record", fail_record)
    report = MemoryRefiner(store, hashing, FakeCaptioner("fire equipment")).run()
    assert report.failed == 1 and report.updated == 0
    assert store.get(old.id) == old and np.array_equal(store.get(old.id).embedding, old.embedding)
    assert store.refinements.pending()[0]["error"] == "disk full"


def test_pass_and_revision_limits_and_undo_do_not_change_lifecycle_state(store, hashing):
    rows = [embedded(hashing, "cabinet", t=t) for t in range(3)]
    store.upsert(rows)
    for row in rows:
        store.refinements.request(row.id)
    refiner = MemoryRefiner(store, hashing, FakeCaptioner("fire equipment"), RefinementPolicy(max_memories=2))
    assert refiner.run().attempted == 2
    assert refiner.run().attempted == 1
    for text in ("red cabinet", "equipment storage", "fire equipment cabinet"):
        store.refinements.request(rows[0].id)
        assert MemoryRefiner(store, hashing, FakeCaptioner(text), RefinementPolicy(keep_revisions=2)).run().updated == 1
    assert len(store.refinements.history(rows[0].id, limit=100)) == 2
    current = store.get(rows[0].id)
    store.upsert([replace(current, confidence=0.2, misses=2, last_miss=2000)])
    store.refinements.request(rows[0].id)
    assert refiner.rollback(rows[0].id)
    restored = store.get(rows[0].id)
    assert restored.confidence == 0.2 and restored.misses == 2 and restored.last_miss == 2000
    assert store.refinements.pending() == []
    assert not refiner.rollback("missing")
    store.delete([rows[0].id])
    assert store.refinements.history(rows[0].id) == []


@pytest.mark.parametrize("change", ["caption", "vector", "evidence"])
def test_undo_refuses_to_overwrite_later_content(store, hashing, change):
    old = embedded(hashing, "cabinet")
    store.upsert([old])
    store.refinements.request(old.id)
    refiner = MemoryRefiner(store, hashing, FakeCaptioner("fire equipment"))
    assert refiner.run().updated == 1
    current = store.get(old.id)
    fields = {
        "caption": {"caption": "operator description"},
        "vector": {"embedding": hashing.embed_text(["door"])[0]},
        "evidence": {"evidence": replace(current.evidence, uri="new.jpg")},
    }[change]
    store.upsert([replace(current, **fields)])
    assert not refiner.rollback(old.id)


def test_requests_attempts_and_revisions_survive_restart(tmp_path, hashing):
    path = tmp_path / "state.sqlite3"
    info = CollectionInfo("test", hashing.model_name, hashing.dimension)
    store = StateStore(info, path)
    old = embedded(hashing, "cabinet")
    store.upsert([old])
    store.refinements.request(old.id)
    job = store.refinements.claim(3, 5, 1000)
    assert job.attempt == 1  # process dies before completing its reserved attempt
    store.close()
    store = StateStore(info, path)
    refiner = MemoryRefiner(store, hashing, FakeCaptioner("fire equipment"), clock=lambda: 1001)
    assert refiner.run().attempted == 0
    refiner = MemoryRefiner(store, hashing, FakeCaptioner("fire equipment"), clock=lambda: 1006)
    assert refiner.run().updated == 1
    store.close()
    store = StateStore(info, path)
    refiner = MemoryRefiner(store, hashing, FakeCaptioner("ignored"))
    assert store.get(old.id).caption == "fire equipment"
    assert refiner.run().attempted == 0
    assert refiner.rollback(old.id)
    assert store.get(old.id) == old
    store.close()


def test_refinement_validates_configuration(store, hashing):
    for kwargs in (
        {"max_memories": 0},
        {"keep_revisions": 0},
        {"max_attempts": 1.5},
        {"retry_delay_s": 0},
        {"retry_delay_s": float("nan")},
    ):
        with pytest.raises(ValidationError):
            RefinementPolicy(**kwargs)
    with pytest.raises(ModelMismatchError):
        MemoryRefiner(store, HashingEmbedder(8), FakeCaptioner())
    with pytest.raises(ValidationError):
        MemoryRefiner(store, hashing, FakeCaptioner(), producer=" ")
    with pytest.raises(ValidationError):
        MemoryRefiner(store, hashing, FakeCaptioner(), clock=lambda: float("nan")).run()
    with pytest.raises(ValidationError):
        store.refinements.request("missing", " ")
    with pytest.raises(ValidationError):
        store.refinements.pending(0)
    with pytest.raises(ValidationError):
        store.refinements.history("missing", 0)


def test_unsupported_embedding_and_invalid_vectors_leave_memory_intact(media_store, media_embedder):
    old = embedded(media_embedder, "cabinet")
    media_store.upsert([old])
    media_store.refinements.request(old.id)
    media_embedder.capabilities = Capabilities(text=False)
    refiner = MemoryRefiner(media_store, media_embedder, FakeCaptioner("updated"))
    assert refiner.run().failed == 1
    media_embedder.capabilities = Capabilities(text=True)
    media_embedder.embed_text = lambda texts: np.full((1, media_embedder.dimension), float("nan"))
    media_store.refinements.request(old.id)
    assert refiner.run().failed == 1
    media_embedder.embed_text = lambda texts: np.zeros((1, media_embedder.dimension))
    media_store.refinements.request(old.id)
    assert refiner.run().failed == 1
    assert media_store.get(old.id) == old
