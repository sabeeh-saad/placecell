"""Synthetic vectors test retrieval mechanics, not CLIP accuracy."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from placecell import CollectionInfo, Filter, InMemoryStore, MemoryRefiner, Pose, Recall, Reinforcer, reembed
from placecell.errors import ValidationError
from placecell.evaluation import RetrievalCase, evaluate_retrieval, load_cases, main
from placecell.providers.embedding import embed_memories
from tests.conftest import FakeCaptioner, embedded


def dual(hashing, image, caption, **kwargs):
    memory = embedded(hashing, caption, **kwargs)
    memory = memory.with_embedding(hashing.embed_text([image])[0], hashing.model_name, kind="image")
    return replace(memory, caption_embedding=hashing.embed_text([caption])[0] if caption else None)


def test_omitted_caption_is_recovered_outside_recent_candidate_window(store, hashing):
    printer = dual(hashing, "printer", "desk", t=1)
    captions = dual(hashing, "door", "printer", t=2)
    store.upsert([printer, captions, *[dual(hashing, "chair", "chair", t=t) for t in range(3, 83)]])
    recall = Recall(store, hashing, clock=lambda: 83)
    assert recall.similar("printer", k=1, mode="image")[0].memory.id == printer.id
    assert recall.similar("printer", k=1, mode="caption")[0].memory.id == captions.id
    combined = recall.similar("printer", k=2)
    assert {r.memory.id for r in combined} == {printer.id, captions.id}
    hit = next(r for r in combined if r.memory.id == printer.id)
    assert hit.image_similarity == pytest.approx(1)
    assert hit.caption_similarity < hit.image_similarity
    assert hit.similarity == hit.image_similarity


def test_channel_filters_missing_vectors_and_atomic_caption_updates(store, hashing):
    first = dual(hashing, "chair", "printer", t=1, x=0)
    image_only = dual(hashing, "printer", "", t=2, x=10)
    text_only = embedded(hashing, "printer", t=3, robot="r2")
    dead = dual(hashing, "printer", "printer", t=4, superseded=True)
    store.upsert([first, image_only, text_only, dead])
    query = hashing.embed_text(["printer"])[0]
    assert {h.memory.id for h in store.search(query, 10, channel="caption")} == {first.id, text_only.id}
    assert {h.memory.id for h in store.search(query, 10, channel="image")} == {first.id, image_only.id}
    assert [
        h.memory.id
        for h in store.search(query, 10, Filter(near=Pose(0, 0), radius=1, robot_id="r1"), channel="caption")
    ] == [first.id]
    changed = replace(first, caption="door", caption_embedding=hashing.embed_text(["door"])[0])
    with store.transaction():
        store.upsert([changed])
        assert store.search(query, 10, Filter(robot_id="r1"), channel="caption")[0].score < 0.5
    assert store.get(first.id).same_embeddings(changed)
    assert store.search(query, 10, Filter(time_from=0, time_to=2), channel="caption")[0].score < 0.5
    assert store.search(query, 10, Filter(robot_id="r1"), channel="caption")[0].score < 0.5
    with pytest.raises(ValidationError):
        store.search(query, 1, channel="unknown")
    with pytest.raises(ValidationError):
        Recall(store, hashing).similar("printer", mode="unknown")


def test_reinforcement_replaces_both_vectors_with_the_view(store, hashing):
    old = dual(hashing, "printer", "desk", t=1)
    new = dual(hashing, "printer", "paper", t=100, x=0.1)
    store.upsert([old])
    memory, merged = Reinforcer(store).reinforce_or_insert(new)
    assert merged and memory.id == old.id and memory.pose == new.pose
    assert memory.same_embeddings(new) and memory.caption == "paper"
    assert memory.evidence == new.evidence and memory.view_timestamp == new.view_timestamp


@pytest.mark.parametrize("concurrent", [False, True])
def test_refinement_rolls_back_both_channels_and_detects_caption_vector_races(media_embedder, concurrent):
    store = InMemoryStore(CollectionInfo("refine", media_embedder.model_name, media_embedder.dimension))
    before = embed_memories([embedded(media_embedder, "cabinet")], media_embedder)[0][0]
    store.upsert([before])
    store.refinements.request(before.id)

    class Captioner(FakeCaptioner):
        def caption(self, items):
            if concurrent:
                store.upsert([replace(before, caption_embedding=-before.caption_embedding)])
            return ["fire equipment"]

    refiner = MemoryRefiner(store, media_embedder, Captioner())
    report = refiner.run()
    if concurrent:
        assert report.deferred == 1 and store.get(before.id).caption == before.caption
    else:
        assert report.updated == 1
        assert not store.get(before.id).same_embeddings(before)
        assert refiner.rollback(before.id)
        assert store.get(before.id).same_embeddings(before)
        assert store.get(before.id).caption == before.caption
    store.close()


def test_rollback_does_not_overwrite_a_new_caption_vector(media_store, media_embedder):
    before = embed_memories([embedded(media_embedder, "cabinet")], media_embedder)[0][0]
    media_store.upsert([before])
    media_store.refinements.request(before.id)
    refiner = MemoryRefiner(media_store, media_embedder, FakeCaptioner("fire equipment"))
    assert refiner.run().updated == 1
    after = media_store.get(before.id)
    media_store.upsert([replace(after, caption_embedding=-after.caption_embedding)])
    assert not refiner.rollback(before.id)


def test_reembedding_populates_both_channels_and_removes_old_auxiliary_vectors(store, hashing, media_embedder):
    old = embedded(hashing, "printer", t=1)
    store.upsert([old])
    target = InMemoryStore(CollectionInfo("dual", media_embedder.model_name, media_embedder.dimension))
    assert reembed(store, target, media_embedder).written == 1
    moved = target.get(old.id)
    assert moved.embedding_kind == "image" and moved.caption_embedding is not None
    assert moved.pose == old.pose and moved.sightings == old.sightings
    assert moved.view_timestamp == old.view_timestamp and moved.localization_checked == old.localization_checked
    back = InMemoryStore(CollectionInfo("text", hashing.model_name, hashing.dimension))
    assert reembed(target, back, hashing).written == 1
    assert back.get(old.id).embedding_kind == "caption" and back.get(old.id).caption_embedding is None


def test_reembedding_a_summary_uses_its_caption_not_the_inherited_anchor_image(media_embedder):
    summary = embedded(media_embedder, "desk and printer", role="summary")
    rows, rejected = embed_memories([summary], media_embedder)
    assert not rejected and rows[0].embedding_kind == "caption"
    assert rows[0].caption_embedding is None and media_embedder.media_calls == []


def test_caption_index_recovers_reopens_and_rebuilds(tmp_path, hashing):
    pytest.importorskip("lancedb")
    from placecell.store.lancedb_store import LanceDBStore

    store = LanceDBStore(tmp_path, CollectionInfo("dual", hashing.model_name, hashing.dimension))
    memory = dual(hashing, "desk", "printer", t=1)
    store.upsert([memory])
    query = hashing.embed_text(["printer"])[0]
    assert store.search(query, 1, channel="caption")[0].score == pytest.approx(1)
    store.upsert([replace(memory, caption="door", caption_embedding=hashing.embed_text(["door"])[0])])
    with (
        patch.object(store._table, "merge_insert", side_effect=RuntimeError("interrupted")),
        pytest.raises(RuntimeError),
    ):
        store.search(query, 1, channel="caption")
    store.close()

    store = LanceDBStore.open(tmp_path, "dual")
    assert store.search(query, 1, channel="caption")[0].score < 0.5
    assert store.get(memory.id).embedding_kind == "image"
    store.rebuild_index()
    assert store.search(query, 1, channel="caption")[0].score < 0.5
    store.delete([memory.id])
    assert store.search(query, 1, channel="caption") == []
    store.close()


def test_both_vector_indexes_handle_frames_without_captions(tmp_path, hashing):
    pytest.importorskip("lancedb")
    from placecell.store.lancedb_store import LanceDBStore

    store = LanceDBStore(tmp_path, CollectionInfo("indexed", hashing.model_name, hashing.dimension))
    store.upsert([dual(hashing, "printer", "desk" if t % 2 else "", t=t) for t in range(1, 33)])
    store.maintain(vector_index_min_rows=8)
    columns = {column for index in store._table.list_indices() for column in index.columns}
    assert {"vector", "caption_vector"} <= columns
    query = hashing.embed_text(["desk"])[0]
    hits = store.search(query, 32, channel="caption")
    assert len(hits) == 16 and all(hit.memory.caption == "desk" for hit in hits)
    assert all(hit.score == pytest.approx(1) for hit in hits)
    store.close()


def test_version_five_upgrade_preserves_unknown_primary_and_does_not_invent_channels(tmp_path, hashing):
    import json

    pytest.importorskip("lancedb")
    import lancedb

    from placecell.store.lancedb_store import LanceDBStore

    store = LanceDBStore(tmp_path, CollectionInfo("old", hashing.model_name, hashing.dimension))
    memory = embedded(hashing, "printer", t=1)
    store.upsert([memory])
    store.close()
    import sqlite3

    with sqlite3.connect(tmp_path / "old.state.sqlite3") as db:
        db.execute("ALTER TABLE memories DROP COLUMN caption_vector")
        db.execute("ALTER TABLE memories DROP COLUMN embedding_kind")
        db.execute("UPDATE memories SET payload=json_remove(payload,'$.embedding_kind')")
    table = lancedb.connect(str(tmp_path)).open_table("old")
    table.drop_columns(["caption_vector", "embedding_kind"])
    meta_path = tmp_path / "old.collection.json"
    metadata = json.loads(meta_path.read_text())
    metadata["schema_version"] = 5
    meta_path.write_text(json.dumps(metadata))
    store = LanceDBStore.open(tmp_path, "old")
    assert store.get(memory.id).embedding_kind == "legacy"
    recall = Recall(store, hashing, clock=lambda: 1)
    assert recall.similar("printer")[0].memory.id == memory.id
    assert recall.similar("printer", mode="image") == []
    assert recall.similar("printer", mode="caption") == []
    store.close()


def test_evaluator_reports_measured_channel_differences_and_rejects_bad_labels(store, hashing, tmp_path):
    printer = dual(hashing, "printer", "desk", t=2)
    desk = dual(hashing, "chair", "printer", t=1, confidence=0.5)
    store.upsert([printer, desk])
    cases = [RetrievalCase("printer", (printer.id,))]
    report = evaluate_retrieval(store, hashing, cases, k=1)
    assert report["reference_time"] == 2 and report["coverage"]["both"] == 2
    assert report["modes"]["caption"]["hit_rate_at_k"] == 0
    assert report["modes"]["image"]["mrr_at_k"] == 1
    assert report["combined_minus_caption_hit_rate"] == 1
    assert report["modes"]["combined"]["queries"][0]["latency_ms"] >= 0
    for invalid in ([], [RetrievalCase("printer", ("missing",))]):
        with pytest.raises(ValidationError):
            evaluate_retrieval(store, hashing, invalid)
    with pytest.raises(ValidationError):
        evaluate_retrieval(store, hashing, cases, now=float("nan"))
    labels = tmp_path / "labels.json"
    for content in ("{}", "[]", "[{}]", '[{"query":"printer","relevant_ids":[]}]'):
        labels.write_text(content)
        with pytest.raises(ValidationError):
            load_cases(labels)
    for query, ids in (("", ("id",)), ("printer", ("id", "id")), ("printer", ("",))):
        with pytest.raises(ValidationError):
            RetrievalCase(query, ids)
    store.upsert([replace(printer, embedding_kind="legacy", caption_embedding=None)])
    with pytest.raises(ValidationError, match="legacy"):
        evaluate_retrieval(store, hashing, cases)
    store.upsert([embedded(hashing, "printer", t=2), embedded(hashing, "desk", t=1)])
    with pytest.raises(ValidationError, match="image embeddings"):
        evaluate_retrieval(store, hashing, cases)


def test_evaluation_command_creates_report_without_overwriting(tmp_path, hashing, monkeypatch):
    import json

    pytest.importorskip("lancedb")
    from placecell.store.lancedb_store import LanceDBStore

    store = LanceDBStore(tmp_path, CollectionInfo("recording", hashing.model_name, hashing.dimension))
    memory = dual(hashing, "printer", "desk")
    store.upsert([memory])
    store.close()
    labels = tmp_path / "queries.json"
    labels.write_text(json.dumps([{"query": "printer", "relevant_ids": [memory.id]}]))
    monkeypatch.setattr("placecell.providers.clip.ClipEmbedder", lambda *a, **k: hashing)
    output = tmp_path / "report.json"
    args = [
        "--backend",
        "clip",
        "--db-path",
        str(tmp_path),
        "--collection",
        "recording",
        "--queries",
        str(labels),
        "--output",
        str(output),
    ]
    main(args)
    assert json.loads(output.read_text())["modes"]["image"]["hit_rate_at_k"] == 1
    with pytest.raises(ValidationError, match="already exists"):
        main(args)
