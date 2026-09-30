from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from placecell import SCHEMA_VERSION, CollectionInfo, Filter, Pose, Reinforcer
from placecell.errors import ModelMismatchError, ValidationError
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
from tests import test_multimodal, test_store
from tests.conftest import DIM, embedded

pytest.importorskip("lancedb")
from placecell.store.lancedb_store import LanceDBStore, _sql


def test_data_survives_reopen_and_identity_is_enforced(tmp_path: Path, hashing: HashingEmbedder) -> None:
    info = CollectionInfo("office", hashing.model_name, DIM)
    store = LanceDBStore(tmp_path, info)
    store.upsert([embedded(hashing, "a printer", t=1), embedded(hashing, "a chair", t=2, x=3)])
    store.close()

    reopened = LanceDBStore.open(tmp_path, "office")
    assert reopened.info == info and reopened.count() == 2
    assert reopened.get("r1:front:1000") is not None
    assert [m.caption for m in reopened.query(Filter(near=Pose(3, 0), radius=0.5))] == ["a chair"]
    hits = reopened.search(hashing.embed_text(["printer"])[0], 1)
    assert hits[0].memory.caption == "a printer" and 0 < hits[0].score <= 1.0
    reopened.close()

    with pytest.raises(ModelMismatchError):
        LanceDBStore(tmp_path, CollectionInfo("office", "other-model", DIM))
    with pytest.raises(ModelMismatchError):
        LanceDBStore(tmp_path, CollectionInfo("office", hashing.model_name, DIM + 1))
    with pytest.raises(ValidationError):
        LanceDBStore(tmp_path, CollectionInfo("office", hashing.model_name, DIM, schema_version=1))
    with pytest.raises(ValidationError):
        LanceDBStore.open(tmp_path, "missing")


def test_two_collections_share_one_directory(tmp_path: Path, hashing: HashingEmbedder) -> None:
    a = LanceDBStore(tmp_path, CollectionInfo("a", hashing.model_name, DIM))
    b = LanceDBStore(tmp_path, CollectionInfo("b", "other", 4))
    a.upsert([embedded(hashing, "x")])
    assert a.count() == 1 and b.count() == 0
    assert (tmp_path / "a.collection.json").exists() and (tmp_path / "b.collection.json").exists()


def test_version_four_memories_need_new_evidence_to_establish_pose_provenance(tmp_path, hashing):
    import lancedb

    info = CollectionInfo("legacy_view", hashing.model_name, DIM)
    store = LanceDBStore(tmp_path, info)
    memory = embedded(hashing, "printer", t=100)
    store.upsert([memory])
    with store.transaction():
        store._conn.execute(
            "UPDATE memories SET payload=json_remove(payload, '$.view_timestamp', '$.localization_checked', "
            "'$.anchor_x', '$.anchor_y', '$.anchor_yaw')"
        )
    store.close()
    table = lancedb.connect(str(tmp_path)).open_table(info.name)
    table.drop_columns(["view_timestamp", "localization_checked", "anchor_x", "anchor_y", "anchor_yaw"])
    metadata = tmp_path / "legacy_view.collection.json"
    old = json.loads(metadata.read_text())
    old["schema_version"] = 4
    metadata.write_text(json.dumps(old))
    reopened = LanceDBStore.open(tmp_path, info.name)
    legacy = reopened.get(memory.id)
    assert legacy.view_timestamp is None and not legacy.localization_checked
    assert legacy.caption == memory.caption and legacy.sighting_times == (100,)
    repeat = embedded(hashing, "printer", t=200, x=0.5)
    retained, merged = Reinforcer(reopened).reinforce_or_insert(repeat)
    assert merged and retained.pose == repeat.pose and retained.view_timestamp == 200
    reopened.close()
    reopened = LanceDBStore.open(tmp_path, info.name)
    assert reopened.get(memory.id).view_timestamp == 200
    assert reopened.get(memory.id).localization_checked
    reopened.close()


def test_delete_handles_large_id_lists_and_quotes(tmp_path: Path, hashing: HashingEmbedder) -> None:
    store = LanceDBStore(tmp_path, CollectionInfo("c", hashing.model_name, DIM))
    rows = [embedded(hashing, "m", t=float(i), camera="o'brien") for i in range(600)]
    assert store.upsert(rows) == 600
    assert store.count(Filter(camera_id="o'brien")) == 600
    assert store.delete([m.id for m in rows] + ["ghost"]) == 600
    assert store.count(EVERYTHING) == 0
    assert store.delete([]) == 0
    assert store.delete_where(EVERYTHING) == 0
    assert store.query(limit=0) == []


def test_sql_translation() -> None:
    assert _sql(EVERYTHING) is None
    assert _sql(Filter()) == "superseded = false"
    where = Filter(
        robot_id="r'1", camera_id="c", time_from=1.5, time_to=2.0, near=Pose(1, -2, 0, "map", "m"), radius=0.5
    )

    assert _sql(where) == (
        "superseded = false AND robot_id = 'r''1' AND camera_id = 'c' AND "
        "array_position(array_sort(array_append(sighting_times, 2.0)), 2.0) > "
        "array_position(array_sort(array_append(sighting_times, 1.5)), 1.5) "
        "AND frame_id = 'map' AND map_id = 'm' AND ((x - 1) * (x - 1) + (y - -2) * (y - -2)) <= 0.25"
    )


def test_vector_projection_recovers_and_can_be_rebuilt(tmp_path: Path, hashing: HashingEmbedder, monkeypatch) -> None:
    from unittest.mock import patch

    from placecell.store import lancedb_store

    monkeypatch.setattr(lancedb_store, "EXACT_SEARCH_ROWS", 0)
    store = LanceDBStore(tmp_path, CollectionInfo("projection", hashing.model_name, DIM))
    rows = [embedded(hashing, f"printer {i}", t=i) for i in range(40)]
    store.upsert(rows)
    with patch.object(store._table, "merge_insert", side_effect=RuntimeError("index write interrupted")):
        # Searches score unsynced rows from the state store and never see the failed write.
        assert store.search(hashing.embed_text(["printer 3"])[0], 1)[0].memory.id == rows[3].id
        with pytest.raises(RuntimeError):
            store.sync_index()
    assert store.count() == 40 and store.query(limit=1) == [rows[0]]
    store.maintain(vector_index_min_rows=32)
    assert any("vector" in index.columns for index in store._table.list_indices())
    store.rebuild_index()
    assert store.search(hashing.embed_text(["printer"])[0], 1)
    store.close()
    assert LanceDBStore.open(tmp_path, "projection").count() == 40


def test_legacy_history_is_imported_without_a_state_sidecar(tmp_path: Path, hashing: HashingEmbedder) -> None:
    from dataclasses import replace

    import lancedb

    from placecell import Sighting
    from placecell.store.codec import to_row

    info = CollectionInfo("legacy_history", hashing.model_name, DIM)
    store = LanceDBStore(tmp_path, info)
    store.close()
    (tmp_path / "legacy_history.state.sqlite3").unlink()
    memory = replace(
        embedded(hashing, "printer", t=0),
        last_seen=149,
        observations=150,
        schema_version=3,
        sightings=tuple(Sighting(f"old-{i}", i) for i in range(150)),
    )
    table = lancedb.connect(str(tmp_path)).open_table(info.name)
    table.add([to_row(memory)])
    metadata = tmp_path / "legacy_history.collection.json"
    metadata.write_text(json.dumps({"name": info.name, "model": info.model, "dimension": DIM, "schema_version": 3}))
    upgraded = LanceDBStore.open(tmp_path, info.name)
    assert len(upgraded.get(memory.id).sightings) == 64
    assert len(upgraded.sightings(memory.id, limit=200)) == 150
    assert upgraded.count(Filter(observation_id="old-1")) == 1
    assert json.loads(metadata.read_text())["schema_version"] == SCHEMA_VERSION
    upgraded.close()


@pytest.mark.parametrize("interrupted", [False, True])
def test_version_two_collections_upgrade_without_losing_memories(
    tmp_path: Path, hashing: HashingEmbedder, interrupted: bool
) -> None:
    import lancedb

    info = CollectionInfo("legacy", hashing.model_name, DIM)
    store = LanceDBStore(tmp_path, info)
    first = embedded(hashing, "printer", t=100, last_seen=200, observations=2)
    gone = embedded(hashing, "chair", t=10, last_seen=300, superseded=True)
    store.upsert([first, gone])
    store.close()
    table = lancedb.connect(str(tmp_path)).open_table(info.name)
    if not interrupted:
        table.drop_columns(["sighting_ids", "sighting_times", "superseded_at", "evidence_managed"])
    table.update(values={"schema_version": 2})
    meta = tmp_path / "legacy.collection.json"
    metadata = json.loads(meta.read_text())
    metadata["schema_version"] = 2
    meta.write_text(json.dumps(metadata))

    upgraded = LanceDBStore.open(tmp_path, info.name)
    assert upgraded.info.schema_version == SCHEMA_VERSION
    assert upgraded.count(EVERYTHING) == 2
    remembered = upgraded.get(first.id)
    assert remembered is not None and remembered.sighting_times == (100, 200) and remembered.observations == 2
    assert upgraded.count(Filter(time_from=190, time_to=210)) == 1
    superseded = upgraded.get(gone.id)
    assert superseded is not None and superseded.superseded_at == 300 and superseded.last_seen == 300
    repeat = embedded(hashing, "printer", t=400, camera="front")
    Reinforcer(upgraded).reinforce_or_insert(repeat)
    upgraded.close()
    reopened = LanceDBStore.open(tmp_path, info.name)
    again, _ = Reinforcer(reopened).reinforce_or_insert(repeat)
    assert again.observations == 3 and again.sighting_times == (100, 200, 400)
    assert reopened.count(Filter(observation_id=repeat.id)) == 1
    assert json.loads(meta.read_text())["schema_version"] == SCHEMA_VERSION
    reopened.close()


def _random(hashing: HashingEmbedder, count: int, robots: int = 1, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [
        embedded(hashing, f"view {i}", t=i, x=i % 17, robot=f"r{i % robots}").with_embedding(
            rng.standard_normal(DIM), hashing.model_name, kind="caption"
        )
        for i in range(count)
    ]


def test_vector_indexes_are_retrained_as_the_collection_doubles(
    tmp_path: Path, hashing: HashingEmbedder, monkeypatch
) -> None:
    from unittest.mock import patch

    from placecell.store import lancedb_store

    monkeypatch.setattr(lancedb_store, "EXACT_SEARCH_ROWS", 0)
    store = LanceDBStore(tmp_path, CollectionInfo("grow", hashing.model_name, DIM))
    rows = _random(hashing, 80)
    store.upsert(rows[:40])
    store.maintain(vector_index_min_rows=32)
    indexes = {index.columns[0]: index for index in store._table.list_indices()}
    assert indexes["vector"].index_type == indexes["caption_vector"].index_type == "IvfSq"
    assert all(indexes[c].index_type == "Bitmap" for c in ("robot_id", "camera_id", "map_id", "frame_id", "role"))
    assert indexes["superseded"].index_type == "Bitmap" and indexes["id"].index_type == "BTree"
    assert store._indexes["vector"] == {"rows": 40, "partitions": 6}
    with patch.object(store._table, "optimize") as optimize:
        store.maintain(vector_index_min_rows=32)
    assert not optimize.called, "an unchanged projection needs no maintenance"

    store.upsert(rows[40:79])
    with patch.object(store._table, "create_index", wraps=store._table.create_index) as create:
        store.maintain(vector_index_min_rows=32)
    assert not create.called and store._indexes["vector"] == {"rows": 40, "partitions": 6}
    store.upsert(rows[79:])
    store.maintain(vector_index_min_rows=32)
    assert store._indexes["vector"] == store._indexes["caption_vector"] == {"rows": 80, "partitions": 9}
    assert store.search(rows[3].embedding, 1)[0].memory.id == rows[3].id
    store.close()

    reopened = LanceDBStore.open(tmp_path, "grow")
    assert reopened._indexes["vector"] == {"rows": 80, "partitions": 9}
    reopened.rebuild_index()
    assert reopened._indexes == {} and json.loads((tmp_path / "grow.index.json").read_text()) == {}
    reopened.maintain(vector_index_min_rows=32)
    assert reopened._indexes["vector"] == {"rows": 80, "partitions": 9}
    reopened.close()


def test_projection_updates_keep_the_index_covering_unchanged_rows(tmp_path: Path, hashing: HashingEmbedder) -> None:
    store = LanceDBStore(tmp_path, CollectionInfo("covered", hashing.model_name, DIM))
    rows = _random(hashing, 64)
    store.upsert(rows)
    store.maintain(vector_index_min_rows=32)
    moved = rows[5].with_embedding(np.ones(DIM), hashing.model_name, kind="caption")
    store.upsert([moved, *_random(hashing, 66)[64:]])
    store._sync_index()
    coverage = {index.name: (index.num_indexed_rows, index.num_unindexed_rows) for index in store._table.list_indices()}
    assert coverage["vector_idx"] == (63, 3)
    store.close()


def test_an_index_without_a_training_record_is_retrained(tmp_path: Path, hashing: HashingEmbedder) -> None:
    from lancedb.index import IvfFlat

    store = LanceDBStore(tmp_path, CollectionInfo("legacy_index", hashing.model_name, DIM))
    store.upsert(_random(hashing, 50))
    store.maintain(vector_index_min_rows=1000)
    store._table.create_index("vector", config=IvfFlat(distance_type="cosine", num_partitions=1))
    assert store._indexes == {}
    store.maintain(vector_index_min_rows=40)
    assert store._indexes == {"vector": {"rows": 50, "partitions": 7}, "caption_vector": {"rows": 50, "partitions": 7}}
    store.close()


def test_small_filtered_sets_are_scored_exactly(tmp_path: Path, hashing: HashingEmbedder, monkeypatch) -> None:
    from unittest.mock import patch

    from placecell.store import lancedb_store

    monkeypatch.setattr(lancedb_store, "EXACT_SEARCH_ROWS", 10)
    store = LanceDBStore(tmp_path, CollectionInfo("exact", hashing.model_name, DIM))
    rows = _random(hashing, 60, robots=6)
    store.upsert(rows)
    query = rows[7].embedding
    with patch.object(store._table, "search", side_effect=AssertionError("vector index used")):
        hits = store.search(query, 3, Filter(robot_id="r1"))
        assert hits[0].memory.id == rows[7].id and hits[0].score == pytest.approx(1)
        with pytest.raises(AssertionError):
            store.search(query, 3)
    assert store.search(query, 3)[0].memory.id == rows[7].id
    store.close()


def test_selective_prefilters_still_fill_k_results(tmp_path: Path, hashing: HashingEmbedder, monkeypatch) -> None:
    from placecell.store import lancedb_store

    monkeypatch.setattr(lancedb_store, "EXACT_SEARCH_ROWS", 0)
    monkeypatch.setattr(lancedb_store, "_probes", lambda partitions: 1)
    store = LanceDBStore(tmp_path, CollectionInfo("sparse", hashing.model_name, DIM))
    rows = _random(hashing, 400, robots=25)
    store.upsert(rows)
    store.maintain(vector_index_min_rows=100)
    assert store._indexes["vector"]["partitions"] == 20
    for robot in ("r0", "r3", "r24"):
        hits = store.search(rows[0].embedding, 10, Filter(robot_id=robot))
        assert len(hits) == 10 and {hit.memory.robot_id for hit in hits} == {robot}
    assert len(store.search(rows[0].embedding, 20, Filter(robot_id="r3"))) == 16
    store.close()


@pytest.mark.parametrize(
    "contract",
    [
        test_store.test_search_ranks_by_cosine_and_respects_filters,
        test_store.test_search_on_empty_store,
        test_store.test_invalid_batch_preserves_rows_and_search_index,
        test_multimodal.test_omitted_caption_is_recovered_outside_recent_candidate_window,
        test_multimodal.test_channel_filters_missing_vectors_and_atomic_caption_updates,
    ],
    ids=lambda contract: contract.__name__,
)
def test_vector_index_path_fulfils_the_search_contract(tmp_path, hashing, monkeypatch, contract) -> None:
    from placecell.store import lancedb_store

    monkeypatch.setattr(lancedb_store, "EXACT_SEARCH_ROWS", 0)
    store = LanceDBStore(tmp_path, CollectionInfo("indexed", hashing.model_name, DIM))
    contract(store, hashing)
    store.close()


def _pending(store: LanceDBStore) -> int:
    return int(store._conn.execute("SELECT COUNT(*) FROM dirty_vectors").fetchone()[0])


def _found(store: LanceDBStore, query, k: int, where: Filter | None = None) -> list[tuple[str, float]]:
    return [(hit.memory.id, hit.score) for hit in store.search(query, k, where)]


def _matches_exact_search(store: LanceDBStore, query, k: int, where: Filter | None = None) -> bool:
    from placecell.store.state import StateStore

    found = store.search(query, k, where)
    exact = StateStore.search(store, query, k, where)
    return [h.memory.id for h in found] == [h.memory.id for h in exact] and [h.score for h in found] == pytest.approx(
        [h.score for h in exact], abs=1e-5
    )


def test_searches_see_unsynced_writes_without_writing_the_index(tmp_path: Path, hashing, monkeypatch) -> None:
    from unittest.mock import patch

    from placecell.store import lancedb_store

    monkeypatch.setattr(lancedb_store, "EXACT_SEARCH_ROWS", 0)
    store = LanceDBStore(tmp_path, CollectionInfo("fresh", hashing.model_name, DIM))
    rows = _random(hashing, 40, robots=2)
    store.upsert(rows)
    assert store.sync_index(limit=16) == 16 and store.sync_index() == 24 and store.sync_index() == 0
    rng = np.random.default_rng(7)
    added = embedded(hashing, "new", t=100).with_embedding(rng.standard_normal(DIM), hashing.model_name, kind="caption")
    moved = rows[5].with_embedding(rng.standard_normal(DIM), hashing.model_name, kind="caption")
    store.upsert([added, moved])
    store.delete([rows[9].id])
    with (
        patch.object(store._table, "merge_insert", side_effect=AssertionError("search wrote the index")),
        patch.object(store._table, "delete", side_effect=AssertionError("search wrote the index")),
    ):
        assert store.search(added.embedding, 1)[0].memory.id == added.id
        # An updated row scores against its current vector, not the one still in the index.
        assert store.search(moved.embedding, 1)[0].score == pytest.approx(1, abs=1e-5)
        stale = dict(_found(store, rows[5].embedding, 40))
        assert stale[moved.id] == pytest.approx(
            float(moved.embedding @ rows[5].embedding)
            / (np.linalg.norm(moved.embedding) * np.linalg.norm(rows[5].embedding)),
            abs=1e-5,
        )
        assert rows[9].id not in dict(_found(store, rows[9].embedding, 40))
        for query in (added.embedding, rows[5].embedding, rows[0].embedding, rows[30].embedding):
            assert _matches_exact_search(store, query, 10)
            assert _matches_exact_search(store, query, 10, Filter(robot_id="r1"))
    assert _pending(store) == 3
    assert store.sync_index() == 3 and _pending(store) == 0
    assert _matches_exact_search(store, rows[5].embedding, 10)
    assert _matches_exact_search(store, rows[5].embedding, 10, EVERYTHING)
    with pytest.raises(ValidationError):
        store.sync_index(0)
    store.close()


def test_a_long_backlog_is_synced_in_a_bounded_batch_or_searched_exactly(tmp_path: Path, hashing, monkeypatch) -> None:
    import threading
    from unittest.mock import patch

    from placecell.store import lancedb_store

    monkeypatch.setattr(lancedb_store, "EXACT_SEARCH_ROWS", 0)
    monkeypatch.setattr(lancedb_store, "SEARCH_BACKLOG_ROWS", 2)
    store = LanceDBStore(tmp_path, CollectionInfo("backlog", hashing.model_name, DIM))
    rows = _random(hashing, 30)
    store.upsert(rows[:20])
    store.sync_index()
    store.upsert(rows[20:24])
    query = rows[22].embedding
    assert _matches_exact_search(store, query, 5)
    assert _pending(store) == 2, "one batch of the backlog limit was synced"

    store.upsert(rows[24:30])
    held, release = threading.Event(), threading.Event()

    def sync_elsewhere() -> None:
        with store._projection_lock:
            held.set()
            release.wait(10)

    syncing = threading.Thread(target=sync_elsewhere)
    syncing.start()
    try:
        assert held.wait(10)
        with patch.object(store._table, "search", side_effect=AssertionError("vector index used")):
            # Another sync is running, so the search neither waits for it nor uses the stale index.
            assert _matches_exact_search(store, query, 5)
        assert _pending(store) == 8
    finally:
        release.set()
        syncing.join()
    with patch.object(store._table, "search", side_effect=AssertionError("vector index used")):
        # Still too long after one batch: scored exactly.
        assert _matches_exact_search(store, query, 5)
    assert _pending(store) == 6
    store.close()


def test_searches_run_beside_writes_and_syncs(tmp_path: Path, hashing, monkeypatch) -> None:
    import threading

    from placecell.store import lancedb_store

    monkeypatch.setattr(lancedb_store, "EXACT_SEARCH_ROWS", 0)
    store = LanceDBStore(tmp_path, CollectionInfo("busy", hashing.model_name, DIM))
    rows = _random(hashing, 120, robots=3)
    store.upsert(rows[:40])
    store.maintain(vector_index_min_rows=32)
    written = threading.Event()
    failures: list[BaseException] = []

    def write() -> None:
        try:
            for row in rows[40:]:
                store.upsert([row])
        except BaseException as e:  # pragma: no cover - reported below
            failures.append(e)
        finally:
            written.set()

    def sync() -> None:
        try:
            while not written.is_set():
                store.sync_index(limit=8)
        except BaseException as e:  # pragma: no cover - reported below
            failures.append(e)

    threads = [threading.Thread(target=write), threading.Thread(target=sync)]
    for thread in threads:
        thread.start()
    searches = 0
    while not written.is_set() or not searches:
        assert len(store.search(rows[searches % 120].embedding, 5, Filter(robot_id="r1"))) == 5
        searches += 1
    for thread in threads:
        thread.join()
    assert not failures
    store.sync_index(limit=200)
    query = rows[77].embedding
    assert _found(store, query, 3)[0] == (rows[77].id, pytest.approx(1, abs=1e-5))
    store.close()
