from __future__ import annotations

import json
from pathlib import Path

import pytest

from placecell import SCHEMA_VERSION, CollectionInfo, Filter, Pose, Reinforcer
from placecell.errors import ModelMismatchError, ValidationError
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
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


def test_vector_projection_recovers_and_can_be_rebuilt(tmp_path: Path, hashing: HashingEmbedder) -> None:
    from unittest.mock import patch

    store = LanceDBStore(tmp_path, CollectionInfo("projection", hashing.model_name, DIM))
    rows = [embedded(hashing, f"printer {i}", t=i) for i in range(40)]
    store.upsert(rows)
    with (
        patch.object(store._table, "merge_insert", side_effect=RuntimeError("index write interrupted")),
        pytest.raises(RuntimeError),
    ):
        store.search(hashing.embed_text(["printer"])[0], 1)
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
    repeat = embedded(hashing, "printer", t=400, camera="back")
    Reinforcer(upgraded).reinforce_or_insert(repeat)
    upgraded.close()
    reopened = LanceDBStore.open(tmp_path, info.name)
    again, _ = Reinforcer(reopened).reinforce_or_insert(repeat)
    assert again.observations == 3 and again.sighting_times == (100, 200, 400)
    assert reopened.count(Filter(observation_id=repeat.id)) == 1
    assert json.loads(meta.read_text())["schema_version"] == SCHEMA_VERSION
    reopened.close()
