from __future__ import annotations

from pathlib import Path

import pytest

from placecell import CollectionInfo, Filter, Pose
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
        "superseded = false AND robot_id = 'r''1' AND camera_id = 'c' AND timestamp >= 1.5 AND timestamp < 2.0 "
        "AND frame_id = 'map' AND map_id = 'm' AND ((x - 1) * (x - 1) + (y - -2) * (y - -2)) <= 0.25"
    )
