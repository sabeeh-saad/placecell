from __future__ import annotations

import numpy as np
import pytest

from placecell import CollectionInfo, Filter, InMemoryStore, Pose, VectorStore
from placecell.errors import ModelMismatchError, ValidationError
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
from tests.conftest import DIM, embedded

# Every test below runs against each backend through the `store` fixture in conftest.


def test_store_fulfils_protocol_and_validates_collection() -> None:
    assert isinstance(InMemoryStore(CollectionInfo("c", "m", 3)), VectorStore)
    with pytest.raises(ValidationError):
        CollectionInfo("", "m", 3)
    with pytest.raises(ValidationError):
        CollectionInfo("c", "m", 0)


def test_upsert_get_delete_and_binding(store: VectorStore, hashing: HashingEmbedder) -> None:
    a = embedded(hashing, "a chair", t=1.0)
    assert store.upsert([a]) == 1 and store.get(a.id) == a and store.count() == 1
    newer = embedded(hashing, "a chair", t=1.0, confidence=0.5)
    store.upsert([newer])
    assert store.count() == 1 and store.get(a.id).confidence == 0.5  # type: ignore[union-attr]
    assert store.delete([a.id, "missing"]) == 1 and store.get(a.id) is None
    foreign = HashingEmbedder(DIM * 2)
    with pytest.raises(ValidationError):
        store.upsert([embedded(foreign, "x", model=hashing.model_name)])  # wrong dimension
    with pytest.raises(ModelMismatchError):
        store.upsert([embedded(hashing, "x", model="other")])
    with pytest.raises(ValidationError):
        store.upsert([embedded(hashing, "x", embedding=None, model="")])


def test_filter_semantics(store: VectorStore, hashing: HashingEmbedder) -> None:
    rows = [
        embedded(hashing, "a", t=10, x=0, y=0, robot="r1", camera="front"),
        embedded(hashing, "b", t=20, x=5, y=0, robot="r1", camera="back"),
        embedded(hashing, "c", t=30, x=0, y=1, robot="r2", camera="front", superseded=True),
    ]
    store.upsert(rows)
    ids = lambda ms: [m.caption for m in ms]  # noqa: E731
    assert ids(store.query()) == ["a", "b"]
    assert ids(store.query(EVERYTHING)) == ["a", "b", "c"]
    assert ids(store.query(Filter(robot_id="r1"))) == ["a", "b"]
    assert ids(store.query(Filter(camera_id="front", include_superseded=True))) == ["a", "c"]
    assert ids(store.query(Filter(time_from=20, time_to=30))) == ["b"]
    assert ids(store.query(Filter(time_from=10, time_to=10))) == []
    assert ids(store.query(Filter(near=Pose(0, 0), radius=2.0))) == ["a"]
    assert ids(store.query(Filter(near=Pose(0, 0), radius=2.0, include_superseded=True))) == ["a", "c"]
    assert ids(store.query(Filter(near=Pose(0, 0, frame_id="odom"), radius=100))) == []
    assert ids(store.query(limit=1)) == ["a"]
    assert store.count(Filter(robot_id="r2")) == 0 and store.count(EVERYTHING) == 3
    assert store.delete_where(Filter(robot_id="r1", camera_id="back")) == 1 and store.count(EVERYTHING) == 2
    with pytest.raises(ValidationError):
        Filter(near=Pose(0, 0))
    with pytest.raises(ValidationError):
        Filter(near=Pose(0, 0), radius=0)
    with pytest.raises(ValidationError):
        Filter(time_from=2, time_to=1)
    with pytest.raises(ValidationError):
        Filter(time_from=float("nan"))
    with pytest.raises(ValidationError):
        Filter(time_to=float("inf"))
    with pytest.raises(ValidationError):
        store.query(limit=-1)


def test_search_ranks_by_cosine_and_respects_filters(store: VectorStore, hashing: HashingEmbedder) -> None:
    store.upsert(
        [
            embedded(hashing, "red fire extinguisher on the wall", t=1, x=0, y=0),
            embedded(hashing, "grey office chair", t=2, x=5, y=5),
            embedded(hashing, "fire extinguisher", t=3, x=9, y=9, superseded=True),
        ]
    )
    q = hashing.embed_text(["fire extinguisher"])[0]
    hits = store.search(q, 5)
    assert hits[0].memory.caption == "red fire extinguisher on the wall"
    assert all(-1.0 <= h.score <= 1.0 for h in hits) and hits[0].score > hits[-1].score
    assert len(hits) == 2  # superseded excluded by default
    assert store.search(q, 5, EVERYTHING)[0].memory.caption == "fire extinguisher"
    assert store.search(q, 5, Filter(near=Pose(5, 5), radius=1))[0].memory.caption == "grey office chair"
    assert store.search(q, 1)[0].memory.caption.startswith("red")
    assert store.search(q, 5, Filter(robot_id="nobody")) == []
    assert store.search(np.zeros(DIM), 5) == []
    with pytest.raises(ValidationError):
        store.search(q, 0)
    with pytest.raises(ValidationError):
        store.search(np.ones(3), 1)


def test_search_on_empty_store(store: VectorStore, hashing: HashingEmbedder) -> None:
    assert store.search(hashing.embed_text(["x"])[0], 3) == []
    assert store.query() == [] and store.count(EVERYTHING) == 0


def test_invalid_batch_preserves_rows_and_search_index(store: VectorStore, hashing: HashingEmbedder) -> None:
    first = embedded(hashing, "printer", t=1)
    store.upsert([first])
    vector = hashing.embed_text(["printer"])[0]
    store.search(vector, 10)  # populate the cached index before attempting a partial write
    replacement = embedded(hashing, "chair", t=1)
    addition = embedded(hashing, "desk", t=2)
    invalid = embedded(hashing, "lamp", t=3, model="foreign")
    with pytest.raises(ModelMismatchError):
        store.upsert(iter([replacement, addition, invalid]))
    assert store.query() == [first]
    assert [h.memory for h in store.search(vector, 10)] == [first]
    assert store.upsert([addition]) == 1
    assert {h.memory.id for h in store.search(vector, 10)} == {first.id, addition.id}


def test_in_memory_close_drops_everything(hashing: HashingEmbedder) -> None:
    store = InMemoryStore(CollectionInfo("c", hashing.model_name, DIM))
    store.upsert([embedded(hashing, "a")])
    store.close()
    assert store.count(EVERYTHING) == 0
