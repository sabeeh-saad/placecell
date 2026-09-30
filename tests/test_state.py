from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from placecell import CollectionInfo, Curator, Filter, InMemoryStore, Recall, Reinforcer, VectorStore
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
from placecell.store.state import StateStore
from tests.conftest import DIM, embedded


def test_history_is_separate_bounded_and_queryable(store: VectorStore, hashing: HashingEmbedder) -> None:
    reinforcer = Reinforcer(store, remover=None)
    for t in range(150):
        reinforcer.reinforce_or_insert(embedded(hashing, "printer", t=t))
    memory = store.query()[0]
    assert len(memory.sightings) <= 64 and memory.observations == 150
    assert store.count(Filter(observation_id="r1:front:1000")) == 1
    assert Recall(store, hashing).between(1, 3)[0].observed_at == (1, 2)
    seen = []
    after = None
    while page := store.sightings(memory.id, limit=17, after=after):
        seen.extend(s.timestamp for s in page)
        after = page[-1].timestamp, page[-1].id
    assert seen == list(range(150))
    assert store.prune_history(100) == 100
    assert store.count(Filter(time_from=1, time_to=3)) == 0
    assert store.get(memory.id).last_seen == 149


def test_transaction_rolls_back_memory_history_and_deletion(store: VectorStore, hashing: HashingEmbedder) -> None:
    first = embedded(hashing, "printer", t=1)
    store.upsert([first])
    with pytest.raises(RuntimeError), store.transaction():
        Reinforcer(store, remover=None).reinforce_or_insert(embedded(hashing, "printer", t=2))
        store.delete([first.id])
        raise RuntimeError("power failure before commit")
    assert store.get(first.id) == first
    assert store.sightings(first.id) == first.sightings


def test_parallel_reinforcement_does_not_lose_observations(store: VectorStore, hashing: HashingEmbedder) -> None:
    store.upsert([embedded(hashing, "printer", t=0)])
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            workers.map(
                lambda t: Reinforcer(store, remover=None).reinforce_or_insert(embedded(hashing, "printer", t=t)),
                range(1, 21),
            )
        )
    assert len(results) == 20
    assert store.query()[0].observations == 21
    assert len(store.sightings("r1:front:0")) == 21


def test_limited_queries_only_deserialize_returned_rows(store: VectorStore, hashing: HashingEmbedder) -> None:
    store.upsert([embedded(hashing, "printer", t=t, x=t) for t in range(300)])
    with patch.object(store, "_read", wraps=store._read) as read:
        assert len(store.query(limit=5)) == 5
    assert read.call_count == 5
    sizes = []
    for batch in store.iter_query(EVERYTHING, batch_size=37):
        sizes.append(len(batch))
        store.delete(m.id for m in batch)
    assert sum(sizes) == 300 and max(sizes) == 37 and store.count(EVERYTHING) == 0


def test_recent_candidates_enter_freshness_ranking() -> None:
    embedder = HashingEmbedder(8)
    store = InMemoryStore(CollectionInfo("ranking", embedder.model_name, 8))
    vector = embedder.embed_text(["printer"])[0]
    perpendicular = np.roll(vector, 1)
    perpendicular -= vector * float(perpendicular @ vector)
    perpendicular /= np.linalg.norm(perpendicular)
    store.upsert([replace(embedded(embedder, "printer", t=i), embedding=vector) for i in range(4)])
    fresh = replace(embedded(embedder, "printer", t=100), embedding=0.8 * vector + 0.6 * perpendicular)
    store.upsert([fresh])
    top = Recall(store, embedder, half_life_s=10, clock=lambda: 100).similar("printer", k=1)[0]
    assert top.memory.id == fresh.id and top.score == pytest.approx(0.8)


def test_default_retention_expires_abandoned_reinforced_memories(store: VectorStore, hashing: HashingEmbedder) -> None:
    store.upsert([embedded(hashing, "printer", t=1, observations=100)])
    assert Curator(store).run(now=91 * 86400).aged_out == 1


def test_equal_scores_resolve_the_same_way_in_any_row_order(store: VectorStore, hashing: HashingEmbedder) -> None:
    store.upsert([embedded(hashing, "printer", t=t) for t in (5, 1, 4, 2, 3)])
    hits = store.search(hashing.embed_text(["printer"])[0], 2)
    assert [hit.memory.id for hit in hits] == ["r1:front:4000", "r1:front:5000"]


def _file_store(tmp_path: Path, hashing: HashingEmbedder) -> StateStore:
    return StateStore(CollectionInfo("pooled", hashing.model_name, DIM), tmp_path / "state.sqlite3")


def test_reads_outside_a_transaction_do_not_wait_for_it(tmp_path: Path, hashing: HashingEmbedder) -> None:
    store = _file_store(tmp_path, hashing)
    old, new = embedded(hashing, "printer", t=1), embedded(hashing, "sofa", t=2, x=3)
    store.upsert([old])
    assert new.embedding is not None
    seen: dict[str, object] = {}
    done = threading.Event()

    def read() -> None:
        seen["get"] = store.get(new.id)
        seen["count"] = store.count()
        seen["query"] = [m.id for m in store.query()]
        seen["pages"] = [m.id for batch in store.iter_query() for m in batch]
        seen["search"] = [hit.memory.id for hit in store.search(new.embedding, 5)]
        seen["sightings"] = store.sightings(old.id)
        done.set()

    with store.transaction():
        store.upsert([new])
        assert store.get(new.id) == new and store.count() == 2
        assert store.search(new.embedding, 1)[0].memory.id == new.id
        reader = threading.Thread(target=read)
        reader.start()
        assert done.wait(10), "a read waited for another thread's transaction"
    reader.join()
    assert seen == {
        "get": None,
        "count": 1,
        "query": [old.id],
        "pages": [old.id],
        "search": [old.id],
        "sightings": old.sightings,
    }
    # Once committed, the change is visible to reads on any thread.
    with ThreadPoolExecutor(max_workers=1) as other:
        assert other.submit(store.get, new.id).result() == new
    store.close()


def test_a_long_read_does_not_block_writes_and_close_ends_every_reader(
    tmp_path: Path, hashing: HashingEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    from placecell.store import state

    store = _file_store(tmp_path, hashing)
    first, second = embedded(hashing, "printer", t=1), embedded(hashing, "sofa", t=2, x=3)
    store.upsert([first])
    reading, release, written = threading.Event(), threading.Event(), threading.Event()
    decode = state.from_row

    def slow(payload: dict[str, object]) -> object:
        if threading.current_thread().name == "slow-reader":
            reading.set()
            release.wait(10)
        return decode(payload)

    monkeypatch.setattr(state, "from_row", slow)
    result: list[list[str]] = []
    reader = threading.Thread(target=lambda: result.append([m.id for m in store.query()]), name="slow-reader")
    writer = threading.Thread(target=lambda: (store.upsert([second]), written.set()))
    reader.start()
    try:
        assert reading.wait(10)
        writer.start()
        assert written.wait(10), "a write waited for a read"
        # Another reader gets its own connection and sees the committed write.
        assert store.count() == 2
    finally:
        release.set()
        reader.join()
        writer.join()
    # The long read kept the snapshot it started with.
    assert result == [[first.id]]
    pooled = list(store._readers)
    assert len(pooled) == 2
    store.close()
    for conn in pooled:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        store.get(first.id)
