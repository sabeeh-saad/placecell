from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from placecell import CollectionInfo, Curator, Filter, InMemoryStore, Recall, Reinforcer, VectorStore
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
from tests.conftest import embedded


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
