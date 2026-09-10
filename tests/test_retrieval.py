from __future__ import annotations

import pytest

from placecell import Filter, InMemoryStore, Pose, Recall
from placecell.errors import ModelMismatchError, ValidationError
from placecell.providers import HashingEmbedder
from tests.conftest import DIM, embedded


def test_recall_binds_to_the_collection_model(store: InMemoryStore) -> None:
    with pytest.raises(ModelMismatchError):
        Recall(store, HashingEmbedder(DIM * 2))
    with pytest.raises(ValidationError):
        Recall(store, HashingEmbedder(DIM), half_life_s=0)


def test_similar_weighs_similarity_by_decayed_confidence(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    day = 86400.0
    store.upsert(
        [
            embedded(hashing, "fire extinguisher next to the door", t=0, x=0, y=0),
            embedded(hashing, "fire extinguisher next to the door", t=30 * day, x=8, y=8, camera="back"),
        ]
    )
    recall = Recall(store, hashing, half_life_s=7 * day, clock=lambda: 30 * day)
    ranked = recall.similar("fire extinguisher", k=2)
    assert [r.memory.camera_id for r in ranked] == ["back", "front"]
    assert ranked[0].similarity == pytest.approx(ranked[1].similarity)
    assert ranked[0].confidence == pytest.approx(1.0) and ranked[1].confidence == pytest.approx(0.5 ** (30 / 7))
    assert ranked[0].score > ranked[1].score
    assert len(recall.similar("fire extinguisher", k=1)) == 1
    assert recall.similar("fire extinguisher", where=Filter(camera_id="front"))[0].memory.camera_id == "front"
    with pytest.raises(ValidationError):
        recall.similar("   ")
    with pytest.raises(ValidationError):
        recall.similar("x", k=0)


def test_between_and_near_are_plain_lookups(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    store.upsert([embedded(hashing, f"m{i}", t=float(i), x=float(i), y=0) for i in range(5)])
    recall = Recall(store, hashing, clock=lambda: 4.0)
    assert [r.memory.caption for r in recall.between(1, 3)] == ["m1", "m2"]
    assert [r.memory.caption for r in recall.between(0, 10, where=Filter(camera_id="front"), limit=2)] == ["m0", "m1"]
    assert [r.memory.caption for r in recall.near(Pose(4, 0), 1.5)] == ["m3", "m4"]
    near = recall.near(Pose(0, 0), 0.1)
    assert near[0].similarity is None and near[0].score == near[0].confidence
    # a caller's time window is overridden by the explicit arguments, never silently combined
    assert [r.memory.caption for r in recall.between(3, 5, where=Filter(time_from=0, time_to=1))] == ["m3", "m4"]
