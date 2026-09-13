from __future__ import annotations

import pytest

from placecell import CollectionInfo, InMemoryStore
from placecell.errors import ModelMismatchError, ValidationError
from placecell.migrate import reembed
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
from tests.conftest import FakeMediaEmbedder, embedded


def test_reembed_copies_everything_it_can_and_keeps_lifecycle_fields(
    store: InMemoryStore, hashing: HashingEmbedder
) -> None:
    old = [
        embedded(hashing, "a printer", t=1, observations=7, confidence=0.4, superseded=True, misses=2, last_miss=3.0),
        embedded(hashing, "", t=2, x=3),  # no caption: only an image-capable model can take it
    ]
    store.upsert(old)
    new_embedder = HashingEmbedder(32)
    target = InMemoryStore(CollectionInfo("v2", new_embedder.model_name, 32))
    report = reembed(store, target, new_embedder, batch_size=1)
    assert (report.read, report.written, report.skipped) == (2, 1, 1) and report.skipped_ids == (old[1].id,)
    moved = target.get(old[0].id)
    assert moved is not None and moved.model == "hashing-32" and moved.embedding.shape == (32,)  # type: ignore[union-attr]
    assert (moved.observations, moved.confidence, moved.superseded, moved.misses, moved.last_miss) == (
        7,
        0.4,
        True,
        2,
        3.0,
    )
    assert moved.sightings == old[0].sightings and moved.superseded_at == old[0].superseded_at
    media = FakeMediaEmbedder()
    target2 = InMemoryStore(CollectionInfo("v3", media.model_name, media.dimension))
    report = reembed(store, target2, media)
    assert report.written == 2 and report.skipped == 0 and target2.count(EVERYTHING) == 2
    assert media.text_calls == [["a printer"]] and len(media.media_calls) == 1
    with pytest.raises(ModelMismatchError):
        reembed(store, target, media)
    with pytest.raises(ValidationError):
        reembed(store, target, new_embedder, batch_size=0)
