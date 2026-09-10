from __future__ import annotations

import pytest

from placecell import Ingester, InMemoryStore, Observation, Pose
from placecell.errors import ValidationError
from placecell.observer import ContradictionPolicy, Observer
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
from tests.conftest import FakeCaptioner, embedded, frame


def test_observer_counts_misses_per_visit_and_supersedes(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    policy = ContradictionPolicy(confirm_similarity=0.6, misses_to_supersede=2, visit_gap_s=100)
    observer = Observer(store, policy)
    shelf = embedded(hashing, "a shelf with a red fire extinguisher", t=0, x=0, y=0)
    store.upsert([shelf])
    wall = embedded(hashing, "a bare white wall", t=50, x=0.2, y=0.1, camera="back")
    first = observer.observe(wall, wall.id)
    assert (first.in_view, first.confirmed, first.missed, first.superseded) == (1, 0, 1, 0)
    assert store.get(shelf.id).misses == 1 and store.get(shelf.id).last_miss == 50  # type: ignore[union-attr]
    again = observer.observe(embedded(hashing, "a bare white wall", t=90, x=0.1, y=0.0, camera="left"), None)
    assert again.missed == 0  # same visit, not counted twice
    later = observer.observe(embedded(hashing, "a bare white wall", t=500, x=0.1, y=0.0, camera="left"), None)
    assert later.missed == 1 and later.superseded == 1
    gone = store.get(shelf.id)
    assert gone is not None and gone.superseded and gone.misses == 2
    assert store.count() == 0 and store.count(EVERYTHING) == 1  # invisible to queries, kept as a record


def test_observer_confirms_and_resets_misses(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    observer = Observer(store, ContradictionPolicy(confirm_similarity=0.6, misses_to_supersede=3, visit_gap_s=10))
    shelf = embedded(hashing, "a shelf with a red fire extinguisher", t=0, x=0, y=0, misses=2, last_miss=1.0)
    store.upsert([shelf])
    seen = embedded(hashing, "a shelf with a red fire extinguisher", t=100, x=0.1, y=0.1, camera="back")
    report = observer.observe(seen, seen.id)
    assert (report.confirmed, report.missed) == (1, 0)
    refreshed = store.get(shelf.id)
    assert refreshed is not None and refreshed.misses == 0 and refreshed.last_miss == 0.0


def test_observer_ignores_other_places_headings_and_summaries(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    observer = Observer(store, ContradictionPolicy(same_place_m=0.5, same_heading_rad=0.3, confirm_similarity=0.9))
    store.upsert(
        [
            embedded(hashing, "printer", t=0, x=5, y=5),  # elsewhere
            embedded(hashing, "printer", t=1, x=0, y=0, pose=Pose(0, 0, 2.0)),  # looking the other way
            embedded(hashing, "printer", t=2, x=0.1, y=0, role="summary"),  # summaries are never contradicted
        ]
    )
    report = observer.observe(embedded(hashing, "a door", t=100, x=0, y=0), None)
    assert report.in_view == 0
    with pytest.raises(ValidationError):
        observer.observe(embedded(hashing, "x", embedding=None, model=""), None)
    with pytest.raises(ValidationError):
        ContradictionPolicy(misses_to_supersede=0)


def test_ingester_runs_the_observer(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    policy = ContradictionPolicy(confirm_similarity=0.6, misses_to_supersede=1, visit_gap_s=0)
    ingester = Ingester(hashing, store, captioner=FakeCaptioner("a plain wall"), observer=Observer(store, policy))
    store.upsert([embedded(hashing, "a red fire extinguisher on the wall", t=0, x=0, y=0)])
    report = ingester.ingest([Observation("r1", "front", 100.0, Pose(0.1, 0), frame("f.jpg"))])
    assert report.inserted == 1 and report.contradicted == 1
    assert store.count() == 1 and store.count(EVERYTHING) == 2
