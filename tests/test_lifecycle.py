from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from placecell import (
    ContradictionPolicy,
    Curator,
    Evidence,
    EvidenceKind,
    Filter,
    Ingester,
    InMemoryStore,
    Observation,
    Observer,
    Pose,
    Recall,
    ReinforcementPolicy,
    Reinforcer,
    RetentionPolicy,
    Sighting,
)
from placecell.errors import ValidationError
from placecell.lifecycle import remove_local_file
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
from tests.conftest import FakeCaptioner, embedded, frame


def test_reinforcer_merges_same_thing_at_same_place(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    r = Reinforcer(store, ReinforcementPolicy(radius_m=1.0, min_similarity=0.95, gain=0.5))
    first, merged = r.reinforce_or_insert(embedded(hashing, "printer on the left", t=100, x=0, y=0, confidence=0.5))
    assert not merged and store.count() == 1
    repeat = embedded(hashing, "printer on the left", t=200, x=0.5, y=0, camera="back")
    stored, merged = r.reinforce_or_insert(repeat)
    assert merged and store.count() == 1
    assert stored.id == first.id and stored.observations == 2 and stored.last_seen == 200
    assert stored.confidence == pytest.approx(0.75)
    assert stored.evidence == repeat.evidence  # newest picture wins
    assert stored.caption == "printer on the left"
    stored, merged = r.reinforce_or_insert(embedded(hashing, "printer on the left", t=300, x=0.5, y=0))
    assert stored.confidence == pytest.approx(0.875) and stored.observations == 3


def test_reinforcer_keeps_different_things_and_places_apart(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    r = Reinforcer(store, ReinforcementPolicy(radius_m=1.0, min_similarity=0.95))
    r.reinforce_or_insert(embedded(hashing, "printer on the left", t=1, x=0, y=0))
    _, merged = r.reinforce_or_insert(embedded(hashing, "printer on the left", t=2, x=5, y=0))
    assert not merged  # same thing, other place
    _, merged = r.reinforce_or_insert(embedded(hashing, "a blue sofa", t=3, x=0, y=0))
    assert not merged  # same place, other thing
    assert store.count() == 3


def test_reinforcer_is_idempotent_on_replay_and_needs_vectors(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    r = Reinforcer(store)
    m = embedded(hashing, "a", t=1)
    r.reinforce_or_insert(m)
    again, merged = r.reinforce_or_insert(embedded(hashing, "a", t=1, confidence=0.1))
    assert merged and again == m and again.confidence == 1.0 and store.count() == 1
    with pytest.raises(ValidationError):
        r.reinforce_or_insert(embedded(hashing, "a", embedding=None, model=""))
    with pytest.raises(ValidationError):
        ReinforcementPolicy(gain=0)


def test_reinforcing_a_superseded_memory_revives_it(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    r = Reinforcer(store, ReinforcementPolicy(min_similarity=0.95))
    store.upsert([embedded(hashing, "a chair", t=1, superseded=True)])
    assert store.count() == 0
    stored, merged = r.reinforce_or_insert(embedded(hashing, "a chair", t=2, x=0.1))
    assert merged and not stored.superseded and stored.observations == 2
    assert store.count() == 1 and store.count(EVERYTHING) == 1


def test_merged_sightings_remain_idempotent_after_restart(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    first = embedded(hashing, "printer", t=100)
    repeat = embedded(hashing, "printer", t=200, camera="back")
    reinforcer = Reinforcer(store)
    reinforcer.reinforce_or_insert(first)
    stored, _ = reinforcer.reinforce_or_insert(repeat)
    again, _ = Reinforcer(store).reinforce_or_insert(repeat)
    assert again == stored and again.observations == 2
    assert again.sightings == (Sighting(first.id, 100), Sighting(repeat.id, 200))
    assert store.query(Filter(observation_id=repeat.id)) == [again]
    assert store.count() == 1


def test_reinforced_times_are_queryable_without_filling_the_gaps(
    store: InMemoryStore, hashing: HashingEmbedder
) -> None:
    reinforcer = Reinforcer(store)
    for timestamp in (100.25, 200.5, 300.75):
        reinforcer.reinforce_or_insert(embedded(hashing, "printer", t=timestamp))
    recall = Recall(store, hashing, clock=lambda: 301)
    assert recall.between(200.5, 300.75)[0].observed_at == (200.5,)
    assert recall.between(100.25, 300.75)[0].observed_at == (100.25, 200.5)
    assert recall.between(150, 190) == []
    assert recall.between(200.5, 200.5) == []
    assert store.count(Filter(time_from=301)) == 0
    assert store.count(Filter(time_to=100.25)) == 0
    assert store.count(Filter(time_from=300.75)) == 1
    assert store.count(Filter(time_to=100.26)) == 1
    # A different camera may see the same thing at exactly the same time.
    reinforcer.reinforce_or_insert(embedded(hashing, "printer", t=200.5, camera="back"))
    assert recall.between(200.5, 200.6)[0].observed_at == (200.5,)
    store.upsert([embedded(hashing, "chair", t=250, x=4)])
    # Limits apply after ordering sightings inside the window, not by the original timestamp.
    assert recall.between(210, 400, limit=1)[0].memory.caption == "chair"


def test_confirmed_and_revived_memories_start_with_no_misses(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    old = embedded(hashing, "printer", t=100, misses=2, last_miss=150)
    store.upsert([old])
    policy = ContradictionPolicy(visit_gap_s=10)
    ingester = Ingester(hashing, store, FakeCaptioner("printer"), observer=Observer(store, policy))
    ingester.ingest([Observation("r1", "front", 200, Pose(0, 0), frame())])
    confirmed = store.get(old.id)
    assert confirmed is not None and confirmed.misses == 0 and confirmed.last_miss == 0
    store.upsert([replace(confirmed, superseded=True, superseded_at=250, misses=3, last_miss=250)])
    revived, _ = Reinforcer(store).reinforce_or_insert(embedded(hashing, "printer", t=300))
    assert not revived.superseded and revived.superseded_at is None and revived.misses == 0
    result = Observer(store, policy).observe(embedded(hashing, "blank wall", t=400))
    assert result.missed == 1 and result.superseded == 0


def test_contradiction_grace_starts_when_the_memory_is_superseded(
    store: InMemoryStore, hashing: HashingEmbedder
) -> None:
    old = embedded(hashing, "printer", t=100)
    store.upsert([old])
    now = 3 * 86400.0
    Observer(store, ContradictionPolicy(misses_to_supersede=1)).observe(embedded(hashing, "blank wall", t=now))
    gone = store.get(old.id)
    assert gone is not None and gone.last_seen == 100 and gone.superseded_at == now
    curator = Curator(store, RetentionPolicy(drop_superseded_after_s=86400))
    assert curator.run(now=now).removed == 0
    assert curator.run(now=now + 86399).removed == 0
    assert curator.run(now=now + 86400).superseded_dropped == 1


def test_shared_evidence_survives_until_its_final_reference(
    store: InMemoryStore, hashing: HashingEmbedder, tmp_path: Path
) -> None:
    path = tmp_path / "shared.jpg"
    path.write_bytes(b"image")
    a = embedded(hashing, "printer", t=100, evidence=Evidence(EvidenceKind.FRAME, str(path)))
    b = embedded(
        hashing,
        "printer",
        t=200,
        role="summary",
        superseded=True,
        evidence=Evidence(EvidenceKind.FRAME, f"file://{path}"),
    )
    store.upsert([a, b])
    curator = Curator(store, remover=remove_local_file)
    assert curator.forget(Filter(time_from=100, time_to=101)) == 1
    assert path.exists() and store.get(b.id) is not None
    assert curator.forget(EVERYTHING) == 1
    assert not path.exists()


def test_curator_expires_ages_and_drops_superseded(
    store: InMemoryStore, hashing: HashingEmbedder, tmp_path: Path
) -> None:
    day = 86400.0
    old = tmp_path / "old.jpg"
    old.write_bytes(b"x")
    removed: list[Evidence] = []
    store.upsert(
        [
            embedded(hashing, "fresh", t=99 * day, x=0),
            embedded(hashing, "faded", t=1, x=1, evidence=Evidence(EvidenceKind.FRAME, str(old))),
            embedded(hashing, "reinforced", t=2, x=2, observations=5),
            embedded(hashing, "ancient", t=3, x=3, observations=50),
            embedded(hashing, "gone", t=4, x=4, superseded=True, last_seen=98 * day),
            embedded(hashing, "just gone", t=5, x=5, superseded=True, last_seen=99.5 * day),
        ]
    )
    policy = RetentionPolicy(
        half_life_s=7 * day, min_confidence=0.05, protected_observations=5, max_age_s=200 * day, max_idle_s=None
    )
    curator = Curator(store, policy, remover=removed.append, clock=lambda: 100 * day)
    report = curator.run()
    assert (report.scanned, report.expired, report.aged_out, report.superseded_dropped, report.removed) == (
        6,
        1,
        0,
        1,
        2,
    )
    assert {m.caption for m in store.query(EVERYTHING)} == {"fresh", "reinforced", "ancient", "just gone"}
    assert [e.uri for e in removed] == [str(old), "frames/front_4.jpg"]
    report = curator.run(now=250 * day)
    # the two old reinforced memories hit max_age despite their observation count, "fresh" has decayed away,
    # and the recently superseded one is past its grace period
    assert (report.expired, report.aged_out, report.superseded_dropped) == (1, 2, 1)
    assert store.count(EVERYTHING) == 0
    assert curator.run().removed == 0  # a pass over an empty store is a no-op
    remove_local_file(Evidence(EvidenceKind.FRAME, f"file://{old}"))
    assert not old.exists()
    remove_local_file(Evidence(EvidenceKind.FRAME, str(old)))  # missing file is fine


def test_curator_scope_supersede_and_forget(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    store.upsert([embedded(hashing, "a", t=0, robot="r1"), embedded(hashing, "b", t=0, robot="r2")])
    curator = Curator(store, RetentionPolicy(half_life_s=1.0, min_confidence=0.5), clock=lambda: 100.0)
    assert curator.run(scope=Filter(robot_id="r1")).expired == 1
    assert store.count() == 1
    b = store.query()[0]
    marked = curator.supersede(b.id, now=200.0)
    assert marked is not None and marked.superseded and marked.last_seen == 0.0 and marked.superseded_at == 200.0
    assert curator.supersede(b.id, now=250.0) == marked  # repeated calls do not restart the grace period
    assert curator.supersede("missing") is None
    assert store.count() == 0 and store.count(EVERYTHING) == 1
    assert curator.forget(Filter(near=Pose(0, 0), radius=1, include_superseded=True)) == 1
    assert store.count(EVERYTHING) == 0
    with pytest.raises(ValidationError):
        RetentionPolicy(min_confidence=2)
