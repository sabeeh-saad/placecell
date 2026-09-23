"""Day 13: retained data, ownership, correction durability and prompt continuity."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from placecell import CollectionInfo, Curator, Evidence, EvidenceKind, InMemoryStore, MissionContext, Pose, Reinforcer
from placecell.corrections import Correction, InMemoryCorrectionLog, JsonlCorrectionLog, Verdicts
from placecell.errors import ValidationError
from placecell.lifecycle import remove_local_file
from placecell.pipeline import Observation
from placecell.store.base import EVERYTHING
from placecell.store.limits import StoreLimits
from tests.conftest import embedded


@pytest.fixture(params=["memory", "disk"])
def bounded_store(request, tmp_path, hashing):
    stores = []

    def create(**kwargs):
        info = CollectionInfo("bounded", hashing.model_name, hashing.dimension)
        limits = StoreLimits(**kwargs)
        if request.param == "memory":
            store = InMemoryStore(info, limits=limits)
        else:
            from placecell.store.lancedb_store import LanceDBStore

            store = LanceDBStore(tmp_path / f"db-{len(stores)}", info, limits=limits)
        stores.append(store)
        return store

    yield create
    for store in stores:
        store.close()


def test_memory_capacity_is_atomic_and_existing_records_still_update(bounded_store, hashing):
    store = bounded_store(max_memories=2)
    first = embedded(hashing, "printer", t=1)
    store.upsert([first])
    with pytest.raises(ValidationError, match="capacity"):
        store.upsert([embedded(hashing, "microwave", t=2), embedded(hashing, "chair", t=3)])
    assert store.count(EVERYTHING) == 1
    assert store.get(embedded(hashing, "microwave", t=2).id) is None
    store.upsert([embedded(hashing, "microwave", t=2)])
    store.upsert([replace(first, caption="corrected printer")])
    assert store.get(first.id).caption == "corrected printer"
    store.delete([first.id])
    store.upsert([embedded(hashing, "chair", t=3)])
    assert store.count(EVERYTHING) == 2


def test_concurrent_memory_admission_cannot_overshoot(bounded_store, hashing):
    store = bounded_store(max_memories=3)
    memories = [embedded(hashing, f"item {i}", t=i) for i in range(12)]

    def insert(memory):
        try:
            store.upsert([memory])
            return True
        except ValidationError:
            return False

    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(insert, memories))
    assert sum(results) == store.count(EVERYTHING) == 3


def test_repeated_visits_bound_history_and_age_pruning_cannot_be_undone(bounded_store, hashing):
    store = bounded_store(max_sightings=8)
    reinforcer = Reinforcer(store, remover=None)
    for t in range(80):
        reinforcer.reinforce_or_insert(embedded(hashing, "printer", t=t))
    before = store.query()[0]
    assert before.observations == 80
    assert [s.timestamp for s in store.sightings(before.id, limit=100)] == list(range(72, 80))
    assert store.prune_history(78) == 6
    # A caption correction captured before pruning must not restore its old preview.
    store.upsert([replace(before, caption="corrected printer")])
    assert [s.timestamp for s in store.sightings(before.id)] == [78, 79]
    assert store.prune_history(1000) == 1
    store.upsert([before])
    assert [s.timestamp for s in store.sightings(before.id)] == [79]
    assert store.get(before.id).observations == 80


def test_cleanup_capacity_rolls_back_memory_deletion(bounded_store, hashing, tmp_path):
    store = bounded_store(max_cleanup=1)
    a = embedded(hashing, "a", t=1, evidence=Evidence(EvidenceKind.FRAME, str(tmp_path / "a"), managed=True))
    b = embedded(hashing, "b", t=2, evidence=Evidence(EvidenceKind.FRAME, str(tmp_path / "b"), managed=True))
    store.upsert([a, b])
    store.delete([a.id])
    with pytest.raises(ValidationError, match="cleanup capacity"):
        store.delete([b.id])
    assert store.get(b.id) is not None
    removed = []
    assert store.drain_cleanup(removed.append) == 1
    store.delete([b.id])
    assert store.drain_cleanup(removed.append) == 1
    assert removed == [a.evidence, b.evidence]


def test_retention_preserves_job_ownership_until_completion(bounded_store, hashing, tmp_path):
    store = bounded_store(max_memories=1, max_cleanup=1)
    path = tmp_path / "owned.jpg"
    path.write_bytes(b"frame")
    evidence = Evidence(EvidenceKind.FRAME, str(path), managed=True)
    memory = embedded(hashing, "printer", t=1, evidence=evidence)
    observation = Observation("r1", "front", 1, Pose(0, 0), evidence)
    store.upsert([memory])
    assert store.jobs.enqueue(observation, 1)
    job = store.jobs.pending(1)[0]
    store.jobs.fail([job.id], "provider unavailable", max_attempts=1)
    assert store.jobs.stats()["failed"] == 1
    assert not store.jobs.enqueue(replace(observation, timestamp=2), 1)
    Curator(store, remover=remove_local_file).run(now=100 * 86400)
    assert store.count(EVERYTHING) == 0 and path.exists()
    assert store.jobs.retry_failed() == 1
    store.jobs.complete([job.id])
    assert store.drain_cleanup(remove_local_file) == 1 and not path.exists()


def test_refinement_queue_coalesces_at_capacity(bounded_store, hashing):
    store = bounded_store(max_refinement_jobs=1)
    a, b = embedded(hashing, "a", t=1), embedded(hashing, "b", t=2)
    store.upsert([a, b])
    assert store.refinements.request(a.id)
    assert not store.refinements.request(b.id)
    assert store.refinements.request(a.id, "updated correction")
    assert len(store.refinements.pending()) == 1
    store.delete([a.id])
    assert store.refinements.request(b.id)


def test_context_limits_apply_across_scopes_and_survive_reopen(tmp_path):
    path = tmp_path / "context.db"
    now = [100.0]
    a = MissionContext(path, scope="a", max_events=4, clock=lambda: now[0])
    b = MissionContext(path, scope="b", max_events=4, clock=lambda: now[0])
    for i in range(6):
        context = a if i % 2 == 0 else b
        context.record(str(i), "instruction", {"text": f"visit item {i}"})
        context.record(str(i), "status", {"state": "succeeded"})
        assert context.stats()["events"] <= 4
    assert {e["request_id"] for e in a.recent()} == {"4"}
    assert {e["request_id"] for e in b.recent()} == {"5"}
    assert "history_boundary" in a.recent()[0]
    assert a.stats()["pruned_events"] == 8
    a.close()
    b.close()
    reopened = MissionContext(path, scope="a", max_events=4, clock=lambda: now[0])
    try:
        assert reopened.stats()["events"] == 4
        assert {e["request_id"] for e in reopened.recent()} == {"4"}
    finally:
        reopened.close()


def test_context_age_and_bytes_are_enforced_on_reads(tmp_path):
    now = [100.0]
    context = MissionContext(tmp_path / "context.db", max_bytes=1200, retention_s=10, clock=lambda: now[0])
    try:
        context.record("old", "instruction", {"text": "a" * 600})
        context.record("new", "instruction", {"text": "b" * 600})
        assert context.stats()["bytes"] <= 1200
        assert {e["request_id"] for e in context.recent()} == {"new"}
        now[0] = 111
        assert context.recent()[0]["kind"] == "retention_boundary"
        assert context.stats()["events"] == 0
    finally:
        context.close()


def test_one_oversized_request_is_refused_without_losing_previous_context(tmp_path):
    context = MissionContext(tmp_path / "context.db", max_events=2)
    try:
        context.record("old", "instruction", {"text": "printer"})
        context.record("active", "instruction", {"text": "microwave"})
        context.record("active", "status", {"state": "planned"})
        before = context.recent()
        with pytest.raises(ValidationError, match="capacity"):
            context.record("active", "status", {"state": "navigating"})
        assert context.recent() == before and context.stats()["events"] == 2
    finally:
        context.close()


def test_deleted_latest_reference_never_falls_back_to_older_destination(tmp_path):
    live = {"printer", "microwave"}
    context = MissionContext(
        tmp_path / "context.db",
        references_available=lambda data: not data.get("object_id") or data["object_id"] in live,
    )
    try:
        for name in ("printer", "microwave"):
            context.record(name, "instruction", {"text": f"go to {name}"})
            context.record(name, "status", {"state": "succeeded", "object_id": name})
        live.remove("microwave")
        assert len(context.recent()) == 1
        assert context.recent()[0]["kind"] == "retention_boundary"
        assert "printer" not in json.dumps(context.recent()) and "microwave" not in json.dumps(context.recent())
        context.record("explicit", "instruction", {"text": "go home"})
        assert [e["request_id"] for e in context.recent()] == ["explicit"]
        assert "history_boundary" in context.recent()[0]
    finally:
        context.close()


def test_oversized_recent_event_cannot_expose_older_history():
    context = MissionContext()
    try:
        context.record("old", "instruction", {"text": "go printer"})
        context.record("new", "instruction", {"text": "x" * 2000})
        assert context.recent(max_chars=1000)[0]["kind"] == "retention_boundary"
        assert "printer" not in json.dumps(context.recent(max_chars=1000))
    finally:
        context.close()


@pytest.mark.parametrize("persistent", [False, True])
def test_correction_bounds_preserve_retained_negative_verdicts(tmp_path, persistent):
    path = tmp_path / "corrections.jsonl"
    log = JsonlCorrectionLog(path, max_records=3) if persistent else InMemoryCorrectionLog(max_records=3)
    for _ in range(3):
        log.record(Correction("printer", "wrong"))
    with pytest.raises(ValidationError, match="capacity"):
        log.record(Correction("other", "right"))
    assert log.prune(["printer"]) == 0
    assert log.verdicts(["printer"])["printer"] == Verdicts(wrong=3)
    assert log.prune(["other"]) == 3
    log.record(Correction("other", "right"))
    if persistent:
        log = JsonlCorrectionLog(path, max_records=3)
    assert len(log) == 1 and log.verdicts(["printer"]) == {}


def test_failed_correction_replace_changes_neither_file_nor_verdicts(tmp_path, monkeypatch):
    path = tmp_path / "corrections.jsonl"
    log = JsonlCorrectionLog(path)
    log.record(Correction("printer", "wrong"))
    before = path.read_bytes()

    def denied(*_):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(OSError, match="disk full"):
        log.record(Correction("printer", "right"))
    assert path.read_bytes() == before
    assert log.verdicts(["printer"]) == {"printer": Verdicts(wrong=1)}
    assert list(tmp_path.iterdir()) == [path]


def test_legacy_oversized_correction_file_is_refused_intact(tmp_path):
    path = tmp_path / "corrections.jsonl"
    log = JsonlCorrectionLog(path)
    for _ in range(3):
        log.record(Correction("printer", "wrong"))
    before = path.read_bytes()
    with pytest.raises(ValidationError, match="record capacity"):
        JsonlCorrectionLog(path, max_records=2)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "policy",
    [
        lambda: StoreLimits(max_memories=0),
        lambda: StoreLimits(max_cleanup=True),
        lambda: MissionContext(max_events=0),
        lambda: MissionContext(retention_s=float("nan")),
        lambda: InMemoryCorrectionLog(max_bytes=1),
    ],
)
def test_invalid_limits_are_refused(policy):
    with pytest.raises(ValidationError):
        policy()


def test_sighting_watermark_survives_persistent_restart(tmp_path, hashing):
    from placecell.store.lancedb_store import LanceDBStore

    info = CollectionInfo("history", hashing.model_name, hashing.dimension)
    store = LanceDBStore(tmp_path, info)
    reinforcer = Reinforcer(store, remover=None)
    for t in range(5):
        reinforcer.reinforce_or_insert(embedded(hashing, "printer", t=t))
    stale = store.query()[0]
    store.prune_history(4)
    store.close()
    reopened = LanceDBStore(tmp_path, info)
    try:
        reopened.upsert([replace(stale, caption="updated")])
        assert [s.timestamp for s in reopened.sightings(stale.id)] == [4]
        with sqlite3.connect(tmp_path / "history.state.sqlite3") as db:
            assert db.execute("SELECT history_before FROM memories").fetchone()[0] == 4
    finally:
        reopened.close()


def test_automatic_retention_does_not_delete_externally_owned_evidence(bounded_store, hashing, tmp_path):
    path = tmp_path / "user-recording.jpg"
    path.write_bytes(b"external frame")
    store = bounded_store()
    memory = embedded(hashing, "printer", t=1, evidence=Evidence(EvidenceKind.FRAME, str(path)))
    store.upsert([memory])
    assert Curator(store, remover=remove_local_file).run(now=100 * 86400).removed == 1
    assert path.exists()


def test_context_aging_is_not_blocked_by_an_active_request_in_another_scope(tmp_path):
    now = [100.0]
    path = tmp_path / "contexts.db"
    a = MissionContext(path, scope="active", retention_s=10, clock=lambda: now[0])
    b = MissionContext(path, scope="inactive", retention_s=10, clock=lambda: now[0])
    try:
        a.record("a", "instruction", {"text": "printer"})
        b.record("b", "instruction", {"text": "microwave"})
        now[0] = 109
        a.record("a", "status", {"state": "navigating"})
        now[0] = 111
        assert b.recent()[0]["kind"] == "retention_boundary"
        assert len(a.recent()) == 2
    finally:
        a.close()
        b.close()


def test_correction_byte_limit_counts_unicode_and_refuses_without_mutation(tmp_path):
    log = JsonlCorrectionLog(tmp_path / "feedback.jsonl", max_bytes=1024)
    log.record(Correction("printer", "wrong"))
    before = (tmp_path / "feedback.jsonl").read_bytes()
    with pytest.raises(ValidationError, match="capacity"):
        log.record(Correction("printer", "right", note="中" * 200))
    assert len(log) == 1
    assert (tmp_path / "feedback.jsonl").read_bytes() == before


def test_context_clock_and_payload_bounds():
    with pytest.raises(ValidationError, match="clock"):
        MissionContext(clock=lambda: float("nan"))
    context = MissionContext()
    try:
        with pytest.raises(ValidationError, match="request id"):
            context.record("x" * 257, "instruction", {})
        with pytest.raises(ValidationError, match="scope"):
            MissionContext(scope="x" * 513)
    finally:
        context.close()


def test_curator_makes_progress_with_a_one_item_cleanup_budget(bounded_store, hashing, tmp_path):
    store = bounded_store(max_cleanup=1)
    paths = [tmp_path / f"old-{i}.jpg" for i in range(4)]
    for i, path in enumerate(paths):
        path.write_bytes(b"old managed image")
        store.upsert(
            [embedded(hashing, str(i), t=i + 1, evidence=Evidence(EvidenceKind.FRAME, str(path), managed=True))]
        )
    assert Curator(store, remover=remove_local_file).run(now=100 * 86400).removed == 4
    assert store.count(EVERYTHING) == 0 and all(not path.exists() for path in paths)


def test_in_memory_close_refuses_overflow_without_deleting_files_or_losing_job_evidence(hashing, tmp_path):
    store = InMemoryStore(
        CollectionInfo("close", hashing.model_name, hashing.dimension), limits=StoreLimits(max_cleanup=2)
    )
    paths = []
    for i in range(5):
        path = tmp_path / f"close-{i}.jpg"
        paths.append(path)
        path.write_bytes(b"managed")
        memory = embedded(hashing, str(i), t=i + 1, evidence=Evidence(EvidenceKind.FRAME, str(path), managed=True))
        store.upsert([memory])
    assert store.jobs.enqueue(Observation("r1", "front", 5, memory.pose, memory.evidence), 1)
    with pytest.raises(ValidationError, match="cleanup capacity"):
        store.close()
    assert store.count(EVERYTHING) == 5 and all(path.exists() for path in paths)
    Curator(store, remover=remove_local_file).forget(EVERYTHING)
    store.close()
    assert store.count(EVERYTHING) == 0 and paths[-1].exists()
    assert not any(path.exists() for path in paths[:-1])
    assert store.jobs.stats()["queued"] == 1
    store.jobs.complete([job.id for job in store.jobs.pending(1)])
    store.drain_cleanup(remove_local_file)
    assert not paths[-1].exists()


def test_explicit_forget_drains_small_batches(bounded_store, hashing, tmp_path):
    store = bounded_store(max_cleanup=1)
    paths = []
    for i in range(3):
        path = tmp_path / f"forget-{i}.jpg"
        paths.append(path)
        path.write_bytes(b"managed")
        store.upsert(
            [embedded(hashing, str(i), t=i + 1, evidence=Evidence(EvidenceKind.FRAME, str(path), managed=True))]
        )
    assert Curator(store, remover=remove_local_file).forget(EVERYTHING) == 3
    assert all(not path.exists() for path in paths)


def test_curator_reserves_cleanup_room_for_job_owned_frames(bounded_store, hashing, tmp_path):
    store = bounded_store(max_cleanup=2)
    paths = []
    for i in range(4):
        path = tmp_path / f"reserved-{i}.jpg"
        paths.append(path)
        path.write_bytes(b"managed")
        memory = embedded(hashing, str(i), t=i + 1, evidence=Evidence(EvidenceKind.FRAME, str(path), managed=True))
        store.upsert([memory])
        if i == 0:
            assert store.jobs.enqueue(Observation("r1", "front", 1, memory.pose, memory.evidence), 1)
    assert Curator(store, remover=remove_local_file).run(now=100 * 86400).removed == 4
    assert paths[0].exists() and not any(path.exists() for path in paths[1:])
    store.jobs.complete([job.id for job in store.jobs.pending(1)])
    assert store.drain_cleanup(remove_local_file) == 1
