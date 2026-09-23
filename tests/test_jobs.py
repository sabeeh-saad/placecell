from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from placecell import CollectionInfo, Ingester, Pose, VectorStore
from placecell.errors import ProviderError, ValidationError
from placecell.lifecycle import remove_local_file
from placecell.pipeline import Segmenter
from placecell.providers import HashingEmbedder
from placecell.ros2.bridge import KeyframeWriter, ObservationBuilder
from placecell.ros2.node import BoundedTasks, IngestWorker
from tests.conftest import FakeCaptioner
from tests.test_ros2_bridge import _Log


def test_journal_is_bounded_orders_retries_and_protects_evidence(store: VectorStore, tmp_path: Path) -> None:
    builder = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path))
    first = builder.from_compressed(1, "jpeg", b"one", Pose(0, 0))
    second = builder.from_compressed(2, "jpeg", b"two", Pose(1, 0))
    assert store.jobs.enqueue(first, 1) and store.jobs.enqueue(first, 1)
    assert not store.jobs.enqueue(second, 1)
    assert store.jobs.enqueue(second, 2)
    jobs = store.jobs.pending(2)
    store.enqueue_cleanup([first.evidence])
    assert store.drain_cleanup(remove_local_file) == 0 and Path(first.evidence.uri).exists()
    store.jobs.fail([jobs[0].id], "temporary", retry_delay_s=10)
    assert store.jobs.pending(2) == []
    assert len(store.jobs.pending(2, now=10**12)) == 2
    store.jobs.fail([jobs[0].id], "persistent", max_attempts=2)
    assert store.jobs.stats()["failed"] == 1 and store.jobs.failed()[0]["error"] == "persistent"
    assert store.jobs.retry_failed() == 1
    store.jobs.complete([jobs[0].id, "missing"])
    assert store.drain_cleanup(remove_local_file) == 1 and not Path(first.evidence.uri).exists()
    assert store.jobs.pending(2)[0].observation == second
    with pytest.raises(ValidationError):
        store.jobs.enqueue(first, 0)
    with pytest.raises(ValidationError):
        store.jobs.pending(0)
    with pytest.raises(ValidationError):
        store.jobs.fail([], "", max_attempts=0)


def test_jobs_and_cleanup_resume_after_reopening(tmp_path: Path, hashing: HashingEmbedder) -> None:
    pytest.importorskip("lancedb")
    from placecell.store.lancedb_store import LanceDBStore

    info = CollectionInfo("resume", hashing.model_name, hashing.dimension)
    store = LanceDBStore(tmp_path / "db", info)
    observation = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path / "frames")).from_compressed(
        100, "jpeg", b"image", Pose(0, 0)
    )
    observation = replace(observation, localization_checked=True, refresh_objects=True)
    store.jobs.enqueue(observation, 10)
    store.close()
    reopened = LanceDBStore.open(tmp_path / "db", info.name)
    assert reopened.jobs.pending(1)[0].observation == observation
    reopened.jobs.complete(job.id for job in reopened.jobs.pending(1))
    reopened.close()
    final = LanceDBStore.open(tmp_path / "db", info.name)
    assert final.drain_cleanup(remove_local_file) == 1 and not Path(observation.evidence.uri).exists()
    final.close()


def test_provider_wait_does_not_block_submission_or_state_reads(
    store: VectorStore,
    hashing: HashingEmbedder,
    tmp_path: Path,
) -> None:
    entered, release, submitted = threading.Event(), threading.Event(), threading.Event()

    class BlockingCaptioner(FakeCaptioner):
        def caption(self, items):
            entered.set()
            assert release.wait(5)
            return super().caption(items)

    lock = threading.Lock()
    worker = IngestWorker(Ingester(hashing, store, BlockingCaptioner("printer")), lock, 1, 1, _Log())
    builder = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path))
    first = builder.from_compressed(100, "jpeg", b"one", Pose(0, 0))
    second = builder.from_compressed(200, "jpeg", b"two", Pose(1, 0))
    worker.submit(first)
    worker.start()

    def submit_full():
        assert store.count() == 0
        assert not worker.submit(second)
        submitted.set()

    try:
        assert entered.wait(5)
        assert lock.acquire(blocking=False)
        lock.release()
        thread = threading.Thread(target=submit_full)
        thread.start()
        assert submitted.wait(2), "provider call blocked the callback or state read"
        thread.join(2)
    finally:
        release.set()
        assert worker.stop()
    assert store.count() == 1


def test_worker_retries_without_double_counting(store: VectorStore, hashing: HashingEmbedder, tmp_path: Path) -> None:
    class FailOnce(FakeCaptioner):
        def caption(self, items):
            if not self.calls:
                self.calls.append(list(items))
                raise ProviderError("temporary")
            return super().caption(items)

    done = threading.Event()

    class Log(_Log):
        def info(self, message):
            super().info(message)
            done.set()

    worker = IngestWorker(Ingester(hashing, store, FailOnce("printer")), None, 1, 4, Log(), retry_delay_s=0)
    obs = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path)).from_compressed(100, "jpeg", b"image", Pose(0, 0))
    worker.submit(obs)
    worker.start()
    try:
        assert done.wait(5)
    finally:
        worker.stop()
    assert store.query()[0].observations == 1 and store.jobs.stats()["queued"] == 0


def test_stationary_refresh_and_admission_before_writing(tmp_path: Path) -> None:
    gate = Segmenter()
    builder = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path))
    assert gate.accept(builder.from_compressed(0, "jpeg", b"one", Pose(0, 0)))
    with patch.object(builder, "from_compressed", wraps=builder.from_compressed) as build:
        for t in (1, 20, 59, 60):
            if gate.eligible("r1", "front", t, Pose(0, 0)):
                gate.accept(builder.from_compressed(t, "jpeg", b"new scene", Pose(0, 0)))
        assert build.call_count == 1
    assert gate.eligible("r1", "front", 61, Pose(0, 0, map_id="new-map"))


def test_creation_markers_recover_only_unowned_images(store: VectorStore, tmp_path: Path) -> None:
    writer = KeyframeWriter(tmp_path)
    builder = ObservationBuilder("r1", "front", writer)
    orphan = builder.from_compressed(1, "jpeg", b"orphan", Pose(0, 0))
    pending = builder.from_compressed(2, "jpeg", b"pending", Pose(1, 0))
    store.jobs.enqueue(pending, 10)
    unrelated = tmp_path / "user.jpg"
    unrelated.write_bytes(b"user")
    KeyframeWriter(tmp_path).recover_pending(store)
    store.drain_cleanup(remove_local_file)
    assert not Path(orphan.evidence.uri).exists()
    assert Path(pending.evidence.uri).exists() and unrelated.exists()
    assert list(tmp_path.glob("*.pending")) == []
    writer.confirm(pending.evidence)


def test_background_tasks_have_a_fixed_worker_and_queue_limit() -> None:
    entered, release = threading.Event(), threading.Event()
    tasks = BoundedTasks(1, 1, _Log())

    def block():
        entered.set()
        release.wait(5)

    try:
        assert tasks.submit(block) and entered.wait(2)
        assert tasks.submit(block)
        assert not tasks.submit(block)
    finally:
        release.set()
        assert tasks.stop()
    assert not tasks.submit(block)
