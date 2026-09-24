from __future__ import annotations

import gc
import threading
import time
import weakref
from dataclasses import replace
from email.utils import formatdate
from types import SimpleNamespace

import pytest

from placecell import CollectionInfo, Ingester, Pose
from placecell.errors import ProviderError, ValidationError
from placecell.providers import OpenAICompatibleEmbedder
from placecell.ros2.bridge import KeyframeWriter, ObservationBuilder
from placecell.ros2.depth import PendingImages
from placecell.ros2.node import BoundedTasks, IngestWorker, build_embedder
from placecell.store.state import StateStore
from tests.conftest import FakeCaptioner, FakeTransport
from tests.test_ros2_bridge import _Log


def wait(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        threading.Event().wait(0.005)


def test_concurrent_flood_is_bounded_and_shutdown_discards_waiting_payloads():
    tasks = BoundedTasks(1, 3, _Log())
    entered, release = threading.Event(), threading.Event()
    ran = []

    def blocked():
        entered.set()
        assert release.wait(5)

    try:
        assert tasks.submit(blocked) and entered.wait(2)
        threads = [
            threading.Thread(target=lambda: [tasks.submit(ran.append, 1) for _ in range(1000)]) for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()
        health = tasks.health()
        assert health["active"] == 1 and health["queued"] == health["high_water"] == 3
        assert health["accepted"] == 4 and health["rejected_full"] == 7997
        assert not tasks.stop(timeout=0)
        assert tasks.health()["discarded"] == 3 and tasks.health()["queued"] == 0
        assert not tasks.submit(ran.append, 2)
    finally:
        release.set()
        assert tasks.stop()
    assert not ran
    assert tasks.health()["accepted"] == tasks.health()["completed"] + tasks.health()["discarded"]


def test_coalescing_covers_running_and_queued_maintenance_but_not_new_commands():
    tasks = BoundedTasks(1, 2, _Log())
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked():
        entered.set()
        assert release.wait(5)

    try:
        assert tasks.submit(blocked, key="refine") and entered.wait(2)
        for _ in range(1000):
            assert tasks.submit(blocked, key="refine")
        assert tasks.submit(calls.append, "curate", key="curate")
        for _ in range(1000):
            assert tasks.submit(calls.append, "curate", key="curate")
        assert tasks.submit(calls.append, "explicit")
        assert not tasks.submit(calls.append, "explicit")
        assert tasks.health()["coalesced"] == 2000
        release.set()
        wait(lambda: tasks.health()["completed"] == 3)
        assert calls == ["curate", "explicit"]
        assert tasks.submit(calls.append, "curate", key="curate")
        wait(lambda: tasks.health()["completed"] == 4)
    finally:
        release.set()
        assert tasks.stop()


def test_idle_worker_releases_completed_request_payload():
    class Payload:
        pass

    tasks = BoundedTasks(1, 1, _Log())
    body = Payload()
    reference = weakref.ref(body)
    try:
        assert tasks.submit(lambda item: None, body)
        del body
        wait(lambda: tasks.health()["completed"] == 1)
        gc.collect()
        assert reference() is None
        assert tasks.submit(lambda: (_ for _ in ()).throw(ProviderError("offline")))
        wait(lambda: tasks.health()["failed"] == 1)
        assert tasks.submit(lambda: None)
        wait(lambda: tasks.health()["completed"] == 2)
    finally:
        assert tasks.stop()


def test_persistent_cooldown_survives_failed_head_restart_and_explicit_retry(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("placecell.store.jobs.time.time", lambda: clock[0])
    info = CollectionInfo("c", "hashing-16", 16)
    path = tmp_path / "state.sqlite3"
    store = StateStore(info, path)
    builder = ObservationBuilder("r", "c", KeyframeWriter(tmp_path / "images"))
    first = builder.from_compressed(100, "jpeg", b"one", Pose(0, 0))
    second = builder.from_compressed(200, "jpeg", b"two", Pose(1, 0))
    store.jobs.enqueue(first, 2)
    store.jobs.enqueue(second, 2)
    first_id = store.jobs.pending(1)[0].id
    store.jobs.fail([first_id], "429", max_attempts=1, defer_queue=True, retry_after_s=60)
    assert store.jobs.stats()["failed"] == 1 and store.jobs.stats()["retry_wait_s"] == 60
    assert store.jobs.pending(2) == []
    store.close()
    reopened = StateStore(info, path)
    try:
        clock[0] = 1020
        assert reopened.jobs.pending(2) == []
        assert reopened.jobs.retry_failed() == 1
        assert reopened.jobs.pending(2) == []  # Manual retry cannot bypass a provider's cooldown.
        clock[0] = 1060
        assert [job.observation.timestamp for job in reopened.jobs.pending(2)] == [100, 200]
    finally:
        reopened.close()


def test_ingestion_flood_retains_only_accepted_images_and_recovers_without_duplicates(tmp_path, hashing):
    entered, release = threading.Event(), threading.Event()

    class Caption(FakeCaptioner):
        def caption(self, items):
            entered.set()
            assert release.wait(5)
            return super().caption(items)

    store = StateStore(CollectionInfo("c", hashing.model_name, hashing.dimension), tmp_path / "state.sqlite3")
    caption = Caption("printer")
    worker = IngestWorker(Ingester(hashing, store, caption), None, 1, 4, _Log())
    writer = KeyframeWriter(tmp_path / "images")
    builder = ObservationBuilder("r", "c", writer)
    observations = []
    try:
        first = builder.from_compressed(100, "jpeg", b"first", Pose(0, 0))
        assert worker.submit(first)
        writer.confirm(first.evidence)
        worker.start()
        assert entered.wait(2)
        for i in range(1, 101):
            obs = builder.from_compressed(100 + i, "jpeg", b"image", Pose(i * 10, 0))
            observations.append((obs, worker.submit(obs)))
            writer.confirm(obs.evidence)
        assert sum(accepted for _, accepted in observations) == 3
        assert worker.health()["queued"] == 4 and worker.health()["dropped"] == 97
        assert len(list((tmp_path / "images").glob("*.jpg"))) == 4
        release.set()
        wait(lambda: store.jobs.stats()["queued"] == 0)
        assert store.count() == 4 and len(caption.calls) == 4
        assert all(m.observations == 1 for m in store.query())
    finally:
        release.set()
        assert worker.stop()
        store.close()


@pytest.mark.parametrize("status", [429, 503])
def test_long_retry_after_is_preserved_without_early_retry(status):
    transport = FakeTransport([(status, {"Retry-After": "60"}, {})])
    sleeps = []
    embed = OpenAICompatibleEmbedder("m", transport=transport, sleep=sleeps.append)
    with pytest.raises(ProviderError) as error:
        embed.embed_text(["printer"])
    assert error.value.retry_after_s == 60
    assert len(transport.requests) == 1 and not sleeps


def test_ros_embedder_uses_one_http_attempt_per_durable_attempt(monkeypatch):
    from placecell.providers._http import UrllibTransport

    calls = []

    def fail(*args):
        calls.append(1)
        return 503, {}, {}

    monkeypatch.setattr(UrllibTransport, "post_json", fail)
    embed = build_embedder("https://example.invalid/v1", "m", None, 16)
    with pytest.raises(ProviderError):
        embed.embed_text(["printer"])
    assert len(calls) == 1


def test_http_date_retry_after_is_not_shortened():
    transport = FakeTransport([(429, {"Retry-After": formatdate(time.time() + 120, usegmt=True)}, {})])
    embed = OpenAICompatibleEmbedder("m", transport=transport)
    with pytest.raises(ProviderError) as error:
        embed.embed_text(["printer"])
    assert 118 <= error.value.retry_after_s <= 120 and len(transport.requests) == 1


def test_rgb_depth_buffers_reject_oversized_packets_and_count_overwrites():
    buffers = PendingImages(capacity=2, max_message_bytes=1024)

    def message(stamp, size=1024):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp, nanosec=0), frame_id="camera"), data=b"x" * size
        )

    for i in range(1, 101):
        assert buffers.add(message(i), False, i)
        assert buffers.add_depth(message(i))
        assert buffers.add_depth(message(i), calibration=True)
    assert not buffers.add(message(101, 1025), True, 101)
    assert not buffers.add_depth(message(101, 1025))
    health = buffers.health()
    assert health["rgb_queued"] == health["depth_queued"] == health["info_queued"] == 2
    assert health["rgb_overwritten"] == health["depth_overwritten"] == health["info_overwritten"] == 98
    assert health["oversized"] == 2
    assert buffers.pop(101)[0].header.stamp.sec == 99


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_ingest_retry_configuration_refused(value, store, hashing):
    with pytest.raises(ValidationError):
        IngestWorker(Ingester(hashing, store), None, 1, 4, _Log(), retry_delay_s=value)
    with pytest.raises(ValidationError):
        store.jobs.fail([], "invalid", retry_delay_s=value)


def test_last_failed_job_cannot_bypass_provider_cooldown_for_next_job(tmp_path, hashing):
    failed = threading.Event()
    calls = []

    class Caption:
        def caption(self, items):
            calls.append(items)
            raise ProviderError("offline rate limit", retry_after_s=30)

    class Log(_Log):
        def error(self, message):
            failed.set()

    store = StateStore(CollectionInfo("c", hashing.model_name, hashing.dimension), tmp_path / "state.sqlite3")
    builder = ObservationBuilder("r", "c", KeyframeWriter(tmp_path / "images"))
    worker = IngestWorker(Ingester(hashing, store, Caption()), None, 1, 4, Log(), max_attempts=1)
    first = builder.from_compressed(100, "jpeg", b"one", Pose(0, 0))
    assert worker.submit(first)
    assert worker.submit(replace(first, timestamp=200))
    worker.start()
    try:
        assert failed.wait(2)
        assert worker.health()["retry_wait_s"] > 29 and worker.health()["failed"] == 1
        assert store.jobs.pending(2) == [] and len(calls) == 1
    finally:
        assert worker.stop()
        store.close()


def test_provider_failure_after_committed_batch_still_defers_next_work(store, tmp_path):
    builder = ObservationBuilder("r", "c", KeyframeWriter(tmp_path / "images"))
    observation = builder.from_compressed(100, "jpeg", b"one", Pose(0, 0))
    assert store.jobs.enqueue(observation, 4)
    # No failed IDs remain when the worker recognizes already committed memories.
    store.jobs.fail([], "failed after commit", defer_queue=True, retry_after_s=60)
    assert store.jobs.stats()["retry_wait_s"] > 59
    assert store.jobs.pending(4) == [] and store.jobs.stats()["failed"] == 0
