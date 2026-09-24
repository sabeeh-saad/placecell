"""Repeat real SIGKILL checkpoints around durable ingestion and evidence ownership."""

from __future__ import annotations

import argparse
import json
import selectors
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from placecell import CollectionInfo, Memory, Pose
from placecell.lifecycle import remove_local_file
from placecell.pipeline import Ingester, Observation
from placecell.providers import HashingEmbedder
from placecell.ros2.bridge import KeyframeWriter
from placecell.store.state import StateStore

POINTS = (
    "image_marker",
    "image_renamed",
    "job_accepted",
    "caption_in_flight",
    "embedding_in_flight",
    "memory_uncommitted",
    "memory_committed",
    "job_completed",
    "cleanup_before_unlink",
    "cleanup_after_unlink",
    "failed_job",
)
LANCE_POINTS = ("memory_committed", "vector_written", "cleanup_after_unlink")
BODY = b"deterministic local evidence; provider fixture does not decode images"


def open_store(root, backend):
    embed = HashingEmbedder(64)
    info = CollectionInfo("recovery", embed.model_name, embed.dimension)
    if backend == "lancedb":
        from placecell.store.lancedb_store import LanceDBStore

        return LanceDBStore(root / "db", info)
    return StateStore(info, root / "state.sqlite3")


def child(root, backend, target):
    def checkpoint(point):
        if point == target:
            print(json.dumps({"checkpoint": point}), flush=True)  # noqa: T201 - parent handshake
            signal.pause()
            raise AssertionError("Crash checkpoint must be killed by its parent")

    store = open_store(root, backend)
    embed = HashingEmbedder(64)
    baseline = Memory.create("r", "c", 1, Pose(-100, -100), caption="acknowledged baseline")
    store.upsert([baseline.with_embedding(embed.embed_text([baseline.caption])[0], embed.model_name)])
    writer = KeyframeWriter(root / "images")
    original_sync = writer._sync_directory
    syncs = 0

    def sync_directory():
        nonlocal syncs
        original_sync()
        syncs += 1
        checkpoint("image_marker" if syncs == 1 else "image_renamed")

    writer._sync_directory = sync_directory
    evidence = writer.write_jpeg("c", 100, BODY)
    observation = Observation("r", "c", 100, Pose(10, 10), evidence, localization_checked=True)
    assert store.jobs.enqueue(observation, 4)
    checkpoint("job_accepted")
    writer.confirm(evidence)
    if target == "failed_job":
        store.jobs.fail([store.jobs.pending(1)[0].id], "scripted provider failure", max_attempts=1)
        checkpoint("failed_job")

    class Caption:
        def caption(self, items):
            checkpoint("caption_in_flight")
            assert Path(items[0].uri).read_bytes() == BODY
            return ["target product"]

    original_embed = embed.embed_text

    def embed_text(texts):
        checkpoint("embedding_in_flight")
        return original_embed(texts)

    embed.embed_text = embed_text
    ingester = Ingester(embed, store, Caption())
    original_persist = ingester.persist

    def persist(memory):
        result = original_persist(memory)
        checkpoint("memory_uncommitted")
        return result

    ingester.persist = persist
    ingester.ingest([observation], preselected=True)
    checkpoint("memory_committed")
    if target == "vector_written":
        # Crash after Lance has committed its projection but before SQLite acknowledges it.
        original_merge = store._table.merge_insert

        def merge(key):
            builder = original_merge(key)
            original_execute = builder.execute

            def execute(rows):
                result = original_execute(rows)
                checkpoint("vector_written")
                return result

            builder.execute = execute
            return builder

        store._table.merge_insert = merge
        store._sync_index()
    store.jobs.complete(job.id for job in store.jobs.pending(4))
    checkpoint("job_completed")
    memory = next(m for m in store.query() if m.id != baseline.id)
    store.delete([memory.id])

    def remove(item):
        checkpoint("cleanup_before_unlink")
        remove_local_file(item)
        checkpoint("cleanup_after_unlink")

    store.drain_cleanup(remove)
    raise AssertionError(f"Checkpoint not reached: {target}")


def verify(root, backend, point):
    store = open_store(root, backend)
    try:
        assert store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        baseline = next(m for m in store.query() if m.timestamp == 1)
        assert baseline.caption == "acknowledged baseline" and baseline.observations == 1
        writer = KeyframeWriter(root / "images")
        writer.recover_pending(store)
        store.drain_cleanup(remove_local_file)
        before = store.jobs.stats()
        image = root / "images/c_100000.jpg"
        acknowledged = point not in {"image_marker", "image_renamed"}
        deleted = point in {"cleanup_before_unlink", "cleanup_after_unlink"}
        completed = point == "job_completed" or deleted
        assert before["queued"] == int(acknowledged and not completed)
        assert image.exists() == (acknowledged and not deleted)
        calls = []

        class Caption:
            def caption(self, items):
                calls.append("caption")
                assert Path(items[0].uri).read_bytes() == BODY
                return ["target product"]

        if point == "failed_job":
            assert before["failed"] == 1 and not store.jobs.pending(4)
            assert store.jobs.failed()[0]["attempts"] == 1
            assert store.jobs.retry_failed() == 1  # explicit retry, never automatic
        ingester = Ingester(HashingEmbedder(64), store, Caption())
        jobs = store.jobs.pending(4)
        if jobs:
            ingester.ingest([job.observation for job in jobs], preselected=True)
            store.jobs.complete(job.id for job in jobs)
            ingester.discard([])
        assert store.jobs.stats()["queued"] == 0
        expected = 1 + int(acknowledged and not deleted)
        assert store.count() == expected
        if expected == 2:
            target = next(m for m in store.query() if m.id != baseline.id)
            assert target.observations == 1 and len(target.sightings) == 1
            assert target.evidence and Path(target.evidence.uri).read_bytes() == BODY
            assert store.search(HashingEmbedder(64).embed_text(["target product"])[0], 1)[0].memory.id == target.id
        if point in {"memory_committed", "vector_written", "job_completed"}:
            assert not calls, "Committed observations must not repeat provider work"
        assert not list((root / "images").glob("*.pending"))
        assert not list((root / "images").glob("*.tmp"))
        return {"acknowledged_job": acknowledged, "recovery_caption_calls": len(calls), "memories": expected}
    finally:
        store.close()


def run_case(root, backend, point):
    root.mkdir(parents=True, exist_ok=False)
    with (root / "child.log").open("w") as log:
        process = subprocess.Popen(  # noqa: S603 - fixed local crash fixture
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                point,
                "--backend",
                backend,
                "--output",
                str(root),
            ],
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(45), "Child did not reach its crash checkpoint"
                event = json.loads(process.stdout.readline())
            assert event == {"checkpoint": point}, event
            process.kill()
            process.wait(10)
            assert process.returncode == -signal.SIGKILL
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(10)
            process.stdout.close()
    return {"exit_code": process.returncode, **verify(root, backend, point)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--lancedb", action="store_true")
    parser.add_argument("--child", choices=(*POINTS, "vector_written"))
    parser.add_argument("--backend", choices=("sqlite", "lancedb"), default="sqlite")
    args = parser.parse_args()
    if args.child:
        child(args.output, args.backend, args.child)
        return
    if args.repeat < 1:
        parser.error("repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "paid_api_calls": 0, "signal": "SIGKILL", "cases": []}
    started = time.monotonic()
    cases = [("sqlite", p) for p in POINTS]
    if args.lancedb:
        cases += [("lancedb", p) for p in LANCE_POINTS]
    for repeat in range(args.repeat):
        for backend, point in cases:
            row = {"backend": backend, "checkpoint": point, "repeat": repeat, "passed": False}
            try:
                row.update(run_case(args.output / f"{repeat}-{backend}-{point}", backend, point))
                row["passed"] = True
            except Exception:
                row["error"] = traceback.format_exc()
            report["cases"].append(row)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["passed"] = all(row["passed"] for row in report["cases"])
    report["duration_s"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}))  # noqa: T201
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
