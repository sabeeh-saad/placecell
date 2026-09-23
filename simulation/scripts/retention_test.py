"""Bounded retention through the production ROS node, SQLite/LanceDB and DDS corrections.

Uses authored memory observations and controlled aging; no camera, Nav2 movement or model calls.
"""

import argparse
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from placecell import Evidence, EvidenceKind, Memory, Pose, Reinforcer
from placecell.depth import Box
from placecell.errors import ValidationError
from placecell.object_types import ObjectRecord, ObjectView
from placecell.pipeline import Observation
from placecell.providers import HashingEmbedder
from placecell.ros2.node import create_node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    params = {
        "db_path": str(output / "db"),
        "collection": "retention",
        "map_id": "retention-v1",
        "keyframe_dir": str(output / "images"),
        "corrections_path": str(output / "corrections.jsonl"),
        "navigation_enabled": True,
        "mission_enabled": True,
        "mission_model": "scripted",
        "mission_context_path": str(output / "context.sqlite3"),
        "command_journal_path": str(output / "commands.sqlite3"),
        "curator_interval_s": 0.0,
        "refine_interval_s": 0.0,
        "contradiction": False,
        "memory_max_records": 3,
        "memory_max_sightings": 8,
        "memory_max_idle_s": 60.0,
        "memory_history_age_s": 30.0,
        "max_queue": 2,
        "cleanup_max_pending": 4,
        "refine_max_pending": 1,
        "correction_max_records": 4,
        "correction_max_bytes": 1024,
        "mission_context_max_events": 6,
        "mission_context_max_bytes": 2048,
    }
    parameters = output / "parameters.yaml"
    parameters.write_text(yaml.safe_dump({"placecell": {"ros__parameters": params}}))
    report = {"passed": False, "checks": [], "measurements": {}, "paid_api_calls": 0}
    node = probe = executor = None
    rclpy.init(args=["--ros-args", "--params-file", str(parameters)])
    try:
        with patch(
            "placecell.providers._http.Endpoint.post", side_effect=AssertionError("unexpected model call")
        ) as api:
            node = create_node()
            # Hold jobs for deterministic capacity/ownership checks; normal retention is unchanged.
            assert node._worker.stop()
            executor = SingleThreadedExecutor()
            probe = Node("retention_probe")
            executor.add_node(probe)
            executor.add_node(node)
            publisher = probe.create_publisher(String, "/placecell/correct", 10)

            def wait(predicate, timeout=10):
                deadline = time.monotonic() + timeout
                while not predicate():
                    if time.monotonic() > deadline:
                        raise TimeoutError("DDS retention check timed out")
                    executor.spin_once(timeout_sec=0.02)

            def passed(name):
                report["checks"].append(name)

            embedder, store = HashingEmbedder(), node._store
            assert store.limits.max_memories == 3 and store.limits.max_sightings == 8
            start = time.time() - 120
            reinforcer = Reinforcer(store)
            for i in range(120):
                path = output / "images" / f"visit-{i}.jpg"
                path.write_bytes(b"authored retention fixture")
                memory = Memory.create(
                    "robot",
                    "front",
                    start + i,
                    Pose(0, 0, map_id="retention-v1"),
                    Evidence(EvidenceKind.FRAME, str(path), managed=True),
                    "printer",
                )
                memory = memory.with_embedding(embedder.embed_text(["printer"])[0], embedder.model_name)
                retained, _ = reinforcer.reinforce_or_insert(memory)
            assert retained.observations == 120 and len(store.sightings(retained.id)) == 8
            assert len(list((output / "images").glob("visit-*.jpg"))) == 1
            report["measurements"].update(visits=120, retained_sightings=8, retained_visit_images=1)
            passed("120 revisits retain one identity, eight sightings and one managed image")

            others = [
                replace(
                    memory,
                    id=f"other-{i}",
                    evidence=None,
                    view_timestamp=None,
                    timestamp=start + 121,
                    last_seen=start + 121,
                    sightings=(),
                )
                for i in range(2)
            ]
            store.upsert(others)
            try:
                store.upsert([replace(others[0], id="overflow")])
                raise AssertionError("memory capacity was ignored")
            except ValidationError:
                pass
            assert store.count() == 3
            passed("configured memory capacity refuses a fourth record without evicting existing targets")

            wait(lambda: publisher.get_subscription_count() == 1)
            for i in range(4):
                identity = retained.id if i < 3 else others[0].id
                publisher.publish(String(data=json.dumps({"memory_id": identity, "verdict": "wrong"})))
                wait(lambda i=i: len(node._corrections) == i + 1)
            publisher.publish(String(data=json.dumps({"memory_id": retained.id, "verdict": "right"})))
            until = time.monotonic() + 0.3
            while time.monotonic() < until:
                executor.spin_once(timeout_sec=0.02)
            assert len(node._corrections) == 4
            assert node._corrections.verdicts([retained.id])[retained.id].wrong == 3
            assert len(store.refinements.pending()) == 1
            passed("DDS correction overflow preserves negative feedback and bounds coalesced refinement requests")

            observation = Observation("robot", "front", start + 120, retained.pose, retained.evidence)
            assert store.jobs.enqueue(observation, 2)
            assert store.jobs.enqueue(replace(observation, timestamp=start + 121), 2)
            jobs = store.jobs.pending(2)
            store.jobs.fail([j.id for j in jobs], "scripted outage", max_attempts=1)
            assert not store.jobs.enqueue(replace(observation, timestamp=start + 122), 2)
            assert store.jobs.stats()["queued"] == store.jobs.stats()["failed"] == 2
            passed("failed and in-flight ownership consume the configured two-job capacity")

            for i in range(100):
                node._mission_context.record(str(i), "instruction", {"text": "visit printer"})
                node._mission_context.record(str(i), "status", {"state": "succeeded", "memory_id": retained.id})
            context_stats = node._mission_context.stats()
            assert context_stats["events"] == 6 and context_stats["bytes"] <= 2048
            assert context_stats["pruned_events"] == 194
            assert "history_boundary" in node._mission_context.recent()[0]
            report["measurements"].update(context_stats)
            passed("100 request histories retain only three whole requests within the row and byte budgets")

            store.delete([retained.id])
            node._run_curator()
            assert Path(retained.evidence.uri).exists()
            assert node._mission_context.recent()[0]["kind"] == "retention_boundary"
            assert len(node._corrections) == 1
            passed(
                "deleted context references cannot expose an older destination; only deleted-memory feedback is pruned"
            )
            assert store.jobs.retry_failed() == 2
            store.jobs.complete([j.id for j in jobs])
            node._run_curator()
            assert not Path(retained.evidence.uri).exists()
            passed(
                "maintenance preserves job-owned evidence until completion, then removes the final unreferenced image"
            )

            external = output / "external-recording.jpg"
            external.write_bytes(b"user-owned")
            old = replace(
                memory,
                id="expired",
                timestamp=start - 100,
                last_seen=start - 100,
                sightings=(),
                evidence=Evidence(EvidenceKind.FRAME, str(external)),
                view_timestamp=start - 100,
            )
            store.upsert([old])
            node._run_curator()
            assert store.get(old.id) is None and external.exists()
            passed("automatic aging removes expired memory while preserving externally owned media")

            object_paths = []
            object_time = time.time() - node._object_policy.retention_s - 60
            for i in range(6):
                path = output / "images" / f"expired-object-{i}.jpg"
                path.write_bytes(b"managed object fixture")
                object_paths.append(path)
                identity = f"expired-object-{i}"
                record = ObjectRecord(
                    identity, "robot", "front", "map", "retention-v1", "printer", object_time, object_time
                )
                view = replace(
                    memory,
                    id=f"view-{i}",
                    timestamp=object_time,
                    last_seen=object_time,
                    view_timestamp=object_time,
                    sightings=(),
                    evidence=Evidence(EvidenceKind.FRAME, str(path), managed=True),
                )
                store.objects.save(record, ObjectView(identity, view, Box(0, 0, 1, 1), b"fixture"))
            node._run_curator()
            assert store.objects.count() == 0 and all(not path.exists() for path in object_paths)
            passed("six expired objects are removed despite a four-entry cleanup limit")

            executor.remove_node(node)
            node.destroy_node()
            node = None
            node = create_node()
            assert node._worker.stop()
            assert node._store.count() == 2 and node._store.jobs.stats()["queued"] == 0
            assert len(node._corrections) == 1 and node._mission_context.stats()["events"] == 6
            assert node._mission_context.recent()[0]["kind"] == "retention_boundary"
            assert not node.navigation_busy()
            passed("node restart preserves bounds and deletion boundaries without resuming movement")
            assert api.call_count == 0
            report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if executor:
            executor.shutdown()
        if node:
            node.destroy_node()
        if probe:
            probe.destroy_node()
        rclpy.try_shutdown()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))  # noqa: T201 - standalone validation report
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
