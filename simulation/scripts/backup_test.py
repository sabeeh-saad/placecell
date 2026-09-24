"""Production ROS node maintenance exclusion and restored-startup admission. No model calls."""

import argparse
import json
import traceback
from pathlib import Path
from unittest.mock import patch

import rclpy

from placecell import Memory, Pose
from placecell.backup import create_backup, restore_backup, verify_backup
from placecell.corrections import Correction
from placecell.errors import ValidationError
from placecell.maintenance import STORAGE_KEYS, StorageLease
from placecell.providers import HashingEmbedder
from placecell.ros2.bridge import KeyframeWriter
from placecell.ros2.node import create_node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    root = output / "source"
    root.mkdir()
    params = {
        "db_path": str(root / "db"),
        "collection": "backup",
        "keyframe_dir": str(root / "keyframes"),
        "corrections_path": str(root / "corrections.jsonl"),
        "command_journal_path": str(root / "commands.sqlite3"),
        "mission_context_path": str(root / "missions.sqlite3"),
        "navigation_ownership_path": str(root / "navigation.sqlite3"),
        "mission_trace_path": str(root / "traces.sqlite3"),
        "map_id": "office-v1",
        "navigation_enabled": True,
        "mission_enabled": True,
        "mission_model": "scripted",
        "curator_interval_s": 0.0,
        "refine_interval_s": 0.0,
    }
    profile = {key: params[key] for key in (*STORAGE_KEYS, "collection")}
    parameters = output / "parameters.yaml"
    parameters.write_text(json.dumps({"placecell": {"ros__parameters": params}}))
    report = {"passed": False, "checks": [], "paid_api_calls": 0}
    node = None
    try:
        with patch(
            "placecell.providers._http.Endpoint.post", side_effect=AssertionError("unexpected model call")
        ) as api:
            rclpy.init(args=["--ros-args", "--params-file", str(parameters)])
            node = create_node()
            try:
                create_backup(profile, output / "busy-backup", confirm_stopped=True)
                raise AssertionError("Running node must exclude backup")
            except ValidationError as exc:
                assert "in use" in str(exc)
                report["checks"].append("running production node excludes backup before any snapshot is published")
            embed = HashingEmbedder()
            writer = KeyframeWriter(root / "keyframes")
            evidence = writer.write_jpeg("front", 100, b"offline ROS backup image")
            memory = Memory.create("robot", "front", 100, Pose(1, 2, map_id="office-v1"), evidence, "printer")
            node._store.upsert([memory.with_embedding(embed.embed_text([memory.caption])[0], embed.model_name)])
            writer.confirm(evidence)
            node._corrections.record(Correction(memory.id, "wrong", timestamp=100))
            node.destroy_node()
            node = None
            rclpy.shutdown()
            create_backup(profile, output / "backup", confirm_stopped=True)
            report["checks"].append("graceful shutdown releases maintenance leases and permits a complete backup")
            assert verify_backup(output / "backup")["counts"]["memories"] == 1
            restore_backup(output / "backup", output / "fresh")
            rclpy.init(
                args=[
                    "--ros-args",
                    "--params-file",
                    str(parameters),
                    "--params-file",
                    str(output / "fresh" / "restore-parameters.yaml"),
                ]
            )
            node = create_node()
            assert node._store.get(memory.id).evidence.uri.startswith(str(output / "fresh" / "keyframes"))
            assert node._commands.snapshot().busy and node._commands.snapshot().status.state == "uncertain"
            assert node._command_journal.scope.conversation_id.startswith("restored-")
            assert node._mission_context.recent() == []
            assert node._navigator._trip is None
            report["checks"].append("generated parameter overlay starts a fresh node against restored data")
            report["checks"].append("restored navigation is uncertain and busy without sending any goal")
            report["checks"].append("new command session isolates old history and stale commands")
            node.destroy_node()
            node = None
            rclpy.shutdown()
            # Same constructor used at production startup: active maintenance refuses admission.
            restored_profile = json.loads((output / "fresh" / "storage-profile.json").read_text())
            with StorageLease.for_parameters(restored_profile):
                rclpy.init(
                    args=[
                        "--ros-args",
                        "--params-file",
                        str(parameters),
                        "--params-file",
                        str(output / "fresh" / "restore-parameters.yaml"),
                    ]
                )
                try:
                    node = create_node()
                    raise AssertionError("Node must refuse storage under maintenance")
                except ValidationError as exc:
                    assert "in use" in str(exc)
                    report["checks"].append("production node refuses startup while maintenance owns its storage")
            assert api.call_count == 0
            report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
