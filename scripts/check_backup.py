"""Offline fresh-process backup/restore, SIGKILL, and previous-revision rollback drill."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def seed(root):
    from placecell import CollectionInfo, Memory, Pose
    from placecell.command_identity import CommandJournal, CommandScope, IdentifiedCommand
    from placecell.corrections import Correction, JsonlCorrectionLog
    from placecell.mission_context import MissionContext
    from placecell.pipeline import Observation
    from placecell.providers import HashingEmbedder
    from placecell.ros2.bridge import KeyframeWriter
    from placecell.store.lancedb_store import LanceDBStore

    root.mkdir()
    embed = HashingEmbedder(16)
    store = LanceDBStore(root / "db", CollectionInfo("office", embed.model_name, 16))
    writer = KeyframeWriter(root / "keyframes")
    evidence = writer.write_jpeg("front", 100, b"offline recovery evidence")
    memory = Memory.create("robot", "front", 100, Pose(1, 2, map_id="office-v1"), evidence, "printer")
    store.upsert([memory.with_embedding(embed.embed_text([memory.caption])[0], embed.model_name)])
    writer.confirm(evidence)
    second = writer.write_jpeg("front", 101, b"pending accepted job")
    store.jobs.enqueue(Observation("robot", "front", 101, memory.pose, second), 4)
    store.close()
    JsonlCorrectionLog(root / "corrections.jsonl").record(Correction(memory.id, "wrong", timestamp=100))
    scope = CommandScope("robot", "office-v1", "original")
    commands = CommandJournal(root / "commands.sqlite3", scope)
    receipt = commands.claim(IdentifiedCommand("original", scope, time.time(), "instruction", "visit printer"))
    commands.close()
    context = MissionContext(root / "missions.sqlite3")
    context.record(receipt.request_id, "instruction", {"text": "visit printer"})
    context.close()


def inspect(root):
    from placecell.providers import HashingEmbedder
    from placecell.store.lancedb_store import LanceDBStore

    store = LanceDBStore.open(root / "db", "office")
    hit = store.search(HashingEmbedder(16).embed_text(["printer"])[0], 1)[0]
    assert hit.memory.caption == "printer" and hit.score > 0.99
    assert Path(hit.memory.evidence.uri).parent == root / "keyframes"
    assert Path(hit.memory.evidence.uri).read_bytes() == b"offline recovery evidence"
    assert store.jobs.stats()["queued"] == 1
    assert Path(store.jobs.pending(1)[0].observation.evidence.uri).read_bytes() == b"pending accepted job"
    store.close()


def checkpoint(name):
    print(json.dumps({"checkpoint": name}), flush=True)  # noqa: T201 - parent handshake
    signal.pause()
    raise AssertionError("Parent must kill this process")


def worker(args):
    if args.worker == "seed":
        seed(args.root)
    elif args.worker == "inspect":
        inspect(args.root)
    elif args.worker == "upgrade":
        import placecell.store.state as state

        original = state.ObjectJournal

        def interrupted(*params):
            result = original(*params)
            checkpoint("upgrade-before-version-commit")
            return result

        state.ObjectJournal = interrupted
        inspect(args.root)
    else:
        import placecell.backup as backup

        if args.point == "copy":
            original = backup._copy_tree

            def copied(*params):
                original(*params)
                checkpoint("copy")

            backup._copy_tree = copied
        else:
            original_publish = backup._publish

            def published(stage, destination):
                if args.point == "before-publish":
                    checkpoint(args.point)
                original_publish(stage, destination)
                checkpoint(args.point)

            backup._publish = published
        if args.worker == "backup":
            backup.create_backup(json.loads(args.profile.read_text()), args.root, confirm_stopped=True)
        else:
            backup.restore_backup(args.backup, args.root)


def run(args, worker_name, root, *, legacy=False, point=None):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", worker_name, "--root", str(root)]
    if point:
        command += [
            "--point",
            point,
            "--profile",
            str(args.output / "profile.json"),
            "--backup",
            str(args.output / "baseline-backup"),
        ]
    env = dict(os.environ)
    if legacy:
        env["PYTHONPATH"] = str(args.legacy_source / "src")
    log = args.output / f"{root.name}-{worker_name}.log"
    with log.open("w") as stderr:
        child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=stderr, text=True)  # noqa: S603
        try:
            if not point and worker_name != "upgrade":
                stdout, _ = child.communicate(timeout=60)
                assert child.returncode == 0, (command, stderr.name, stdout)
                return {"exit": 0, "log": str(log)}
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                assert selector.select(60), f"No crash checkpoint: {log}"
                message = json.loads(child.stdout.readline())
            child.kill()
            child.wait(timeout=10)
            assert child.returncode == -signal.SIGKILL
            return {"exit": child.returncode, **message, "log": str(log)}
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)


def campaign(args):
    from placecell.backup import create_backup, restore_backup, verify_backup

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "paid_api_calls": 0,
        "cases": [],
        "legacy_source": str(args.legacy_source) if args.legacy_source else None,
    }
    report_path = args.output / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    source = args.output / "source"
    run(args, "seed", source, legacy=bool(args.legacy_source))
    with sqlite3.connect(source / "db" / "office.state.sqlite3") as connection:
        report["original_state_version"] = connection.execute("PRAGMA user_version").fetchone()[0]
    profile = {
        "db_path": str(source / "db"),
        "keyframe_dir": str(source / "keyframes"),
        "collection": "office",
        "corrections_path": str(source / "corrections.jsonl"),
        "command_journal_path": str(source / "commands.sqlite3"),
        "mission_context_path": str(source / "missions.sqlite3"),
        "navigation_ownership_path": "",
        "mission_trace_path": "",
    }
    (args.output / "profile.json").write_text(json.dumps(profile))
    snapshot = args.output / "baseline-backup"
    create_backup(profile, snapshot, confirm_stopped=True)
    for repeat in range(args.repeat):
        for operation, points in (
            ("backup", ("copy", "before-publish", "after-publish")),
            ("restore", ("before-publish", "after-publish")),
        ):
            for point in points:
                root = args.output / f"{operation}-{repeat}-{point}"
                outcome = run(args, operation, root, point=point)
                assert root.exists() == (point == "after-publish")
                if root.exists():
                    if operation == "backup":
                        verify_backup(root)
                    else:
                        run(args, "inspect", root)
                verify_backup(snapshot)
                report["cases"].append({"case": operation, "point": point, "repeat": repeat, "passed": True, **outcome})
                save()
        upgrade = args.output / f"upgrade-{repeat}"
        restore_backup(snapshot, upgrade)
        outcome = run(args, "upgrade", upgrade)
        run(args, "inspect", upgrade)
        with sqlite3.connect(upgrade / "db" / "office.state.sqlite3") as connection:
            assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        rollback = args.output / f"rollback-{repeat}"
        restore_backup(snapshot, rollback)
        with sqlite3.connect(rollback / "db" / "office.state.sqlite3") as connection:
            assert connection.execute("PRAGMA user_version").fetchone() == (report["original_state_version"],)
        run(args, "inspect", rollback, legacy=bool(args.legacy_source))
        report["cases"].append(
            {
                "case": "interrupted-upgrade-and-backup-rollback",
                "repeat": repeat,
                "passed": True,
                "previous_revision_tested": bool(args.legacy_source),
                **outcome,
            }
        )
        save()
    report["passed"] = True
    report["trials"] = len(report["cases"])
    save()
    print(json.dumps(report, indent=2))  # noqa: T201 - CLI report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--worker", choices=("seed", "inspect", "backup", "restore", "upgrade"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--point")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.output is None or not 1 <= args.repeat <= 20:
            parser.error("--output and --repeat in 1..20 are required")
        campaign(args)


if __name__ == "__main__":
    main()
