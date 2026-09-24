"""Offline, verified deployment snapshots and fresh-directory restore.

The reference deployment has one Lance collection and local evidence. All writers
must be stopped; the ROS node's cooperative leases enforce this for that process.
Snapshots are directories, not executable archives, and contain no provider config.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from placecell import __version__
from placecell.corrections import Correction
from placecell.errors import ValidationError
from placecell.maintenance import STORAGE_KEYS, StorageLease
from placecell.memory import SCHEMA_VERSION
from placecell.navigation_ownership import NavigationScope
from placecell.store.base import CollectionInfo
from placecell.store.schema import check_connection

FORMAT = 1
FILES = {
    "corrections_path": "corrections.jsonl",
    "command_journal_path": "commands.sqlite3",
    "mission_context_path": "missions.sqlite3",
    "mission_trace_path": "traces.sqlite3",
    "navigation_ownership_path": "navigation.sqlite3",
}


def profile(data: dict[str, Any]) -> dict[str, str]:
    if set(data) != {*STORAGE_KEYS, "collection"} or any(not isinstance(v, str) for v in data.values()):
        raise ValidationError("Profile must explicitly contain collection and every documented storage path.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", data["collection"]):
        raise ValidationError("Invalid collection name.")
    result = {"collection": data["collection"]}
    for key in STORAGE_KEYS:
        value = data[key]
        if value == ":memory:" or (key in {"db_path", "keyframe_dir", "corrections_path"} and not value):
            raise ValidationError("Backup requires persistent memory, local evidence and a correction log path.")
        result[key] = str(Path(value).expanduser().resolve()) if value else ""
    paths = [Path(result[key]) for key in STORAGE_KEYS if result[key]]
    for index, first in enumerate(paths):
        if any(first == other or first in other.parents or other in first.parents for other in paths[index + 1 :]):
            raise ValidationError("Storage paths must be distinct and must not contain one another.")
    return result


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _files(root: Path) -> dict[str, Path]:
    result = {}
    if root.is_symlink() or not root.is_dir():
        raise ValidationError(f"Expected a real directory: {root}")
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValidationError(f"Links and special files are not supported: {path}")
        result[path.relative_to(root).as_posix()] = path
    return result


@contextmanager
def _database(path: Path, *, writable: bool = False) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path.resolve().as_uri() + ("?mode=rw" if writable else "?mode=ro"), uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValidationError(f"SQLite integrity failure: {path.name}")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValidationError(f"SQLite reference failure: {path.name}")
        yield connection
    finally:
        connection.close()


def _copy_sqlite(source: Path, target: Path) -> None:
    with _database(source) as connection:
        destination = sqlite3.connect(target)
        try:
            connection.backup(destination)
        finally:
            destination.close()
    with _database(target, writable=True) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")


def _copy_tree(source: Path, target: Path) -> None:
    target.mkdir(mode=0o700)
    files = _files(source)
    databases = {name for name in files if name.endswith(".sqlite3")}
    for name, path in files.items():
        if name.endswith((".lock", ".sqlite3-wal", ".sqlite3-shm", ".sqlite3-journal")):
            if name.endswith(".lock") or name.rsplit("-", 1)[0] in databases:
                continue
            raise ValidationError(f"Orphan SQLite sidecar: {name}")
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in databases:
            _copy_sqlite(path, destination)
        else:
            shutil.copyfile(path, destination)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _validate(root: Path, settings: dict[str, str]) -> dict[str, Any]:
    """Validate authoritative state and every live image reference, including failed jobs."""
    collection = settings["collection"]
    db_root = root / "db"
    if {p.name for p in db_root.glob("*.collection.json")} != {f"{collection}.collection.json"}:
        raise ValidationError("This workflow supports exactly one collection per database directory.")
    if {p.name for p in db_root.glob("*.state.sqlite3")} != {f"{collection}.state.sqlite3"}:
        raise ValidationError("Unexpected state database in the collection directory.")
    info = CollectionInfo(**_json(db_root / f"{collection}.collection.json"))
    if info.name != collection or info.schema_version != SCHEMA_VERSION:
        raise ValidationError(f"Backup supports collection schema {SCHEMA_VERSION}; use matching backup software.")
    original_images = Path(settings["keyframe_dir"])
    references = 0

    def evidence(uri: str, expected_digest: str = "", *, required: bool = True) -> None:
        nonlocal references
        if not uri:
            raise ValidationError("Referenced evidence has an empty URI.")
        path = Path(uri.removeprefix("file://"))
        if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(original_images):
            raise ValidationError(f"Evidence must be inside the configured keyframe directory: {uri}")
        target = root / "keyframes" / path.relative_to(original_images)
        if not target.is_file():
            if required:
                raise ValidationError(f"Missing referenced evidence: {uri}")
            return
        if expected_digest and digest(target) != expected_digest:
            raise ValidationError(f"Evidence digest mismatch: {uri}")
        references += 1

    counts: dict[str, Any] = {}
    with _database(db_root / f"{collection}.state.sqlite3") as connection:
        counts["state_schema"] = check_connection(connection)
        for table in ("memories", "sightings", "objects", "object_views", "jobs", "cleanup", "refinement_jobs"):
            counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
        for table in ("objects", "object_events"):
            for (payload,) in connection.execute(f"SELECT payload FROM {table}"):  # noqa: S608
                if not isinstance(json.loads(payload), dict):
                    raise ValidationError(f"Invalid {table} payload.")
        for table, column in (("memories", "evidence_uri"), ("object_views", "uri")):
            for uri, payload, vector, caption_vector in connection.execute(
                f"SELECT {column},payload,vector,caption_vector FROM {table}"  # noqa: S608
            ):
                row = json.loads(payload)
                if uri != row["evidence_uri"].removeprefix("file://"):
                    raise ValidationError("Evidence index and payload disagree.")
                if len(vector) != 4 * info.dimension or (caption_vector and len(caption_vector) != len(vector)):
                    raise ValidationError("Stored vector dimension mismatch.")
                if not np.isfinite(np.frombuffer(vector, dtype=np.float32)).all() or (
                    caption_vector and not np.isfinite(np.frombuffer(caption_vector, dtype=np.float32)).all()
                ):
                    raise ValidationError("Stored vector contains non-finite values.")
                if row["model"] != info.model or row["schema_version"] != SCHEMA_VERSION:
                    raise ValidationError("Stored memory model or schema mismatch.")
                if row["evidence_kind"]:
                    evidence(row["evidence_uri"], row["evidence_digest"])
                elif row["evidence_uri"]:
                    raise ValidationError("Evidence URI has no evidence kind.")
        for uri, payload in connection.execute("SELECT uri,payload FROM jobs"):
            row = json.loads(payload)["evidence"]
            if uri != row["uri"].removeprefix("file://"):
                raise ValidationError("Job evidence index and payload disagree.")
            evidence(row["uri"], row["digest"])
        for uri, payload in connection.execute("SELECT uri,payload FROM cleanup"):
            row = json.loads(payload)
            if uri != row["uri"].removeprefix("file://"):
                raise ValidationError("Cleanup evidence index and payload disagree.")
            evidence(row["uri"], row["digest"], required=False)
    counts["evidence_references"] = references
    for key, name in FILES.items():
        if not settings[key]:
            if (root / name).exists():
                raise ValidationError(f"Undeclared component: {name}")
            continue
        if key == "corrections_path":
            corrections = [
                Correction(**json.loads(line)) for line in (root / name).read_text().splitlines() if line.strip()
            ]
            counts["corrections"] = len(corrections)
            continue
        with _database(root / name) as connection:
            if key == "navigation_ownership_path":
                rows = connection.execute("SELECT version,scope,state,goal_id FROM ownership").fetchall()
                if len(rows) != 1 or rows[0][0] != 1 or rows[0][2] not in {"clean", "unknown", "pending"}:
                    raise ValidationError("Invalid navigation ownership journal.")
                NavigationScope(**json.loads(rows[0][1]))
                if rows[0][2] == "pending" and (
                    not rows[0][3] or uuid.UUID(rows[0][3]).int == 0 or uuid.UUID(rows[0][3]).hex != rows[0][3]
                ):
                    raise ValidationError("Invalid pending navigation identity.")
                counts["navigation"] = {"scope": json.loads(rows[0][1]), "state": rows[0][2]}
            elif key == "command_journal_path":
                if connection.execute("SELECT version FROM command_policy WHERE id=1").fetchone() != (1,):
                    raise ValidationError("Unsupported command journal.")
                counts["commands"] = connection.execute("SELECT COUNT(*) FROM command_claims").fetchone()[0]
            elif key == "mission_context_path":
                counts["context_events"] = 0
                for (payload,) in connection.execute("SELECT payload FROM mission_events"):
                    json.loads(payload)
                    counts["context_events"] += 1
            else:
                if connection.execute("SELECT value FROM trace_meta WHERE key='schema_version'").fetchone() != ("1",):
                    raise ValidationError("Unsupported trace schema.")
    return counts


def _sync_tree(root: Path) -> None:
    for path in _files(root).values():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    for path in [*sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True), root]:
        _sync_directory(path)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(stage: Path, destination: Path) -> None:
    # Linux reference platform: atomic directory publication that cannot replace
    # even an empty destination created by another process during the operation.
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(destination), 1)
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    _sync_directory(destination.parent)


@contextmanager
def _staging(destination: Path) -> Iterator[Path]:
    if destination.exists() or destination.is_symlink():
        raise ValidationError("Destination must be new; existing data is never overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.incomplete-", dir=destination.parent))
    try:
        yield stage
        _sync_tree(stage)
        _publish(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def create_backup(settings: dict[str, Any], destination: Path, *, confirm_stopped: bool = False) -> dict[str, Any]:
    settings = profile(settings)
    if not confirm_stopped:
        raise ValidationError("Stop all writers, including standalone scripts, and pass --confirm-stopped.")
    destination = destination.expanduser().resolve()
    for key in STORAGE_KEYS:
        if settings[key] and destination.is_relative_to(Path(settings[key])):
            raise ValidationError("Backup destination cannot be inside source storage.")
    with StorageLease.for_parameters(settings), _staging(destination) as stage:
        # Also exclude the ownership inspection/attestation CLI and standalone adapters.
        native_locks = (
            [Path(settings["navigation_ownership_path"] + ".lock")] if settings["navigation_ownership_path"] else []
        )
        if settings["mission_trace_path"] and Path(settings["mission_trace_path"]).is_file():
            native_locks.append(Path(settings["mission_trace_path"]))
        with StorageLease(native_locks):
            recovery_point = time.time()
            _copy_tree(Path(settings["db_path"]), stage / "db")
            _copy_tree(Path(settings["keyframe_dir"]), stage / "keyframes")
            for key, name in FILES.items():
                if settings[key]:
                    source = Path(settings[key])
                    if source.is_symlink() or not source.is_file():
                        raise ValidationError(f"Configured component is missing or a link: {key}")
                    if name.endswith(".sqlite3"):
                        _copy_sqlite(source, stage / name)
                    else:
                        shutil.copyfile(source, stage / name)
            counts = _validate(stage, settings)
            manifest = {
                "format": FORMAT,
                "producer_version": __version__,
                "profile": settings,
                "recovery_point_unix_s": recovery_point,
                "completed_unix_s": time.time(),
                "consistency": "offline; all configured writers stopped and cooperative leases held",
                "counts": counts,
                "files": {
                    name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in _files(stage).items()
                },
            }
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            (stage / "manifest.sha256").write_text(digest(stage / "manifest.json") + "\n")
    return manifest


def verify_backup(source: Path) -> dict[str, Any]:
    files = _files(source)
    if digest(source / "manifest.json") != (source / "manifest.sha256").read_text().strip():
        raise ValidationError("Backup manifest checksum mismatch.")
    manifest: dict[str, Any] = _json(source / "manifest.json")
    if type(manifest.get("format")) is not int or manifest["format"] != FORMAT:
        raise ValidationError("Unsupported backup format.")
    settings = profile(manifest["profile"])
    expected = manifest["files"]
    if set(files) != {*expected, "manifest.json", "manifest.sha256"}:
        raise ValidationError("Backup file inventory mismatch (missing or unexpected files).")
    for name, item in expected.items():
        if files[name].stat().st_size != item["bytes"] or digest(files[name]) != item["sha256"]:
            raise ValidationError(f"Backup file checksum mismatch: {name}")
    if _validate(source, settings) != manifest["counts"]:
        raise ValidationError("Backup state differs from its recorded inventory.")
    return manifest


def _relocate(state: Path, old: str, new: str) -> None:
    def uri(value: str) -> str:
        if not value:
            return value
        prefix = "file://" if value.startswith("file://") else ""
        return prefix + str(Path(new) / Path(value.removeprefix("file://")).relative_to(old))

    with _database(state, writable=True) as connection, connection:
        for table, column in (
            ("memories", "evidence_uri"),
            ("object_views", "uri"),
            ("jobs", "uri"),
            ("cleanup", "uri"),
        ):
            rows = connection.execute(f"SELECT rowid,{column},payload FROM {table}").fetchall()  # noqa: S608
            for rowid, value, payload in rows:
                data = json.loads(payload)
                if table == "jobs":
                    data["evidence"]["uri"] = uri(data["evidence"]["uri"])
                else:
                    field = "uri" if table == "cleanup" else "evidence_uri"
                    data[field] = uri(data[field])
                connection.execute(
                    f"UPDATE {table} SET {column}=?,payload=? WHERE rowid=?",  # noqa: S608
                    (uri(value), json.dumps(data), rowid),
                )
        connection.execute(
            "INSERT INTO dirty_vectors SELECT id,1 FROM memories WHERE 1 "
            "ON CONFLICT(id) DO UPDATE SET generation=generation+1"
        )


def restore_backup(source: Path, destination: Path) -> dict[str, Any]:
    destination = destination.expanduser().resolve()
    if destination.is_relative_to(source.expanduser().resolve()):
        raise ValidationError("Restore destination cannot be inside the backup.")
    manifest = verify_backup(source)
    old = manifest["profile"]
    new = {
        "collection": old["collection"],
        "db_path": str(destination / "db"),
        "keyframe_dir": str(destination / "keyframes"),
    }
    new.update({key: str(destination / name) if old[key] else "" for key, name in FILES.items()})
    with _staging(destination) as stage:
        # Verify the private copy too: a source changed during copying cannot be published.
        for name in (*manifest["files"], "manifest.json", "manifest.sha256"):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        (stage / "keyframes").mkdir(exist_ok=True)
        verify_backup(stage)
        _relocate(stage / "db" / f"{old['collection']}.state.sqlite3", old["keyframe_dir"], new["keyframe_dir"])
        epoch = "restored-" + uuid.uuid4().hex
        if old["command_journal_path"]:
            with _database(stage / FILES["command_journal_path"], writable=True) as connection, connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS command_restore "
                    "(id INTEGER PRIMARY KEY CHECK(id=1), epoch TEXT NOT NULL)"
                )
                connection.execute("INSERT OR REPLACE INTO command_restore VALUES (1,?)", (epoch,))
        if old["navigation_ownership_path"]:
            with _database(stage / FILES["navigation_ownership_path"], writable=True) as connection, connection:
                connection.execute(
                    "UPDATE ownership SET state='unknown',"
                    "outcome='Backup restored; independent Nav2 reset required' WHERE id=1"
                )
        counts = _validate(stage, new)
        report = {
            "format": FORMAT,
            "backup_sha256": digest(source / "manifest.json"),
            "recovery_point_unix_s": manifest["recovery_point_unix_s"],
            "restored_unix_s": time.time(),
            "counts": counts,
            "navigation": "unknown; supervised Nav2 reset required",
            "mission_conversation_id": epoch,
            "data_after_recovery_point": "not present; retained command history cannot deduplicate later commands",
        }
        (stage / "restore-report.json").write_text(json.dumps(report, indent=2) + "\n")
        (stage / "storage-profile.json").write_text(json.dumps(new, indent=2) + "\n")
        parameters = {**new, "mission_conversation_id": epoch}
        (stage / "restore-parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
        # JSON is a YAML subset; the ROS parameter loader accepts this overlay.
        (stage / "restore-parameters.yaml").write_text(
            json.dumps({"placecell": {"ros__parameters": parameters}}, indent=2) + "\n"
        )
        # The original snapshot remains immutable at source, with its own checksums.
        (stage / "manifest.json").unlink()
        (stage / "manifest.sha256").unlink()
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    create = commands.add_parser("create")
    create.add_argument("--profile", type=Path, required=True)
    create.add_argument("--destination", type=Path, required=True)
    create.add_argument("--confirm-stopped", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--backup", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "create":
            result = create_backup(_json(args.profile), args.destination, confirm_stopped=args.confirm_stopped)
        elif args.operation == "verify":
            result = verify_backup(args.backup)
        else:
            result = restore_backup(args.backup, args.destination)
    except (OSError, sqlite3.Error, ValidationError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Backup operation refused: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201 - CLI report


if __name__ == "__main__":  # pragma: no cover
    main()
