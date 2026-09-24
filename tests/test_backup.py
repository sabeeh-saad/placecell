from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from placecell import CollectionInfo, Memory, Pose
from placecell.backup import FILES, create_backup, digest, main, restore_backup, verify_backup
from placecell.command_identity import CommandJournal, CommandScope, IdentifiedCommand
from placecell.corrections import Correction, JsonlCorrectionLog
from placecell.depth import Box
from placecell.errors import ValidationError
from placecell.maintenance import STORAGE_KEYS, StorageLease
from placecell.mission_context import MissionContext
from placecell.navigation_ownership import NavigationOwnership, NavigationScope
from placecell.object_types import ObjectRecord, ObjectView
from placecell.pipeline import Observation
from placecell.providers import HashingEmbedder
from placecell.ros2.bridge import KeyframeWriter
from placecell.store.state import StateStore
from placecell.tracing import TraceStore


def deployment(root, *, lance=False):
    root.mkdir()
    db_path = root / "db"
    db_path.mkdir()
    embed = HashingEmbedder(16)
    info = CollectionInfo("office", embed.model_name, 16)
    if lance:
        from placecell.store.lancedb_store import LanceDBStore

        store = LanceDBStore(db_path, info)
    else:
        store = StateStore(info, db_path / "office.state.sqlite3")
        (db_path / "office.collection.json").write_text(json.dumps(asdict(info)))
    writer = KeyframeWriter(root / "keyframes")
    image = writer.write_jpeg("front", 100, b"offline evidence fixture")
    writer.confirm(image)
    memory = Memory.create("robot", "front", 100, Pose(1, 2, map_id="office-v1"), image, "printer")
    memory = memory.with_embedding(embed.embed_text([memory.caption])[0], embed.model_name)
    store.upsert([memory])
    view = replace(memory, id="view", evidence=replace(image, uri="file://" + image.uri))
    record = ObjectRecord("printer", "robot", "front", "map", "office-v1", "printer", 100, 100)
    store.objects.save(record, ObjectView("printer", view, Box(0, 0, 1, 1), b"crop bytes"))
    queued = writer.write_jpeg("front", 101, b"pending job evidence")
    store.jobs.enqueue(Observation("robot", "front", 101, memory.pose, queued), 4)
    failed = writer.write_jpeg("front", 102, b"failed job evidence")
    store.jobs.enqueue(Observation("robot", "front", 102, memory.pose, failed), 4)
    store.jobs.fail([store.jobs.pending(4)[1].id], "offline failure", max_attempts=1)
    orphan = writer.write_jpeg("front", 103, b"cleanup evidence")
    writer.confirm(orphan)
    store.enqueue_cleanup([orphan, replace(orphan, uri=str(root / "keyframes" / "already-removed.jpg"))])
    store.close()
    settings = {"collection": "office", "db_path": str(db_path), "keyframe_dir": str(root / "keyframes")}
    settings.update({key: str(root / name) for key, name in FILES.items()})
    corrections = JsonlCorrectionLog(settings["corrections_path"])
    corrections.record(Correction(memory.id, "wrong", note="test correction", timestamp=100))
    scope = CommandScope("robot", "office-v1", "original")
    journal = CommandJournal(settings["command_journal_path"], scope, clock=lambda: 1000)
    command = IdentifiedCommand("old-command", scope, 1000, "instruction", "visit printer")
    receipt = journal.claim(command)
    journal.close()
    context = MissionContext(settings["mission_context_path"], clock=lambda: 1000)
    context.record(receipt.request_id, "instruction", {"text": "visit printer"})
    context.close()
    owner = NavigationOwnership(
        settings["navigation_ownership_path"], NavigationScope("robot", "office-v1", "/navigate_to_pose")
    )
    owner.attest_clean("isolated test fixture")
    owner.reserve(receipt.request_id)
    owner.close()
    traces = TraceStore(settings["mission_trace_path"])
    assert traces.close()
    return settings, info, memory


@pytest.fixture
def stored(tmp_path):
    return deployment(tmp_path / "source")


def backup(stored, tmp_path):
    settings, _, _ = stored
    target = tmp_path / "backup"
    create_backup(settings, target, confirm_stopped=True)
    return target


def test_fresh_restore_preserves_state_and_blocks_old_navigation_and_commands(stored, tmp_path):
    settings, info, original = stored
    source = backup(stored, tmp_path)
    manifest_hash = digest(source / "manifest.json")
    target = tmp_path / "fresh"
    report = restore_backup(source, target)
    assert report["recovery_point_unix_s"] == verify_backup(source)["recovery_point_unix_s"]
    assert digest(source / "manifest.json") == manifest_hash
    store = StateStore(info, target / "db" / "office.state.sqlite3")
    restored = store.get(original.id)
    assert restored.same_embeddings(original) and restored.sightings == original.sightings
    assert Path(restored.evidence.uri).parent == target / "keyframes"
    assert Path(restored.evidence.uri).read_bytes() == Path(original.evidence.uri).read_bytes()
    assert store.objects.count() == 1 and store.jobs.stats()["queued"] == 2
    assert store.jobs.stats()["failed"] == 1 and store.jobs.failed()[0]["attempts"] == 1
    assert Path(store.jobs.pending(4)[0].observation.evidence.uri).parent == target / "keyframes"
    with sqlite3.connect(target / "db" / "office.state.sqlite3") as db:
        payload = json.loads(db.execute("SELECT payload FROM object_views").fetchone()[0])
        assert payload["evidence_uri"].startswith("file://" + str(target / "keyframes"))
    deleted = []
    assert store.drain_cleanup(deleted.append) == 2
    assert all(Path(item.uri).parent == target / "keyframes" for item in deleted)
    store.close()
    owner = NavigationOwnership(
        target / "navigation.sqlite3", NavigationScope("robot", "office-v1", "/navigate_to_pose")
    )
    assert owner.snapshot()["state"] == "unknown"
    with pytest.raises(ValidationError):
        owner.reserve("new trip")
    owner.close()
    with pytest.raises(ValidationError, match="new mission_conversation_id"):
        CommandJournal(target / "commands.sqlite3", CommandScope("robot", "office-v1", "original"))
    scope = CommandScope("robot", "office-v1", report["mission_conversation_id"])
    journal = CommandJournal(target / "commands.sqlite3", scope, clock=lambda: 1001)
    old = IdentifiedCommand(
        "old-command", CommandScope("robot", "office-v1", "original"), 1000, "instruction", "visit printer"
    )
    assert journal.claim(old).disposition == "wrong_scope"
    journal.close()
    assert JsonlCorrectionLog(target / "corrections.jsonl").verdicts([original.id])[original.id].wrong == 1
    assert json.loads((target / "storage-profile.json").read_text())["db_path"] == str(target / "db")
    assert all(Path(settings[k]).exists() for k in STORAGE_KEYS)


def test_lance_restore_rebuilds_relocated_projection_and_retrieves(tmp_path):
    from placecell.store.lancedb_store import LanceDBStore

    stored = deployment(tmp_path / "source", lance=True)
    source = backup(stored, tmp_path)
    restore_backup(source, tmp_path / "fresh")
    store = LanceDBStore.open(tmp_path / "fresh" / "db", "office")
    hit = store.search(stored[2].embedding, 1)[0]
    assert hit.memory.id == stored[2].id and hit.score > 0.99
    assert str(tmp_path / "fresh" / "keyframes") in hit.memory.evidence.uri
    assert store.objects.count() == 1
    store.close()


@pytest.mark.parametrize(
    "component", ["manifest.json", "corrections.jsonl", "db/office.state.sqlite3", "image", "missing", "extra"]
)
def test_damaged_backup_is_refused_before_destination_creation(stored, tmp_path, component):
    source = backup(stored, tmp_path)
    if component == "image":
        next((source / "keyframes").glob("*.jpg")).write_bytes(b"tampered")
    elif component == "missing":
        (source / "commands.sqlite3").unlink()
    elif component == "extra":
        (source / "extra").write_text("unexpected")
    else:
        (source / component).write_bytes(b"damaged")
    with pytest.raises((ValidationError, OSError)):
        restore_backup(source, tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "missing_image",
        "digest",
        "database",
        "future",
        "external_image",
        "symbolic_image",
        "missing_commands",
        "payload",
    ],
)
def test_bad_source_cannot_be_published(stored, tmp_path, fault):
    settings, _, memory = stored
    state = Path(settings["db_path"]) / "office.state.sqlite3"
    if fault == "missing_image":
        Path(memory.evidence.uri).unlink()
    elif fault == "digest":
        Path(memory.evidence.uri).write_bytes(b"wrong image")
    elif fault == "database":
        state.write_bytes(b"not sqlite")
    elif fault == "future":
        with sqlite3.connect(state) as connection:
            connection.execute("PRAGMA user_version=999")
    elif fault == "external_image":
        with sqlite3.connect(state) as connection:
            data = json.loads(connection.execute("SELECT payload FROM memories").fetchone()[0])
            data["evidence_uri"] = "/outside/secret.jpg"
            connection.execute("UPDATE memories SET evidence_uri=?,payload=?", (data["evidence_uri"], json.dumps(data)))
    elif fault == "symbolic_image":
        (Path(settings["keyframe_dir"]) / "link").symlink_to(Path(memory.evidence.uri))
    elif fault == "missing_commands":
        Path(settings["command_journal_path"]).unlink()
    else:
        with sqlite3.connect(state) as connection:
            connection.execute("UPDATE memories SET payload='not json'")
    with pytest.raises((ValidationError, sqlite3.Error, ValueError)):
        create_backup(settings, tmp_path / "backup", confirm_stopped=True)
    assert not (tmp_path / "backup").exists()
    assert not list(tmp_path.glob(".backup.incomplete-*"))


def test_lease_excludes_backup_and_releases_partial_acquisition(stored, tmp_path):
    settings, _, _ = stored
    with StorageLease.for_parameters(settings), pytest.raises(ValidationError, match="in use"):
        create_backup(settings, tmp_path / "backup", confirm_stopped=True)
    backup(stored, tmp_path)
    with StorageLease.for_parameters(settings):
        pass


def test_owned_navigation_alone_excludes_backup(stored, tmp_path):
    settings, _, _ = stored
    owner = NavigationOwnership(
        settings["navigation_ownership_path"], NavigationScope("robot", "office-v1", "/navigate_to_pose")
    )
    try:
        with pytest.raises(ValidationError, match="in use"):
            create_backup(settings, tmp_path / "backup", confirm_stopped=True)
    finally:
        owner.close()


@pytest.mark.parametrize("fault", ["no_attestation", "missing_key", "memory", "nested", "bad_name", "inside_source"])
def test_invalid_profiles_and_unconfirmed_writers(stored, tmp_path, fault):
    settings = dict(stored[0])
    target = tmp_path / "backup"
    if fault == "missing_key":
        settings.pop("mission_trace_path")
    elif fault == "memory":
        settings["db_path"] = ":memory:"
    elif fault == "nested":
        settings["keyframe_dir"] = settings["db_path"] + "/images"
    elif fault == "bad_name":
        settings["collection"] = "../bad"
    elif fault == "inside_source":
        target = Path(settings["keyframe_dir"]) / "backup"
    with pytest.raises(ValidationError):
        create_backup(settings, target, confirm_stopped=fault != "no_attestation")


def test_optional_components_can_be_explicitly_disabled(tmp_path):
    settings, info, _ = deployment(tmp_path / "source")
    store = StateStore(info, Path(settings["db_path"]) / "office.state.sqlite3")
    store.close()
    for key in FILES:
        if key != "corrections_path":
            settings[key] = ""
    target = tmp_path / "backup"
    create_backup(settings, target, confirm_stopped=True)
    restore_backup(target, tmp_path / "restored")


def test_refuses_overwrite_and_directory_publication_race(stored, tmp_path, monkeypatch):
    import placecell.backup as module

    source = backup(stored, tmp_path)
    target = tmp_path / "restored"
    target.mkdir()
    (target / "keep").write_text("do not overwrite")
    with pytest.raises(ValidationError):
        restore_backup(source, target)
    assert (target / "keep").read_text() == "do not overwrite"
    publish = module._publish

    def race(stage, destination):
        destination.mkdir()
        publish(stage, destination)

    monkeypatch.setattr(module, "_publish", race)
    with pytest.raises(FileExistsError):
        restore_backup(source, tmp_path / "race")
    assert list((tmp_path / "race").iterdir()) == []


@pytest.mark.parametrize("operation", ["create", "restore"])
def test_write_failure_never_publishes_partial_result(stored, tmp_path, monkeypatch, operation):
    import placecell.backup as module

    source = backup(stored, tmp_path)

    def fail(*args):
        raise OSError("simulated disk full")

    monkeypatch.setattr(module, "_sync_tree", fail)
    with pytest.raises(OSError, match="disk full"):
        if operation == "create":
            create_backup(stored[0], tmp_path / "failed", confirm_stopped=True)
        else:
            restore_backup(source, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert verify_backup(source)["counts"]["memories"] == 1


def test_cli_reports_and_errors(stored, tmp_path, capsys):
    settings = tmp_path / "profile.json"
    settings.write_text(json.dumps(stored[0]))
    source = tmp_path / "backup"
    main(["create", "--profile", str(settings), "--destination", str(source), "--confirm-stopped"])
    assert json.loads(capsys.readouterr().out)["counts"]["memories"] == 1
    main(["verify", "--backup", str(source)])
    assert json.loads(capsys.readouterr().out)["format"] == 1
    main(["restore", "--backup", str(source), "--destination", str(tmp_path / "fresh")])
    assert json.loads(capsys.readouterr().out)["navigation"].startswith("unknown")
    with pytest.raises(SystemExit) as error:
        main(["verify", "--backup", str(tmp_path / "absent")])
    assert error.value.code == 1 and "refused" in capsys.readouterr().err


def test_schema_upgrade_retries_after_interrupted_ddl_and_refuses_future(stored, monkeypatch):
    import placecell.store.state as module

    settings, info, memory = stored
    state = Path(settings["db_path"]) / "office.state.sqlite3"
    with sqlite3.connect(state) as connection:
        connection.execute("PRAGMA user_version=0")
        connection.execute("ALTER TABLE memories DROP COLUMN history_before")
    original = module.ObjectJournal

    def fail(*args):
        raise OSError("interrupted migration")

    monkeypatch.setattr(module, "ObjectJournal", fail)
    with pytest.raises(OSError, match="interrupted"):
        StateStore(info, state)
    with sqlite3.connect(state) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert "history_before" in {row[1] for row in connection.execute("PRAGMA table_info(memories)")}
    monkeypatch.setattr(module, "ObjectJournal", original)
    reopened = StateStore(info, state)
    assert reopened.get(memory.id).same_embeddings(memory)
    reopened.close()
    with sqlite3.connect(state) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        connection.execute("PRAGMA user_version=2")
    before = digest(state)
    with pytest.raises(ValidationError, match="Unsupported state schema"):
        StateStore(info, state)
    assert digest(state) == before


def test_backup_captures_committed_wal_and_excludes_uncommitted_write(stored, tmp_path):
    settings, info, memory = stored
    store = StateStore(info, Path(settings["db_path"]) / "office.state.sqlite3")
    store.upsert([replace(memory, caption="committed in WAL")])
    store._conn.execute("BEGIN IMMEDIATE")
    store._conn.execute("UPDATE memories SET payload='uncommitted damage'")
    try:
        source = backup(stored, tmp_path)
    finally:
        store._conn.execute("ROLLBACK")
        store.close()
    restore_backup(source, tmp_path / "fresh")
    reopened = StateStore(info, tmp_path / "fresh" / "db" / "office.state.sqlite3")
    assert reopened.get(memory.id).caption == "committed in WAL"
    reopened.close()


def test_trace_writer_excludes_backup(stored, tmp_path):
    trace = TraceStore(stored[0]["mission_trace_path"])
    try:
        with pytest.raises(ValidationError, match="in use"):
            backup(stored, tmp_path)
    finally:
        assert trace.close()


def test_future_lance_state_is_refused_before_projection_changes(tmp_path):
    from placecell.store.lancedb_store import LanceDBStore

    settings, info, _ = deployment(tmp_path / "source", lance=True)
    state = Path(settings["db_path"]) / "office.state.sqlite3"
    with sqlite3.connect(state) as connection:
        connection.execute("PRAGMA user_version=42")
    connection.close()

    # SQLite read-only admission may create empty WAL/shared-memory coordination
    # files. Authoritative bytes and all Lance projection files must stay intact.
    def persisted():
        return {
            str(p): digest(p)
            for p in Path(settings["db_path"]).rglob("*")
            if p.is_file() and not p.name.endswith((".sqlite3-wal", ".sqlite3-shm"))
        }

    before = persisted()
    with pytest.raises(ValidationError, match="Unsupported state schema"):
        LanceDBStore(settings["db_path"], info)
    assert before == persisted()
    wal = state.with_name(state.name + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0


def test_modified_source_during_restore_cannot_be_published(stored, tmp_path, monkeypatch):
    import placecell.backup as module

    source = backup(stored, tmp_path)
    copy = module.shutil.copyfile

    def mutate(origin, target):
        result = copy(origin, target)
        if Path(origin).name == "commands.sqlite3":
            Path(target).write_bytes(b"changed while copying")
        return result

    monkeypatch.setattr(module.shutil, "copyfile", mutate)
    with pytest.raises(ValidationError, match="checksum"):
        restore_backup(source, tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()


def test_unknown_backup_format_and_symlink_are_refused(stored, tmp_path):
    source = backup(stored, tmp_path)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["format"] = 2
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "manifest.sha256").write_text(digest(source / "manifest.json"))
    with pytest.raises(ValidationError, match="Unsupported backup format"):
        verify_backup(source)
    (source / "link").symlink_to(tmp_path)
    with pytest.raises(ValidationError, match="Links"):
        verify_backup(source)


def test_destination_alias_cannot_recurse_into_source_or_modify_backup(stored, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(stored[0]["keyframe_dir"], target_is_directory=True)
    with pytest.raises(ValidationError, match="inside source"):
        create_backup(stored[0], alias / "snapshot", confirm_stopped=True)
    source = backup(stored, tmp_path)
    with pytest.raises(ValidationError, match="inside the backup"):
        restore_backup(source, source / "fresh")
    assert verify_backup(source)["counts"]["memories"] == 1
