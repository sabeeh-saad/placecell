from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from dataclasses import asdict, replace

import pytest

from placecell.command_identity import CommandJournal, CommandScope, IdentifiedCommand
from placecell.errors import ValidationError
from placecell.operator import parse_operator_command

SCOPE = CommandScope("r1", "office-v1", "operator")


def envelope(**changes):
    return {
        "schema_version": 2,
        "command_id": "command-1",
        "scope": asdict(SCOPE),
        "issued_at_unix_s": 1000,
        "command": "instruction",
        "text": "Visit the printer",
        **changes,
    }


def command(**changes):
    parsed = parse_operator_command(json.dumps(envelope(**changes)))
    assert isinstance(parsed, IdentifiedCommand)
    return parsed


@pytest.mark.parametrize(
    "changes",
    [
        {"command_id": ""},
        {"command_id": "a" * 129},
        {"command_id": "é"},
        {"command_id": True},
        {"scope": {"robot_id": "r1"}},
        {"scope": []},
        {"scope": {**asdict(SCOPE), "map_id": " "}},
        {"scope": {**asdict(SCOPE), "extra": 1}},
        {"issued_at_unix_s": True},
        {"issued_at_unix_s": float("nan")},
        {"issued_at_unix_s": float("inf")},
        {"issued_at_unix_s": -1},
        {"issued_at_unix_s": 10**100},
        {"issued_at_unix_s": "1000"},
        {"text": "stop"},
        {"text": "cancel"},
        {"text": "option one"},
        {"target_request_id": "a" * 32},
    ],
)
def test_invalid_identity_never_reaches_admission(changes):
    with pytest.raises(ValidationError):
        command(**changes)


@pytest.mark.parametrize("kind", ["stop", "choose"])
def test_continuations_require_a_target(kind):
    value = envelope(command=kind)
    del value["text"]
    if kind == "choose":
        value["option"] = 2
    for target in (None, "", "wrong", True):
        with pytest.raises(ValidationError):
            parse_operator_command(json.dumps({**value, "target_request_id": target}))
    parsed = parse_operator_command(json.dumps({**value, "target_request_id": "b" * 32}))
    assert parsed.text == ("stop" if kind == "stop" else "option 2")


def test_duplicate_conflict_and_deliberate_repeat(tmp_path):
    journal = CommandJournal(tmp_path / "commands.db", SCOPE, clock=lambda: 1000)
    try:
        first = journal.claim(command())
        assert first.disposition == "recorded" and len(first.request_id) == 32
        assert journal.claim(command(issued_at_unix_s=1000.0)).request_id == first.request_id
        assert journal.claim(command()).disposition == "duplicate"
        assert journal.claim(command(text="Visit cupboard")).disposition == "conflict"
        assert journal.claim(command(issued_at_unix_s=1001)).disposition == "conflict"
        repeated = journal.claim(command(command_id="command-2"))
        assert repeated.disposition == "recorded" and repeated.request_id != first.request_id
    finally:
        journal.close()


def test_atomic_claim_across_connections(tmp_path):
    path = tmp_path / "commands.db"
    journals = [CommandJournal(path, SCOPE, clock=lambda: 1000) for _ in range(4)]
    barrier, results, errors = threading.Barrier(4), [], []

    def claim(journal):
        try:
            barrier.wait(timeout=5)
            results.append(journal.claim(command()))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=claim, args=(j,)) for j in journals]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert not errors and len(results) == 4
        assert [r.disposition for r in results].count("recorded") == 1
        assert [r.disposition for r in results].count("duplicate") == 3
        assert len({r.request_id for r in results}) == 1
    finally:
        for journal in journals:
            journal.close()


def test_scope_change_does_not_accept_an_old_scope_or_poison_new_one(tmp_path):
    path = tmp_path / "commands.db"
    old = CommandJournal(path, SCOPE, clock=lambda: 1000)
    original = old.claim(command())
    old.close()
    for field in ("robot_id", "map_id", "conversation_id"):
        new_scope = replace(SCOPE, **{field: "new"})
        new = CommandJournal(path, new_scope, clock=lambda: 1000)
        try:
            assert new.claim(command()).disposition == "wrong_scope"
            fresh = new.claim(command(scope=asdict(new_scope)))
            assert fresh.disposition == "recorded" and fresh.request_id != original.request_id
        finally:
            new.close()
    reopened = CommandJournal(path, SCOPE, clock=lambda: 1000)
    assert reopened.claim(command()).request_id == original.request_id
    reopened.close()


def test_retention_capacity_expiry_and_clock_rollback_across_restart(tmp_path):
    now = [1000]
    path = tmp_path / "commands.db"
    journal = CommandJournal(path, SCOPE, retry_window_s=10, max_records=1, clock=lambda: now[0])
    first = journal.claim(command())
    assert journal.claim(command(command_id="two")).disposition == "capacity"
    assert journal.claim(command()).request_id == first.request_id  # Never evict to admit new work.
    assert journal.claim(command(command_id="future", issued_at_unix_s=1006)).disposition == "future_timestamp"
    now[0] = 1010
    assert journal.claim(command()).disposition == "expired"
    assert journal.claim(command(command_id="two", issued_at_unix_s=1010)).disposition == "recorded"
    assert journal._db.execute("SELECT count(*) FROM command_claims").fetchone()[0] == 1
    journal.close()
    now[0] = 1000
    journal = CommandJournal(path, SCOPE, retry_window_s=10, max_records=1, clock=lambda: now[0])
    assert journal.claim(command()).disposition == "expired"
    journal.close()
    with pytest.raises(ValidationError, match="policy"):
        CommandJournal(path, SCOPE, retry_window_s=20)


def test_crash_after_reservation_never_replays(tmp_path):
    path = tmp_path / "commands.db"
    script = """
import json, os, sys
from placecell.command_identity import CommandJournal
from placecell.operator import parse_operator_command
command = parse_operator_command(sys.argv[2])
journal = CommandJournal(sys.argv[1], command.scope, clock=lambda: 1000)
receipt = journal.claim(command)
print(receipt.request_id, flush=True)
os._exit(17)
"""
    crashed = subprocess.run(  # noqa: S603 - fixed local crash fixture, no external input
        [sys.executable, "-c", script, str(path), json.dumps(envelope())], capture_output=True, text=True, timeout=10
    )
    assert crashed.returncode == 17, crashed.stderr
    journal = CommandJournal(path, SCOPE, clock=lambda: 1000)
    recovered = journal.claim(command())
    assert recovered.disposition == "duplicate" and recovered.request_id == crashed.stdout.strip()
    journal.close()


def test_journal_lock_failure_and_corruption_do_not_create_reservations(tmp_path):
    path = tmp_path / "commands.db"
    journal = CommandJournal(path, SCOPE, clock=lambda: 1000)
    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        journal.claim(command())
    blocker.rollback()
    blocker.close()
    assert journal.claim(command()).disposition == "recorded"
    journal.close()
    path.write_bytes(b"broken sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        CommandJournal(path, SCOPE)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retry_window_s": 0},
        {"retry_window_s": float("nan")},
        {"max_records": True},
        {"max_records": 0},
        {"max_records": 1000001},
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValidationError):
        CommandJournal(":memory:", SCOPE, **kwargs)


def test_invalid_clock_fails_closed():
    journal = CommandJournal(":memory:", SCOPE, clock=lambda: float("nan"))
    with pytest.raises(ValidationError, match="clock"):
        journal.claim(command())
    assert journal._db.execute("SELECT count(*) FROM command_claims").fetchone()[0] == 0
    journal.close()
