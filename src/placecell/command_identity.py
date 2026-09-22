"""Bounded, durable admission of identified commands; never a queue to replay."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from placecell.errors import ValidationError


@dataclass(frozen=True, slots=True)
class CommandScope:
    robot_id: str
    map_id: str
    conversation_id: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() or len(v) > 256 for v in asdict(self).values()):
            raise ValidationError("Command scope values must contain 1..256 characters and not be blank.")

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class IdentifiedCommand:
    command_id: str
    scope: CommandScope
    issued_at_unix_s: float
    command: str
    text: str
    target_request_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.command_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.command_id):
            raise ValidationError("command_id must contain 1..128 ASCII letters, digits, underscores or hyphens.")
        if type(self.issued_at_unix_s) not in (int, float) or not 0 < self.issued_at_unix_s < 1e12:
            raise ValidationError("issued_at_unix_s must be a finite positive Unix timestamp below 1e12.")
        if self.command not in {"instruction", "stop", "choose"} or not self.text.strip() or len(self.text) > 2000:
            raise ValidationError("Invalid identified command.")
        if self.command in {"stop", "choose"} and (
            not isinstance(self.target_request_id, str) or not re.fullmatch(r"[a-f0-9]{32}", self.target_request_id)
        ):
            raise ValidationError("stop/choose require the snapshot's 32-character target_request_id.")
        if self.command == "instruction" and self.target_request_id:
            raise ValidationError("An instruction cannot have a target_request_id.")

    def fingerprint(self) -> str:
        data = asdict(self)
        data["issued_at_unix_s"] = float(self.issued_at_unix_s)
        return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    disposition: str
    request_id: str = ""


class CommandJournal:
    """Commit before routing, including refusals. A reservation never proves execution.

    The database is a single-controller resource. Atomic claims also prevent duplicate
    routing by concurrent connections, but do not coordinate different active missions.
    """

    def __init__(
        self,
        path: str | Path,
        scope: CommandScope,
        *,
        retry_window_s: float = 86400.0,
        max_records: int = 10000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not math.isfinite(retry_window_s) or not 1 <= retry_window_s <= 604800:
            raise ValidationError("command_retry_window_s must be within 1..604800.")
        if type(max_records) is not int or not 1 <= max_records <= 1000000:
            raise ValidationError("command_max_records must be an integer within 1..1000000.")
        self.scope, self.retry_window_s, self.max_records = scope, retry_window_s, max_records
        self.durable = str(path) != ":memory:"
        self._clock, self._lock = clock, threading.Lock()
        if self.durable:
            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), timeout=0.05, check_same_thread=False)
        try:
            self._db.execute("PRAGMA synchronous=FULL")
            if self._db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValidationError("Command journal integrity check failed.")
            with self._db:
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS command_policy "
                    "(id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL, "
                    "retry_window_s REAL NOT NULL, high_water REAL NOT NULL)"
                )
                self._db.execute("INSERT OR IGNORE INTO command_policy VALUES (1, 1, ?, 0)", (retry_window_s,))
                policy = self._db.execute("SELECT version, retry_window_s FROM command_policy WHERE id=1").fetchone()
                if policy != (1, retry_window_s):
                    raise ValidationError(
                        "Command journal version/retry window mismatch; preserve its existing policy."
                    )
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS command_claims "
                    "(scope TEXT NOT NULL, command_id TEXT NOT NULL, fingerprint TEXT NOT NULL, "
                    "request_id TEXT NOT NULL, expires_at REAL NOT NULL, PRIMARY KEY(scope, command_id))"
                )
                self._db.execute("CREATE INDEX IF NOT EXISTS command_expiry ON command_claims(expires_at)")
        except BaseException:
            self._db.close()
            raise

    def claim(self, command: IdentifiedCommand) -> CommandReceipt:
        if command.scope != self.scope:
            return CommandReceipt("wrong_scope")
        fingerprint = command.fingerprint()
        with self._lock, self._db:
            # Serialize lookup + reservation across connections, before any controller work.
            self._db.execute("BEGIN IMMEDIATE")
            wall_now = self._clock()
            if not math.isfinite(wall_now) or wall_now <= 0:
                raise ValidationError("Command admission requires a valid UTC clock.")
            high_water = self._db.execute("SELECT high_water FROM command_policy WHERE id=1").fetchone()[0]
            now = max(wall_now, high_water)
            self._db.execute("UPDATE command_policy SET high_water=? WHERE id=1", (now,))
            # Expiry survives restart and wall-clock rollback; never evict live records.
            self._db.execute("DELETE FROM command_claims WHERE expires_at<=?", (now,))
            expires = command.issued_at_unix_s + self.retry_window_s
            if expires <= now:
                return CommandReceipt("expired")
            if command.issued_at_unix_s > wall_now + 5:
                return CommandReceipt("future_timestamp")
            row = self._db.execute(
                "SELECT fingerprint, request_id FROM command_claims WHERE scope=? AND command_id=?",
                (self.scope.key(), command.command_id),
            ).fetchone()
            if row:
                return CommandReceipt("duplicate" if row[0] == fingerprint else "conflict", row[1])
            if self._db.execute("SELECT count(*) FROM command_claims").fetchone()[0] >= self.max_records:
                return CommandReceipt("capacity")
            request_id = uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO command_claims VALUES (?, ?, ?, ?, ?)",
                (self.scope.key(), command.command_id, fingerprint, request_id, expires),
            )
            return CommandReceipt("recorded", request_id)

    def close(self) -> None:
        with self._lock:
            self._db.close()
