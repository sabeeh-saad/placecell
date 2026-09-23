"""Durable ingestion jobs sharing the memory store's transaction boundary."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from placecell.depth import DepthSnapshot
from placecell.errors import ValidationError
from placecell.memory import Evidence, EvidenceKind, Pose, memory_id

if TYPE_CHECKING:
    from placecell.pipeline import Observation


@dataclass(frozen=True)
class Job:
    id: str
    observation: Observation
    attempts: int


class WorkJournal:
    def __init__(
        self,
        connection: sqlite3.Connection,
        transaction: Callable[[], AbstractContextManager[None]],
        cleanup: Callable[[Iterable[Evidence]], None],
    ) -> None:
        self._conn = connection
        self._transaction = transaction
        self._cleanup = cleanup
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, timestamp REAL NOT NULL, uri TEXT NOT NULL, payload TEXT NOT NULL,
                enqueued_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                retry_at REAL NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS job_order ON jobs(failed, timestamp, id);
            CREATE INDEX IF NOT EXISTS job_evidence ON jobs(uri);
        """)

    def enqueue(self, observation: Observation, capacity: int) -> bool:
        if capacity < 1:
            raise ValidationError("queue capacity must be positive")
        identity = memory_id(observation.robot_id, observation.camera_id, observation.timestamp)
        with self._transaction():
            if self._conn.execute("SELECT 1 FROM jobs WHERE id=?", (identity,)).fetchone():
                return True
            if self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] >= capacity:
                return False
            self._conn.execute(
                "INSERT INTO jobs(id,timestamp,uri,payload,enqueued_at) VALUES (?,?,?,?,?)",
                (
                    identity,
                    observation.timestamp,
                    observation.evidence.uri.removeprefix("file://"),
                    json.dumps(asdict(observation)),
                    time.time(),
                ),
            )
            return True

    def pending(self, limit: int, now: float | None = None) -> list[Job]:
        from placecell.pipeline import Observation

        if limit < 1:
            raise ValidationError("batch size must be positive")
        now = time.time() if now is None else now
        with self._transaction():
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE failed=0 ORDER BY timestamp,id LIMIT ?", (limit,)
            ).fetchall()
        jobs = []
        for row in rows:
            # Keep later observations behind a pending retry so contradiction visits remain ordered.
            if row["retry_at"] > now:
                break
            data = json.loads(row["payload"])
            data["pose"] = Pose(**data["pose"])
            data["evidence"]["kind"] = EvidenceKind(data["evidence"]["kind"])
            data["evidence"] = Evidence(**data["evidence"])
            if data.get("depth") is not None:
                data["depth"]["map_from_camera"] = tuple(data["depth"]["map_from_camera"])
                data["depth"] = DepthSnapshot(**data["depth"])
            jobs.append(Job(row["id"], Observation(**data), row["attempts"]))
        return jobs

    def complete(self, ids: Iterable[str]) -> None:
        with self._transaction():
            for identity in ids:
                row = self._conn.execute("SELECT payload,uri FROM jobs WHERE id=?", (identity,)).fetchone()
                if row is None:
                    continue
                evidence = json.loads(row["payload"])["evidence"]
                if evidence["managed"]:
                    evidence["kind"] = EvidenceKind(evidence["kind"])
                    self._cleanup([Evidence(**evidence)])
                self._conn.execute("DELETE FROM jobs WHERE id=?", (identity,))

    def fail(self, ids: Iterable[str], error: str, *, max_attempts: int = 5, retry_delay_s: float = 1) -> None:
        if max_attempts < 1 or retry_delay_s < 0:
            raise ValidationError("invalid retry policy")
        with self._transaction():
            for identity in ids:
                row = self._conn.execute("SELECT attempts FROM jobs WHERE id=?", (identity,)).fetchone()
                if row is None:
                    continue
                attempts = int(row[0]) + 1
                retry_at = time.time() + min(300, retry_delay_s * 2 ** min(attempts - 1, 20))
                self._conn.execute(
                    "UPDATE jobs SET attempts=?,retry_at=?,failed=?,error=? WHERE id=?",
                    (attempts, retry_at, attempts >= max_attempts, error[:2000], identity),
                )

    def retry_failed(self) -> int:
        with self._transaction():
            return self._conn.execute("UPDATE jobs SET failed=0,attempts=0,retry_at=0,error='' WHERE failed=1").rowcount

    def failed(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._transaction():
            rows = self._conn.execute(
                "SELECT id,attempts,error,uri FROM jobs WHERE failed=1 ORDER BY timestamp,id LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def stats(self) -> dict[str, float | int]:
        with self._transaction():
            row = self._conn.execute("SELECT COUNT(*),COALESCE(SUM(failed),0),MIN(enqueued_at) FROM jobs").fetchone()
            return {
                "queued": int(row[0]),
                "failed": int(row[1]),
                "oldest_age_s": max(0, time.time() - row[2]) if row[2] is not None else 0.0,
            }
