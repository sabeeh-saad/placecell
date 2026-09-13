"""Durable recheck requests and bounded caption revision history."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field

import numpy as np

from placecell.errors import ValidationError
from placecell.memory import Evidence, Memory, Vector


def evidence_key(evidence: Evidence | None) -> str:
    return json.dumps(asdict(evidence) if evidence else None, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RefinementJob:
    memory_id: str
    generation: str
    attempt: int
    reason: str


@dataclass(frozen=True)
class MemoryRevision:
    id: int
    memory_id: str
    timestamp: float
    reason: str
    producer: str
    evidence_key: str
    before_caption: str
    after_caption: str
    before_vector: Vector = field(repr=False, compare=False)
    after_vector: Vector = field(repr=False, compare=False)
    rolled_back: bool = False


class RefinementJournal:
    def __init__(self, connection: sqlite3.Connection, transaction: Callable[[], AbstractContextManager[None]]) -> None:
        self._conn, self._transaction = connection, transaction
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS refinement_jobs (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                generation TEXT NOT NULL, requested_at REAL NOT NULL, reason TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS refinement_ready ON refinement_jobs(retry_at, requested_at);
            CREATE TABLE IF NOT EXISTS memory_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                timestamp REAL NOT NULL, reason TEXT NOT NULL, producer TEXT NOT NULL,
                evidence_key TEXT NOT NULL, before_caption TEXT NOT NULL, after_caption TEXT NOT NULL,
                before_vector BLOB NOT NULL, after_vector BLOB NOT NULL,
                rolled_back INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS revision_memory ON memory_revisions(memory_id, id DESC);
        """)

    def request(self, memory_id: str, reason: str = "explicit recheck") -> bool:
        """Coalesce requests for a live episodic memory; a new request resets failed attempts."""
        if not reason.strip() or len(reason) > 200:
            raise ValidationError("refinement reason must contain 1 to 200 characters")
        with self._transaction():
            if (
                self._conn.execute(
                    "SELECT 1 FROM memories WHERE id=? AND role='episodic' AND superseded=0 AND evidence_uri<>''",
                    (memory_id,),
                ).fetchone()
                is None
            ):
                return False
            self._conn.execute(
                "INSERT INTO refinement_jobs(memory_id,generation,requested_at,reason) VALUES (?,?,?,?) "
                "ON CONFLICT(memory_id) DO UPDATE SET generation=excluded.generation,reason=excluded.reason,"
                "attempts=0,retry_at=0,error=''",
                (memory_id, uuid.uuid4().hex, time.time(), reason),
            )
            return True

    def claim(self, max_attempts: int, retry_delay_s: float, now: float) -> RefinementJob | None:
        """Reserve one attempt before provider work, so crashes cannot cause unlimited retries."""
        with self._transaction():
            row = self._conn.execute(
                "SELECT j.* FROM refinement_jobs j JOIN memories m ON m.id=j.memory_id "
                "WHERE j.attempts<? AND j.retry_at<=? AND m.role='episodic' AND m.superseded=0 "
                "AND m.evidence_uri<>'' ORDER BY j.requested_at,j.memory_id LIMIT 1",
                (max_attempts, now),
            ).fetchone()
            if row is None:
                return None
            attempt = int(row["attempts"]) + 1
            self._conn.execute(
                "UPDATE refinement_jobs SET attempts=?,retry_at=?,error='attempt interrupted or still running' "
                "WHERE memory_id=?",
                (attempt, now + retry_delay_s, row["memory_id"]),
            )
            return RefinementJob(row["memory_id"], row["generation"], attempt, row["reason"])

    def current(self, job: RefinementJob) -> bool:
        with self._transaction():
            return (
                self._conn.execute(
                    "SELECT 1 FROM refinement_jobs WHERE memory_id=? AND generation=? AND attempts=?",
                    (job.memory_id, job.generation, job.attempt),
                ).fetchone()
                is not None
            )

    def complete(self, job: RefinementJob) -> None:
        with self._transaction():
            self._conn.execute(
                "DELETE FROM refinement_jobs WHERE memory_id=? AND generation=? AND attempts=?",
                (job.memory_id, job.generation, job.attempt),
            )

    def fail(self, job: RefinementJob, error: str) -> None:
        with self._transaction():
            self._conn.execute(
                "UPDATE refinement_jobs SET error=? WHERE memory_id=? AND generation=? AND attempts=?",
                (error[:2000], job.memory_id, job.generation, job.attempt),
            )

    def cancel(self, memory_id: str) -> None:
        with self._transaction():
            self._conn.execute("DELETE FROM refinement_jobs WHERE memory_id=?", (memory_id,))

    def pending(self, limit: int = 100) -> list[dict[str, str | int | float]]:
        """Inspect queued and exhausted requests, including the last error."""
        if limit < 1:
            raise ValidationError("request limit must be positive")
        with self._transaction():
            return [
                dict(r)
                for r in self._conn.execute(
                    "SELECT * FROM refinement_jobs ORDER BY requested_at,memory_id LIMIT ?", (limit,)
                )
            ]

    def record(self, before: Memory, after: Memory, job: RefinementJob, producer: str, now: float, keep: int) -> None:
        assert before.embedding is not None and after.embedding is not None
        with self._transaction():
            self._conn.execute(
                "INSERT INTO memory_revisions(memory_id,timestamp,reason,producer,evidence_key,"
                "before_caption,after_caption,before_vector,after_vector) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    before.id,
                    now,
                    job.reason,
                    producer,
                    evidence_key(before.evidence),
                    before.caption,
                    after.caption,
                    before.embedding.tobytes(),
                    after.embedding.tobytes(),
                ),
            )
            self._conn.execute(
                "DELETE FROM memory_revisions WHERE memory_id=? AND id NOT IN "
                "(SELECT id FROM memory_revisions WHERE memory_id=? ORDER BY id DESC LIMIT ?)",
                (before.id, before.id, keep),
            )

    def history(self, memory_id: str, limit: int = 3) -> list[MemoryRevision]:
        if limit < 1:
            raise ValidationError("revision limit must be positive")
        with self._transaction():
            rows = self._conn.execute(
                "SELECT * FROM memory_revisions WHERE memory_id=? ORDER BY id DESC LIMIT ?", (memory_id, limit)
            ).fetchall()
            return [
                MemoryRevision(
                    r["id"],
                    r["memory_id"],
                    r["timestamp"],
                    r["reason"],
                    r["producer"],
                    r["evidence_key"],
                    r["before_caption"],
                    r["after_caption"],
                    np.frombuffer(r["before_vector"], dtype=np.float32),
                    np.frombuffer(r["after_vector"], dtype=np.float32),
                    bool(r["rolled_back"]),
                )
                for r in rows
            ]

    def mark_rolled_back(self, revision_id: int) -> None:
        with self._transaction():
            self._conn.execute("UPDATE memory_revisions SET rolled_back=1 WHERE id=?", (revision_id,))
