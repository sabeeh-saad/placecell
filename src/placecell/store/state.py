"""Transactional current memories, indexed observation history, and cleanup intents.

SQLite owns the state. Vector backends are derived indexes and can be rebuilt from it.
Returned memories carry at most 64 recent sightings; the complete retained history is
available through the paged sightings API. Temporal filters always use that history.
"""

# SQL fragments below contain only fixed column names and operators; all values are bound parameters.
# ruff: noqa: S608

from __future__ import annotations

import heapq
import json
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from placecell.errors import ModelMismatchError, ValidationError
from placecell.memory import Evidence, EvidenceKind, Memory, Sighting
from placecell.store.base import CollectionInfo, Filter, Hit
from placecell.store.codec import from_row, to_row
from placecell.store.jobs import WorkJournal

HISTORY_PREVIEW = 64


class StateStore:
    def __init__(self, info: CollectionInfo, path: str | Path = ":memory:") -> None:
        self._info = info
        self._lock = threading.RLock()
        self._depth = 0
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, robot_id TEXT, camera_id TEXT, timestamp REAL, last_seen REAL,
                x REAL, y REAL, frame_id TEXT, map_id TEXT, role TEXT, superseded INTEGER,
                consolidated_into TEXT, evidence_uri TEXT, payload TEXT NOT NULL, vector BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS memory_time ON memories(timestamp, id);
            CREATE INDEX IF NOT EXISTS memory_recent ON memories(last_seen DESC, id);
            CREATE INDEX IF NOT EXISTS memory_place ON memories(frame_id, map_id, x, y);
            CREATE INDEX IF NOT EXISTS memory_robot ON memories(robot_id, camera_id);
            CREATE INDEX IF NOT EXISTS memory_role ON memories(role, consolidated_into, id);
            CREATE INDEX IF NOT EXISTS memory_evidence ON memories(evidence_uri);
            CREATE TABLE IF NOT EXISTS sightings (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                observation_id TEXT NOT NULL, timestamp REAL NOT NULL,
                PRIMARY KEY(memory_id, timestamp, observation_id)
            );
            CREATE INDEX IF NOT EXISTS sighting_identity ON sightings(observation_id, memory_id);
            CREATE INDEX IF NOT EXISTS sighting_time ON sightings(timestamp, memory_id);
            CREATE TABLE IF NOT EXISTS dirty_vectors (id TEXT PRIMARY KEY, generation INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS cleanup (uri TEXT PRIMARY KEY, payload TEXT NOT NULL);
        """)

        self.jobs = WorkJournal(self._conn, self.transaction)

    @property
    def info(self) -> CollectionInfo:
        return self._info

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            depth = self._depth
            self._conn.execute("BEGIN IMMEDIATE" if depth == 0 else f"SAVEPOINT nested_{depth}")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK" if depth == 0 else f"ROLLBACK TO nested_{depth}")
                if depth:
                    self._conn.execute(f"RELEASE nested_{depth}")
                raise
            else:
                self._conn.execute("COMMIT" if depth == 0 else f"RELEASE nested_{depth}")
            finally:
                self._depth -= 1

    def _check(self, memory: Memory) -> None:
        if memory.embedding is None:
            raise ValidationError(f"memory {memory.id} has no embedding")
        if memory.model != self.info.model:
            raise ModelMismatchError(f"collection is bound to {self.info.model!r}, not {memory.model!r}")
        if memory.embedding.shape != (self.info.dimension,):
            raise ValidationError(f"memory {memory.id} must have dimension {self.info.dimension}")

    def upsert(self, memories: Iterable[Memory]) -> int:
        batch = list(memories)
        for memory in batch:
            self._check(memory)
        with self.transaction():
            for memory in batch:
                previous = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory.id,)).fetchone()
                if previous and previous["consolidated_into"]:
                    old = json.loads(previous["payload"])
                    if any(old[k] != getattr(memory, k) for k in ("caption", "last_seen", "superseded")) or (
                        memory.embedding is not None and previous["vector"] != memory.embedding.tobytes()
                    ):
                        self._invalidate_summary(
                            previous["consolidated_into"], memory.superseded_at or memory.last_seen
                        )
                        memory = replace(memory, consolidated_into="")
                if memory.consolidated_into:
                    parent = self._conn.execute(
                        "SELECT superseded FROM memories WHERE id=?", (memory.consolidated_into,)
                    ).fetchone()
                    if parent is not None and parent[0]:
                        memory = replace(memory, consolidated_into="")
                self._save(memory, previous)
        return len(batch)

    def _save(self, memory: Memory, previous: sqlite3.Row | None) -> None:
        row = to_row(memory)
        row.pop("vector")
        row.pop("sighting_ids")
        row.pop("sighting_times")
        assert memory.embedding is not None
        vector = memory.embedding.tobytes()
        columns = (
            "id",
            "robot_id",
            "camera_id",
            "timestamp",
            "last_seen",
            "x",
            "y",
            "frame_id",
            "map_id",
            "role",
            "superseded",
            "consolidated_into",
            "evidence_uri",
        )
        values = [row[c] for c in columns]
        values[-1] = str(values[-1]).removeprefix("file://")
        self._conn.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            + ",".join(f"{c}=excluded.{c}" for c in (*columns[1:], "payload", "vector")),
            [*values, json.dumps(row, separators=(",", ":")), vector],
        )
        self.append_sightings(memory.id, memory.sightings)
        projection = ("robot_id", "camera_id", "x", "y", "frame_id", "map_id", "role", "superseded")
        if previous is None or previous["vector"] != vector or any(previous[c] != row[c] for c in projection):
            self._conn.execute(
                "INSERT INTO dirty_vectors VALUES (?,1) ON CONFLICT(id) DO UPDATE SET generation=generation+1",
                (memory.id,),
            )
        if previous:
            old = json.loads(previous["payload"])
            if old["evidence_managed"] and old["evidence_uri"] != row["evidence_uri"]:
                self.enqueue_cleanup(
                    [
                        Evidence(
                            EvidenceKind(old["evidence_kind"]),
                            old["evidence_uri"],
                            old["evidence_digest"],
                            old["evidence_duration"],
                            managed=True,
                        )
                    ]
                )

    def _invalidate_summary(self, summary_id: str, now: float) -> None:
        summary = self._conn.execute("SELECT * FROM memories WHERE id=?", (summary_id,)).fetchone()
        if summary and not summary["superseded"]:
            memory = self._read(summary)
            self._save(replace(memory, superseded=True, superseded_at=now), summary)
        self._conn.execute(
            "UPDATE memories SET consolidated_into='', payload=json_set(payload, '$.consolidated_into', '') "
            "WHERE consolidated_into=?",
            (summary_id,),
        )

    def _read(self, row: sqlite3.Row) -> Memory:
        payload = json.loads(row["payload"])
        payload["vector"] = np.frombuffer(row["vector"], dtype=np.float32)
        sightings = self._conn.execute(
            "SELECT observation_id,timestamp FROM sightings WHERE memory_id=? "
            "ORDER BY timestamp DESC, observation_id DESC LIMIT ?",
            (row["id"], HISTORY_PREVIEW),
        ).fetchall()
        payload["sighting_ids"] = [s[0] for s in reversed(sightings)]
        payload["sighting_times"] = [s[1] for s in reversed(sightings)]
        return from_row(payload)

    def get(self, memory_id: str) -> Memory | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            return self._read(row) if row else None

    def append_sightings(self, memory_id: str, sightings: Iterable[Sighting]) -> None:
        with self.transaction():
            self._conn.executemany(
                "INSERT OR IGNORE INTO sightings VALUES (?,?,?)", ((memory_id, s.id, s.timestamp) for s in sightings)
            )

    def sightings(
        self,
        memory_id: str,
        *,
        limit: int = HISTORY_PREVIEW,
        after: tuple[float, str] | None = None,
        time_from: float | None = None,
        time_to: float | None = None,
    ) -> tuple[Sighting, ...]:
        if limit < 1:
            raise ValidationError("history page limit must be positive")
        terms, args = ["memory_id=?"], [memory_id]
        params: list[Any] = args
        if time_from is not None:
            terms.append("timestamp>=?")
            params.append(time_from)
        if time_to is not None:
            terms.append("timestamp<?")
            params.append(time_to)
        if after is not None:
            terms.append("(timestamp, observation_id)>(?,?)")
            params.extend(after)
        with self._lock:
            rows = self._conn.execute(
                "SELECT observation_id,timestamp FROM sightings WHERE "
                + " AND ".join(terms)
                + " ORDER BY timestamp, observation_id LIMIT ?",
                [*params, limit],
            ).fetchall()
            return tuple(Sighting(r[0], r[1]) for r in rows)

    def prune_history(self, before: float, *, where: Filter | None = None, limit: int = 4096) -> int:
        if limit < 1:
            raise ValidationError("history pruning limit must be positive")
        expr, params = self._predicate(where or Filter(include_superseded=True))
        with self.transaction():
            return self._conn.execute(
                "DELETE FROM sightings WHERE rowid IN (SELECT s.rowid FROM sightings s "
                "JOIN memories m ON m.id=s.memory_id WHERE s.timestamp<? AND s.timestamp<m.last_seen AND "
                + expr
                + " ORDER BY s.timestamp LIMIT ?)",
                [before, *params, limit],
            ).rowcount

    def _predicate(self, where: Filter) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if not where.include_superseded:
            clauses.append("m.superseded=0")
        for field in ("robot_id", "camera_id", "role", "frame_id", "map_id"):
            value = getattr(where, field)
            if value is not None:
                clauses.append(f"m.{field}=?")
                params.append(value)
        if where.unconsolidated:
            clauses.append("m.consolidated_into=''")
        if where.evidence_uri is not None:
            clauses.append("m.evidence_uri=?")
            params.append(where.evidence_uri.removeprefix("file://"))
        if where.near is not None and where.radius is not None:
            p, r = where.near, where.radius
            clauses.extend(
                [
                    "m.frame_id=?",
                    "m.map_id=?",
                    "m.x BETWEEN ? AND ?",
                    "m.y BETWEEN ? AND ?",
                    "(m.x-?)*(m.x-?)+(m.y-?)*(m.y-?)<=?",
                ]
            )
            params.extend([p.frame_id, p.map_id, p.x - r, p.x + r, p.y - r, p.y + r, p.x, p.x, p.y, p.y, r * r])
        events, event_params = self._event_predicate(where)
        if events:
            clauses.append("m.id IN (SELECT s.memory_id FROM sightings s WHERE " + events + ")")
            params.extend(event_params)
        return " AND ".join(clauses) or "1", params

    @staticmethod
    def _event_predicate(where: Filter) -> tuple[str, list[Any]]:
        clauses, params = [], []
        for field, op, value in (
            ("observation_id", "=", where.observation_id),
            ("timestamp", ">=", where.time_from),
            ("timestamp", "<", where.time_to),
        ):
            if value is not None:
                clauses.append(f"s.{field}{op}?")
                params.append(value)
        return " AND ".join(clauses), params

    def query(
        self, where: Filter | None = None, limit: int | None = None, *, order: Literal["oldest", "recent"] = "oldest"
    ) -> list[Memory]:
        if limit is not None and limit < 0:
            raise ValidationError("limit must not be negative")
        scope = where or Filter()
        expr, params = self._predicate(scope)
        sort_params: list[Any] = []
        sort = "m.last_seen DESC" if order == "recent" else "m.timestamp"
        if order == "oldest" and (scope.time_from is not None or scope.time_to is not None):
            events, sort_params = self._event_predicate(scope)
            sort = "(SELECT MIN(s.timestamp) FROM sightings s WHERE s.memory_id=m.id AND " + events + ")"
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.* FROM memories m WHERE " + expr + " ORDER BY " + sort + ",m.id LIMIT ?",
                [*params, *sort_params, -1 if limit is None else limit],
            ).fetchall()
            return [self._read(row) for row in rows]

    def iter_query(self, where: Filter | None = None, batch_size: int = 256) -> Iterator[list[Memory]]:
        if batch_size < 1:
            raise ValidationError("batch_size must be positive")
        expr, params = self._predicate(where or Filter())
        after = ""
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT m.* FROM memories m WHERE " + expr + " AND m.id>? ORDER BY m.id LIMIT ?",
                    [*params, after, batch_size],
                ).fetchall()
                batch = [self._read(row) for row in rows]
            if not batch:
                return
            after = batch[-1].id
            yield batch

    def search(self, vector: ArrayLike, k: int, where: Filter | None = None) -> list[Hit]:
        query = np.asarray(vector, dtype=np.float32)
        if k < 1 or query.shape != (self.info.dimension,) or not np.all(np.isfinite(query)):
            raise ValidationError("invalid search size or query vector")
        norm = float(np.linalg.norm(query))
        if not norm:
            return []
        query = query / norm
        expr, params = self._predicate(where or Filter())
        best: list[tuple[float, str]] = []
        with self._lock:
            cursor = self._conn.execute("SELECT m.id,m.vector FROM memories m WHERE " + expr + " ORDER BY m.id", params)
            while rows := cursor.fetchmany(256):
                matrix = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
                norms = np.linalg.norm(matrix, axis=1)
                scores = matrix @ query / np.where(norms, norms, 1)
                for row, score in zip(rows, scores, strict=True):
                    heapq.heappush(best, (float(score), row[0]))
                    if len(best) > k:
                        heapq.heappop(best)
            return [
                Hit(memory, score)
                for score, identity in sorted(best, key=lambda v: (-v[0], v[1]))
                if (memory := self.get(identity)) is not None
            ]

    def count(self, where: Filter | None = None) -> int:
        expr, params = self._predicate(where or Filter())
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM memories m WHERE " + expr, params).fetchone()[0])

    def delete(self, ids: Iterable[str]) -> int:
        removed = 0
        with self.transaction():
            for identity in ids:
                memory = self.get(identity)
                if memory is None:
                    continue
                if memory.consolidated_into:
                    self._invalidate_summary(memory.consolidated_into, memory.last_seen)
                if memory.role == "summary":
                    self._invalidate_summary(memory.id, memory.last_seen)
                if memory.evidence is not None and memory.evidence.managed:
                    self.enqueue_cleanup([memory.evidence])
                removed += self._conn.execute("DELETE FROM memories WHERE id=?", (identity,)).rowcount
                self._conn.execute(
                    "INSERT INTO dirty_vectors VALUES (?,1) ON CONFLICT(id) DO UPDATE SET generation=generation+1",
                    (identity,),
                )
        return removed

    def delete_where(self, where: Filter) -> int:
        return sum(self.delete(m.id for m in batch) for batch in self.iter_query(where))

    def enqueue_cleanup(self, evidence: Iterable[Evidence]) -> None:
        with self.transaction():
            for e in evidence:
                uri = e.uri.removeprefix("file://")
                self._conn.execute(
                    "INSERT OR IGNORE INTO cleanup VALUES (?,?)",
                    (
                        uri,
                        json.dumps(
                            {
                                "kind": e.kind.value,
                                "uri": e.uri,
                                "digest": e.digest,
                                "duration_s": e.duration_s,
                                "managed": e.managed,
                            }
                        ),
                    ),
                )

    def drain_cleanup(self, remover: Callable[[Evidence], None], limit: int = 256) -> int:
        removed = 0
        with self._lock:
            # File deletion cannot roll back. Wait until the enclosing memory transaction commits.
            if self._depth:
                return 0
        with self.transaction():
            rows = self._conn.execute(
                "SELECT * FROM cleanup c WHERE NOT EXISTS (SELECT 1 FROM jobs j WHERE j.uri=c.uri) LIMIT ?", (limit,)
            ).fetchall()
            for row in rows:
                referenced = self._conn.execute(
                    "SELECT 1 FROM memories WHERE evidence_uri=? LIMIT 1", (row["uri"],)
                ).fetchone()
                if not referenced:
                    data = json.loads(row["payload"])
                    data["kind"] = EvidenceKind(data["kind"])
                    remover(Evidence(**data))
                    removed += 1
                self._conn.execute("DELETE FROM cleanup WHERE uri=?", (row["uri"],))
        return removed

    def close(self) -> None:
        with self._lock:
            self._conn.close()
