"""Transactional current memories, indexed observation history, and cleanup intents.

SQLite owns the state. Vector backends are derived indexes and can be rebuilt from it.
Returned memories carry at most 64 recent sightings; the complete retained history is
available through the paged sightings API. Temporal filters always use that history.

One connection writes. A file-backed store serves reads outside a transaction from pooled
read-only connections, which see committed state and neither wait for the writer nor block
it; a thread inside `transaction()` reads through the writer and sees its own changes.
"""

# SQL fragments below contain only fixed column names and operators; all values are bound parameters.
# ruff: noqa: S608

from __future__ import annotations

import itertools
import json
import math
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from placecell.errors import ModelMismatchError, ValidationError
from placecell.memory import Evidence, EvidenceKind, Memory, SearchChannel, Sighting, Vector
from placecell.store.base import CollectionInfo, Filter, Hit
from placecell.store.codec import from_row, to_row
from placecell.store.jobs import WorkJournal
from placecell.store.limits import StoreLimits
from placecell.store.objects import ObjectJournal
from placecell.store.refinements import RefinementJournal
from placecell.store.schema import STATE_VERSION, check_connection

HISTORY_PREVIEW = 64
EVICTION_CANDIDATES = 32
"""How many of a robot's least recently seen memories are ranked for eviction."""
EVICTION_HALF_LIFE_S = 7 * 24 * 3600.0
"""Confidence half-life for that ranking; the retention and retrieval default."""

# Row counts per robot, kept in the writing transaction so every path and rollback stays exact.
_MEMORY_COUNTS = (
    "CREATE TABLE memory_counts (robot_id TEXT PRIMARY KEY, memories INTEGER NOT NULL)",
    "INSERT INTO memory_counts SELECT IFNULL(robot_id,''),COUNT(*) FROM memories GROUP BY 1",
    "CREATE TRIGGER memory_count_insert AFTER INSERT ON memories BEGIN "
    "INSERT INTO memory_counts VALUES (IFNULL(new.robot_id,''),1) "
    "ON CONFLICT(robot_id) DO UPDATE SET memories=memories+1; END",
    "CREATE TRIGGER memory_count_delete AFTER DELETE ON memories BEGIN "
    "UPDATE memory_counts SET memories=memories-1 WHERE robot_id=IFNULL(old.robot_id,''); END",
    "CREATE TRIGGER memory_count_move AFTER UPDATE OF robot_id ON memories "
    "WHEN IFNULL(old.robot_id,'')<>IFNULL(new.robot_id,'') BEGIN "
    "UPDATE memory_counts SET memories=memories-1 WHERE robot_id=IFNULL(old.robot_id,''); "
    "INSERT INTO memory_counts VALUES (IFNULL(new.robot_id,''),1) "
    "ON CONFLICT(robot_id) DO UPDATE SET memories=memories+1; END",
)


class StateStore:
    def __init__(
        self, info: CollectionInfo, path: str | Path = ":memory:", *, limits: StoreLimits | None = None
    ) -> None:
        self._info = info
        self.limits = limits or StoreLimits()
        self._lock = threading.RLock()
        self._depth = 0
        self._owner: int | None = None
        # An in-memory database exists only on its own connection, so it has no reader pool.
        self._database = None if str(path) == ":memory:" else str(path)
        self._readers: list[sqlite3.Connection] = []
        self._idle: list[sqlite3.Connection] = []
        self._pool_lock = threading.Lock()
        self._local = threading.local()
        self._closed = False
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        try:
            self._initialize()
        except BaseException:
            self._conn.close()
            raise

    def _initialize(self) -> None:
        check_connection(self._conn)
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
            CREATE INDEX IF NOT EXISTS memory_robot_age ON memories(robot_id, last_seen, id);
            CREATE INDEX IF NOT EXISTS memory_robot_superseded ON memories(robot_id, last_seen, id) WHERE superseded=1;
            CREATE INDEX IF NOT EXISTS memory_robot_folded ON memories(robot_id, last_seen, id)
                WHERE consolidated_into<>'';
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

        columns = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)")}
        with self.transaction():
            if "history_before" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN history_before REAL NOT NULL DEFAULT 0")
            if "caption_vector" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN caption_vector BLOB")
            if "embedding_kind" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN embedding_kind TEXT NOT NULL DEFAULT 'legacy'")
            if not self._conn.execute("SELECT 1 FROM sqlite_master WHERE name='memory_counts'").fetchone():
                for statement in _MEMORY_COUNTS:
                    self._conn.execute(statement)

        self.jobs = WorkJournal(self._conn, self.transaction, self.enqueue_cleanup)
        self.refinements = RefinementJournal(self._conn, self.transaction, self.limits.max_refinement_jobs)
        self.objects = ObjectJournal(
            self._conn, self.transaction, self.info.model, self.info.dimension, self.enqueue_cleanup
        )
        self._conn.execute(f"PRAGMA user_version={STATE_VERSION}")

    @property
    def info(self) -> CollectionInfo:
        return self._info

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            depth = self._depth
            self._conn.execute("BEGIN IMMEDIATE" if depth == 0 else f"SAVEPOINT nested_{depth}")
            self._depth += 1
            self._owner = threading.get_ident()
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
                if not self._depth:
                    self._owner = None

    def _in_transaction(self) -> bool:
        """Whether the calling thread is inside this store's transaction."""
        return self._owner == threading.get_ident()

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection]:
        """A connection for one consistent read: the writer inside a transaction, else a pooled reader."""
        if self._database is None or self._in_transaction():
            with self._lock:
                yield self._conn
            return
        current: sqlite3.Connection | None = getattr(self._local, "reader", None)
        if current is not None:
            yield current
            return
        conn = self._checkout()
        self._local.reader = conn
        try:
            conn.execute("BEGIN")
            try:
                yield conn
            finally:
                conn.execute("COMMIT")
        finally:
            self._local.reader = None
            with self._pool_lock:
                self._idle.append(conn)

    def _checkout(self) -> sqlite3.Connection:
        with self._pool_lock:
            if self._closed:
                raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
            if self._idle:
                return self._idle.pop()
            assert self._database is not None
            conn = sqlite3.connect(self._database, isolation_level=None, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            self._readers.append(conn)
            return conn

    def _check(self, memory: Memory) -> None:
        if memory.embedding is None:
            raise ValidationError(f"memory {memory.id} has no embedding")
        if memory.model != self.info.model:
            raise ModelMismatchError(f"collection is bound to {self.info.model!r}, not {memory.model!r}")
        if memory.embedding.shape != (self.info.dimension,):
            raise ValidationError(f"memory {memory.id} must have dimension {self.info.dimension}")

    def upsert(self, memories: Iterable[Memory]) -> int:
        return self._write(memories, admit=True)

    def _write(self, memories: Iterable[Memory], *, admit: bool) -> int:
        """Upsert a batch. Without admission, existing records are imported regardless of capacity."""
        batch = list(memories)
        for memory in batch:
            self._check(memory)
        written = {memory.id for memory in batch}
        with self.transaction():
            for memory in batch:
                previous = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory.id,)).fetchone()
                if previous is None and admit:
                    self._admit(memory.robot_id, written)
                if previous and previous["consolidated_into"]:
                    old = json.loads(previous["payload"])
                    if any(old[k] != getattr(memory, k) for k in ("caption", "last_seen", "superseded")) or (
                        not self._read(previous, self._conn).same_embeddings(memory)
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

    def _admit(self, robot_id: str, protected: set[str]) -> None:
        """Make room for one new memory, evicting one unless eviction is disabled.

        Each robot holding memories is entitled to an equal share of the capacity. A robot at
        or above its share gives up its own memory; below it, the largest holder gives one up.
        Memories written by the current batch are never evicted.
        """
        counts: dict[str, int] = dict(
            self._conn.execute("SELECT robot_id,memories FROM memory_counts WHERE memories>0").fetchall()
        )
        if sum(counts.values()) < self.limits.max_memories:
            return
        if not self.limits.evict_at_capacity:
            raise ValidationError("memory capacity reached; prune retained memories or raise max_memories")
        owner = robot_id
        if counts.get(robot_id, 0) * len(counts.keys() | {robot_id}) < self.limits.max_memories:
            owner = max(counts, key=lambda r: (counts[r], r))
        victim, folded = self._victim(owner, protected)
        if victim is None:
            raise ValidationError("memory capacity reached and nothing can be evicted; raise max_memories")
        self._remove(victim, keep_summary=folded)

    def _victim(self, robot_id: str, protected: set[str]) -> tuple[Memory | None, bool]:
        """The robot's least valuable memory and whether it is covered by a summary.

        Superseded memories go first, then memories folded into a summary, then the least
        recently seen memories ranked by decayed confidence.
        """
        for condition, folded in (("superseded=1", False), ("consolidated_into<>''", True)):
            oldest = self._oldest(robot_id, condition, protected, 1)
            if oldest:
                return self.get(oldest[0][0]), folded
        candidates = self._oldest(robot_id, "1", protected, EVICTION_CANDIDATES)
        if not candidates:
            return None, False
        newest = candidates[-1][1]
        weakest = min(candidates, key=lambda r: (r[2] * 0.5 ** ((newest - r[1]) / EVICTION_HALF_LIFE_S), r[1], r[0]))
        return self.get(weakest[0]), False

    def _oldest(self, robot_id: str, condition: str, protected: set[str], limit: int) -> list[sqlite3.Row]:
        with closing(
            self._conn.execute(
                "SELECT id,last_seen,json_extract(payload,'$.confidence') FROM memories "
                "WHERE robot_id=? AND " + condition + " ORDER BY last_seen,id",
                (robot_id,),
            )
        ) as cursor:
            return list(itertools.islice((row for row in cursor if row[0] not in protected), limit))

    def _save(self, memory: Memory, previous: sqlite3.Row | None) -> None:
        row = to_row(memory)
        row.pop("vector")
        row.pop("caption_vector")
        row.pop("sighting_ids")
        row.pop("sighting_times")
        assert memory.embedding is not None
        vector = memory.embedding.tobytes()
        caption = memory.vector_for("caption")
        caption_vector = caption.tobytes() if caption is not None else None
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
            "INSERT INTO memories ("
            + ",".join((*columns, "payload", "vector", "caption_vector", "embedding_kind"))
            + ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            + ",".join(
                f"{c}=excluded.{c}" for c in (*columns[1:], "payload", "vector", "caption_vector", "embedding_kind")
            ),
            [*values, json.dumps(row, separators=(",", ":")), vector, caption_vector, memory.embedding_kind],
        )
        self.append_sightings(memory.id, memory.sightings)
        projection = ("robot_id", "camera_id", "x", "y", "frame_id", "map_id", "role", "superseded")
        if (
            previous is None
            or previous["vector"] != vector
            or previous["caption_vector"] != caption_vector
            or previous["embedding_kind"] != memory.embedding_kind
            or any(previous[c] != row[c] for c in projection)
        ):
            self._conn.execute(
                "INSERT INTO dirty_vectors VALUES (?,1) ON CONFLICT(id) DO UPDATE SET generation=generation+1",
                (memory.id,),
            )
        if previous:
            old = json.loads(previous["payload"])
            evidence_fields = ("evidence_kind", "evidence_uri", "evidence_digest", "evidence_duration")
            same_digest = bool(row["evidence_digest"]) and all(
                old[k] == row[k] for k in ("evidence_kind", "evidence_digest", "evidence_duration")
            )
            if not same_digest and any(old[k] != row[k] for k in evidence_fields):
                self.refinements.request(memory.id, "new evidence")
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
        elif memory.evidence is not None and not memory.caption:
            self.refinements.request(memory.id, "missing caption")
        if memory.superseded or memory.role != "episodic" or memory.evidence is None:
            self._conn.execute("DELETE FROM refinement_jobs WHERE memory_id=?", (memory.id,))

    def _invalidate_summary(self, summary_id: str, now: float) -> None:
        summary = self._conn.execute("SELECT * FROM memories WHERE id=?", (summary_id,)).fetchone()
        if summary and not summary["superseded"]:
            memory = self._read(summary, self._conn)
            self._save(replace(memory, superseded=True, superseded_at=now), summary)
        self._conn.execute(
            "UPDATE memories SET consolidated_into='', payload=json_set(payload, '$.consolidated_into', '') "
            "WHERE consolidated_into=?",
            (summary_id,),
        )

    def _read(self, row: sqlite3.Row, conn: sqlite3.Connection) -> Memory:
        payload = json.loads(row["payload"])
        payload["vector"] = np.frombuffer(row["vector"], dtype=np.float32)
        payload["caption_vector"] = (
            np.frombuffer(row["caption_vector"], dtype=np.float32) if row["caption_vector"] is not None else None
        )
        payload["embedding_kind"] = row["embedding_kind"]
        sightings = conn.execute(
            "SELECT observation_id,timestamp FROM sightings WHERE memory_id=? "
            "ORDER BY timestamp DESC, observation_id DESC LIMIT ?",
            (row["id"], HISTORY_PREVIEW),
        ).fetchall()
        payload["sighting_ids"] = [s[0] for s in reversed(sightings)]
        payload["sighting_times"] = [s[1] for s in reversed(sightings)]
        return from_row(payload)

    def get(self, memory_id: str) -> Memory | None:
        with self._reading() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            return self._read(row, conn) if row else None

    def append_sightings(self, memory_id: str, sightings: Iterable[Sighting]) -> None:
        with self.transaction():
            row = self._conn.execute(
                "SELECT history_before,last_seen FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row is None:
                raise ValidationError("cannot append sightings to a missing memory")
            self._conn.executemany(
                "INSERT OR IGNORE INTO sightings VALUES (?,?,?)",
                ((memory_id, s.id, s.timestamp) for s in sightings if s.timestamp >= row[0] or s.timestamp == row[1]),
            )
            self._conn.execute(
                "DELETE FROM sightings WHERE memory_id=? AND rowid NOT IN "
                "(SELECT rowid FROM sightings WHERE memory_id=? ORDER BY timestamp DESC,observation_id DESC LIMIT ?)",
                (memory_id, memory_id, self.limits.max_sightings),
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
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT observation_id,timestamp FROM sightings WHERE "
                + " AND ".join(terms)
                + " ORDER BY timestamp, observation_id LIMIT ?",
                [*params, limit],
            ).fetchall()
            return tuple(Sighting(r[0], r[1]) for r in rows)

    def prune_history(self, before: float, *, where: Filter | None = None, limit: int = 4096) -> int:
        if type(limit) is not int or limit < 1 or not math.isfinite(before):
            raise ValidationError("history pruning needs a positive limit and finite cutoff")
        expr, params = self._predicate(where or Filter(include_superseded=True))
        with self.transaction():
            self._conn.execute(
                "UPDATE memories SET history_before=MAX(history_before,?) WHERE id IN "
                "(SELECT m.id FROM memories m WHERE " + expr + ")",
                [before, *params],
            )
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
        with self._reading() as conn:
            rows = conn.execute(
                "SELECT m.* FROM memories m WHERE " + expr + " ORDER BY " + sort + ",m.id LIMIT ?",
                [*params, *sort_params, -1 if limit is None else limit],
            ).fetchall()
            return [self._read(row, conn) for row in rows]

    def iter_query(self, where: Filter | None = None, batch_size: int = 256) -> Iterator[list[Memory]]:
        if batch_size < 1:
            raise ValidationError("batch_size must be positive")
        expr, params = self._predicate(where or Filter())
        after = ""
        while True:
            with self._reading() as conn:
                rows = conn.execute(
                    "SELECT m.* FROM memories m WHERE " + expr + " AND m.id>? ORDER BY m.id LIMIT ?",
                    [*params, after, batch_size],
                ).fetchall()
                batch = [self._read(row, conn) for row in rows]
            if not batch:
                return
            after = batch[-1].id
            yield batch

    def search(
        self, vector: ArrayLike, k: int, where: Filter | None = None, *, channel: SearchChannel = "primary"
    ) -> list[Hit]:
        if channel not in {"primary", "image", "caption"}:
            raise ValidationError("search channel must be primary, image or caption")
        query = np.asarray(vector, dtype=np.float32)
        if k < 1 or query.shape != (self.info.dimension,) or not np.all(np.isfinite(query)):
            raise ValidationError("invalid search size or query vector")
        norm = float(np.linalg.norm(query))
        if not norm:
            return []
        query = query / norm
        expr, params, column = self._vector_predicate(where or Filter(), channel)
        with self._reading() as conn:
            cursor = conn.execute(f"SELECT m.id,m.{column} FROM memories m WHERE " + expr, params)
            best = _best(iter(lambda: cursor.fetchmany(1024), []), query, k)
            return [Hit(memory, score) for identity, score in best if (memory := self.get(identity)) is not None]

    def _vector_predicate(self, where: Filter, channel: SearchChannel) -> tuple[str, list[Any], str]:
        """The filter restricted to rows that carry the channel's vector, and that vector's column."""
        expr, params = self._predicate(where)
        column = "caption_vector" if channel == "caption" else "vector"
        expr += f" AND m.{column} IS NOT NULL"
        if channel == "image":
            expr += " AND m.embedding_kind='image'"
        return expr, params, column

    def _best_changed(
        self, conn: sqlite3.Connection, query: Vector, k: int, where: Filter, channel: SearchChannel
    ) -> list[tuple[str, float]]:
        """The k best (id, cosine) pairs among rows changed since the last projection sync that the filter keeps."""
        expr, params, column = self._vector_predicate(where, channel)
        # CROSS JOIN starts from the few changed rows instead of the filtered memories.
        cursor = conn.execute(
            f"SELECT m.id,m.{column} FROM dirty_vectors d CROSS JOIN memories m ON m.id=d.id WHERE " + expr, params
        )
        return _best(iter(lambda: cursor.fetchmany(1024), []), query, k)

    def _count_upto(self, where: Filter, channel: SearchChannel, limit: int) -> int:
        """How many rows a search would score, counting no further than `limit`."""
        expr, params, _ = self._vector_predicate(where, channel)
        with self._reading() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM (SELECT 1 FROM memories m WHERE " + expr + " LIMIT ?)", [*params, limit]
                ).fetchone()[0]
            )

    def count(self, where: Filter | None = None) -> int:
        expr, params = self._predicate(where or Filter())
        with self._reading() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories m WHERE " + expr, params).fetchone()[0])

    def delete(self, ids: Iterable[str]) -> int:
        removed = 0
        with self.transaction():
            for identity in ids:
                memory = self.get(identity)
                if memory is not None:
                    removed += self._remove(memory)
        return removed

    def _remove(self, memory: Memory, *, keep_summary: bool = False) -> int:
        """Delete one memory, queue its managed evidence and release its summary.

        Eviction keeps a member's summary, which already represents it.
        """
        if memory.consolidated_into and not keep_summary:
            self._invalidate_summary(memory.consolidated_into, memory.last_seen)
        if memory.role == "summary":
            self._invalidate_summary(memory.id, memory.last_seen)
        if memory.evidence is not None and memory.evidence.managed:
            self.enqueue_cleanup([memory.evidence])
        removed = int(self._conn.execute("DELETE FROM memories WHERE id=?", (memory.id,)).rowcount)
        self._conn.execute(
            "INSERT INTO dirty_vectors VALUES (?,1) ON CONFLICT(id) DO UPDATE SET generation=generation+1",
            (memory.id,),
        )
        return removed

    def delete_where(self, where: Filter) -> int:
        return sum(self.delete(m.id for m in batch) for batch in self.iter_query(where))

    def enqueue_cleanup(self, evidence: Iterable[Evidence]) -> None:
        with self.transaction():
            for e in evidence:
                uri = e.uri.removeprefix("file://")
                if (
                    not self._conn.execute("SELECT 1 FROM cleanup WHERE uri=?", (uri,)).fetchone()
                    and self._conn.execute("SELECT COUNT(*) FROM cleanup").fetchone()[0] >= self.limits.max_cleanup
                ):
                    raise ValidationError(
                        "evidence cleanup capacity reached; restore cleanup before accepting more media"
                    )
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
                    "SELECT 1 FROM memories WHERE evidence_uri=? UNION ALL "
                    "SELECT 1 FROM object_views WHERE uri=? LIMIT 1",
                    (row["uri"], row["uri"]),
                ).fetchone()
                if not referenced:
                    data = json.loads(row["payload"])
                    data["kind"] = EvidenceKind(data["kind"])
                    remover(Evidence(**data))
                    removed += 1
                self._conn.execute("DELETE FROM cleanup WHERE uri=?", (row["uri"],))
        return removed

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            readers, self._readers, self._idle = self._readers, [], []
        for reader in readers:
            reader.close()
        with self._lock:
            self._conn.close()


def _best(pages: Iterable[list[sqlite3.Row]], query: Vector, k: int) -> list[tuple[str, float]]:
    """The k best (id, cosine) pairs of (id, vector blob) rows, best first.

    Keeps the k highest (score, id) pairs, so equal scores are resolved the same way
    whatever order the rows arrive in, and lists equal scores by id.
    """
    ids: list[str] = []
    scores: Vector = np.empty(0, dtype=np.float32)
    for rows in pages:
        matrix = np.frombuffer(b"".join(row[1] for row in rows), dtype=np.float32).reshape(len(rows), -1)
        norms = np.linalg.norm(matrix, axis=1)
        ids += [row[0] for row in rows]
        scores = np.concatenate([scores, matrix @ query / np.where(norms, norms, 1)])
        if len(ids) > k:
            kth = np.partition(scores, len(ids) - k)[len(ids) - k]
            keep = np.flatnonzero(scores > kth).tolist()
            tied = sorted(np.flatnonzero(scores == kth).tolist(), key=ids.__getitem__, reverse=True)
            keep += tied[: k - len(keep)]
            ids, scores = [ids[i] for i in keep], scores[keep]
    return sorted(zip(ids, scores.tolist(), strict=True), key=lambda v: (-v[1], v[0]))
