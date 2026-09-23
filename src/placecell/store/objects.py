"""SQLite object records, crop vectors and bounded change histories.

The original full-frame file is shared safely with scene memory. Crops live inside
SQLite, so creation, replacement and deletion have the same atomic boundary as identities.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict
from typing import Any

import numpy as np

from placecell.depth import Box, ObjectPosition
from placecell.errors import ValidationError
from placecell.memory import Vector
from placecell.object_types import ObjectEvent, ObjectRecord, ObjectView
from placecell.store.codec import from_row, to_row


class ObjectJournal:
    def __init__(
        self,
        connection: sqlite3.Connection,
        transaction: Callable[[], AbstractContextManager[None]],
        model: str,
        dimension: int,
    ) -> None:
        self._conn, self._transaction = connection, transaction
        self._model, self._dimension = model, dimension
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS objects (
                id TEXT PRIMARY KEY, robot_id TEXT, camera_id TEXT, frame_id TEXT, map_id TEXT,
                last_seen REAL, payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS object_scope ON objects(robot_id,camera_id,frame_id,map_id,last_seen);
            CREATE TABLE IF NOT EXISTS object_views (
                id TEXT PRIMARY KEY, object_id TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
                timestamp REAL, uri TEXT, payload TEXT, box TEXT, crop BLOB, vector BLOB, caption_vector BLOB
            );
            CREATE INDEX IF NOT EXISTS object_view_owner ON object_views(object_id,timestamp DESC);
            CREATE INDEX IF NOT EXISTS object_view_evidence ON object_views(uri);
            CREATE TABLE IF NOT EXISTS object_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE, payload TEXT
            );
            CREATE INDEX IF NOT EXISTS object_event_owner ON object_events(object_id,sequence DESC);
            INSERT OR IGNORE INTO settings VALUES ('objects_generation','0');
            INSERT OR IGNORE INTO settings VALUES ('objects_evidence_generation','0');
        """)

    @property
    def generation(self) -> int:
        with self._transaction():
            return int(self._conn.execute("SELECT value FROM settings WHERE key='objects_generation'").fetchone()[0])

    @property
    def evidence_generation(self) -> int:
        """Changes to identities or views, excluding scan scheduling metadata."""
        with self._transaction():
            return int(self._conn.execute(
                "SELECT value FROM settings WHERE key='objects_evidence_generation'"
            ).fetchone()[0])

    def _changed(self, *, evidence: bool = True) -> None:
        self._conn.execute("UPDATE settings SET value=CAST(value AS INTEGER)+1 WHERE key='objects_generation'")
        if evidence:
            self._conn.execute(
                "UPDATE settings SET value=CAST(value AS INTEGER)+1 WHERE key='objects_evidence_generation'"
            )

    def scan_time(self, scope: str) -> float | None:
        with self._transaction():
            row = self._conn.execute("SELECT value FROM settings WHERE key=?", ("object_scan:" + scope,)).fetchone()
            return float(row[0]) if row else None

    def clear(self) -> None:
        """Remove identities, their dependent evidence, and replay scan state."""
        with self._transaction():
            for record in self.iter_records():
                self.delete(record.id)
            self._conn.execute("DELETE FROM settings WHERE key LIKE 'object_scan:%'")
            self._changed()

    def record_scan(self, scope: str, timestamp: float) -> None:
        with self._transaction():
            self._conn.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", ("object_scan:" + scope, str(timestamp)))
            self._changed(evidence=False)

    def iter_records(self) -> Iterator[ObjectRecord]:
        after = ""
        while True:
            with self._transaction():
                rows = self._conn.execute(
                    "SELECT id,payload FROM objects WHERE id>? ORDER BY id LIMIT 128", (after,)
                ).fetchall()
            if not rows:
                return
            for row in rows:
                yield self._record(row["payload"])
            after = rows[-1]["id"]

    @staticmethod
    def _record(payload: str) -> ObjectRecord:
        data = json.loads(payload)
        if data["position"] is not None:
            data["position"] = ObjectPosition(**data["position"])
        return ObjectRecord(**data)

    def get(self, identity: str) -> ObjectRecord | None:
        with self._transaction():
            row = self._conn.execute("SELECT payload FROM objects WHERE id=?", (identity,)).fetchone()
            return self._record(row[0]) if row else None

    def records(self, *, robot_id: str, camera_id: str, frame_id: str, map_id: str) -> list[ObjectRecord]:
        with self._transaction():
            rows = self._conn.execute(
                "SELECT payload FROM objects WHERE robot_id=? AND camera_id=? AND frame_id=? AND map_id=? ORDER BY id",
                (robot_id, camera_id, frame_id, map_id),
            ).fetchall()
            return [self._record(row[0]) for row in rows]

    def count(self) -> int:
        with self._transaction():
            return int(self._conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0])

    def save(
        self,
        record: ObjectRecord,
        view: ObjectView | None = None,
        *,
        event: str = "",
        event_time: float | None = None,
        max_views: int = 4,
        max_events: int = 32,
    ) -> None:
        if max_views < 1 or max_events < 1:
            raise ValidationError("object history limits must be positive")
        if view is not None and (
            view.object_id != record.id
            or view.memory.model != self._model
            or view.memory.embedding is None
            or view.memory.embedding.shape != (self._dimension,)
            or len(view.crop_png) > 1_000_000
        ):
            raise ValidationError("invalid object view, crop size or embedding space")
        with self._transaction():
            self._conn.execute(
                "INSERT INTO objects VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "last_seen=excluded.last_seen,payload=excluded.payload",
                (
                    record.id,
                    record.robot_id,
                    record.camera_id,
                    record.frame_id,
                    record.map_id,
                    record.last_seen,
                    json.dumps(asdict(record)),
                ),
            )
            if view is not None:
                data = to_row(view.memory)
                data.pop("vector")
                data.pop("caption_vector")
                assert view.memory.embedding is not None
                caption = view.memory.caption_embedding
                self._conn.execute(
                    "INSERT OR REPLACE INTO object_views VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        view.memory.id,
                        record.id,
                        view.memory.timestamp,
                        view.memory.evidence.uri.removeprefix("file://") if view.memory.evidence else "",
                        json.dumps(data),
                        json.dumps(asdict(view.box)),
                        view.crop_png,
                        view.memory.embedding.astype("<f4").tobytes(),
                        caption.astype("<f4").tobytes() if caption is not None else None,
                    ),
                )
                stale = self._conn.execute(
                    "SELECT * FROM object_views WHERE object_id=? ORDER BY timestamp DESC,id DESC LIMIT -1 OFFSET ?",
                    (record.id, max_views),
                ).fetchall()
                self._release(stale)
                for row in stale:
                    self._conn.execute("DELETE FROM object_views WHERE id=?", (row["id"],))
            if event:
                self._conn.execute(
                    "INSERT INTO object_events(object_id,payload) VALUES (?,?)",
                    (
                        record.id,
                        json.dumps(
                            asdict(
                                ObjectEvent(
                                    record.last_seen if event_time is None else event_time, event, record.position
                                )
                            )
                        ),
                    ),
                )
                self._conn.execute(
                    "DELETE FROM object_events WHERE object_id=? AND sequence NOT IN "
                    "(SELECT sequence FROM object_events WHERE object_id=? ORDER BY sequence DESC LIMIT ?)",
                    (record.id, record.id, max_events),
                )
            self._changed()

    def _view(self, row: sqlite3.Row) -> ObjectView:
        data = json.loads(row["payload"])
        data["vector"] = np.frombuffer(row["vector"], dtype="<f4")
        data["caption_vector"] = np.frombuffer(row["caption_vector"], dtype="<f4") if row["caption_vector"] else None
        return ObjectView(row["object_id"], from_row(data), Box(**json.loads(row["box"])), row["crop"])

    def views(self, identity: str, *, include_crops: bool = True, limit: int = -1) -> list[ObjectView]:
        query = (
            "SELECT * FROM object_views WHERE object_id=? ORDER BY timestamp DESC,id DESC LIMIT ?"
            if include_crops
            else "SELECT id,object_id,payload,box,vector,caption_vector,X'' AS crop FROM object_views "
            "WHERE object_id=? ORDER BY timestamp DESC,id DESC LIMIT ?"
        )
        with self._transaction():
            return [self._view(row) for row in self._conn.execute(query, (identity, limit)).fetchall()]

    def history(self, identity: str) -> list[ObjectEvent]:
        with self._transaction():
            rows = self._conn.execute(
                "SELECT payload FROM object_events WHERE object_id=? ORDER BY sequence", (identity,)
            ).fetchall()
        result = []
        for row in rows:
            data = json.loads(row[0])
            if data["position"] is not None:
                data["position"] = ObjectPosition(**data["position"])
            result.append(ObjectEvent(**data))
        return result

    def scores(
        self, vector: Vector, *, robot_id: str, camera_id: str, frame_id: str, map_id: str
    ) -> Iterator[tuple[str, float]]:
        """Score bounded pages of vectors without reading crop BLOBs or full histories."""
        after = ""
        while True:
            with self._transaction():
                rows = self._conn.execute(
                    "SELECT v.id,v.object_id,v.vector,v.caption_vector FROM object_views v JOIN objects o "
                    "ON o.id=v.object_id WHERE o.robot_id=? AND o.camera_id=? AND o.frame_id=? AND o.map_id=? "
                    "AND v.id>? ORDER BY v.id LIMIT 128",
                    (robot_id, camera_id, frame_id, map_id, after),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                scores = []
                for column in ("vector", "caption_vector"):
                    if row[column]:
                        other = np.frombuffer(row[column], dtype="<f4")
                        norm = float(np.linalg.norm(other))
                        scores.append(float(vector @ other / norm) if norm else 0.0)
                yield row["object_id"], max(scores)
            after = rows[-1]["id"]

    def _release(self, rows: list[sqlite3.Row]) -> None:
        for row in rows:
            data: dict[str, Any] = json.loads(row["payload"])
            if data["evidence_managed"]:
                self._conn.execute(
                    "INSERT OR IGNORE INTO cleanup VALUES (?,?)",
                    (
                        row["uri"],
                        json.dumps(
                            {
                                "kind": data["evidence_kind"],
                                "uri": data["evidence_uri"],
                                "digest": data["evidence_digest"],
                                "duration_s": data["evidence_duration"],
                                "managed": True,
                            }
                        ),
                    ),
                )

    def delete(self, identity: str) -> bool:
        with self._transaction():
            self._release(self._conn.execute("SELECT * FROM object_views WHERE object_id=?", (identity,)).fetchall())
            removed = self._conn.execute("DELETE FROM objects WHERE id=?", (identity,)).rowcount > 0
            if removed:
                self._changed()
            return removed

    def prune(self, before: float, limit: int = 128) -> int:
        with self._transaction():
            rows = self._conn.execute(
                "SELECT id FROM objects WHERE last_seen<? ORDER BY last_seen LIMIT ?", (before, limit)
            ).fetchall()
            return sum(self.delete(row[0]) for row in rows)
