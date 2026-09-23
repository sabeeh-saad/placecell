"""Persistent, bounded conversation events; never a queue of robot commands."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from placecell.errors import ValidationError

_WINDOW_NOTICE = "Earlier context is unavailable. Do not infer missing destinations or treat older events as recent."


class MissionContext:
    """Retain whole requests under global row/byte/age limits, with scoped prompt windows.

    Limits bound logical retained data, not SQLite's allocated file size. Old requests
    are removed first; a single oversized active request is refused without partial writes.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        scope: str = "default",
        max_events: int = 1000,
        max_bytes: int = 2_097_152,
        retention_s: float = 30 * 86400,
        clock: Callable[[], float] = time.time,
        references_available: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        if not isinstance(scope, str) or not scope.strip() or len(scope) > 512:
            raise ValidationError("mission context scope must contain 1..512 characters")
        if type(max_events) is not int or max_events < 1 or type(max_bytes) is not int or max_bytes < 1024:
            raise ValidationError("mission context needs a positive event limit and at least 1024 bytes")
        if not math.isfinite(retention_s) or retention_s <= 0:
            raise ValidationError("mission context retention must be finite and positive")
        if str(path) != ":memory:":
            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
        self._scope, self._lock = scope, threading.Lock()
        self._max_events, self._max_bytes, self._retention_s = max_events, max_bytes, retention_s
        self._clock = clock
        self._references_available = references_available
        self._db = sqlite3.connect(str(path), timeout=0.25, check_same_thread=False)
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS mission_events (
                id INTEGER PRIMARY KEY, scope TEXT NOT NULL, request_id TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL, timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS mission_scope ON mission_events(scope, id);
            CREATE INDEX IF NOT EXISTS mission_request ON mission_events(scope, request_id, id);
            CREATE TABLE IF NOT EXISTS context_retention (id INTEGER PRIMARY KEY CHECK(id=1), removed INTEGER NOT NULL);
            INSERT OR IGNORE INTO context_retention VALUES (1,0);
        """)
        try:
            self.prune()
        except Exception:
            self._db.close()
            raise

    def _now(self) -> float:
        now = self._clock()
        if not math.isfinite(now) or now < 0:
            raise ValidationError("mission context clock must be a finite non-negative time")
        return now

    def _totals(self) -> tuple[int, int]:
        row = self._db.execute(
            "SELECT COUNT(*),COALESCE(SUM(length(CAST(payload AS BLOB))+length(CAST(scope AS BLOB))"
            "+length(CAST(request_id AS BLOB))+length(kind)+32),0) FROM mission_events"
        ).fetchone()
        return int(row[0]), int(row[1])

    def _prune(self, now: float, protected: str = "") -> int:
        removed = 0
        # Whole requests prevent retained statuses from silently changing the meaning
        # of an instruction after its initial event has been deleted.
        while True:
            count, size = self._totals()
            over = count > self._max_events or size > self._max_bytes
            row = self._db.execute(
                "SELECT scope,request_id,MAX(timestamp) FROM mission_events "
                "WHERE NOT (scope=? AND request_id=?) GROUP BY scope,request_id "
                "HAVING MAX(timestamp)<? OR ? ORDER BY MIN(id) LIMIT 1",
                (self._scope, protected, now - self._retention_s, over),
            ).fetchone()
            if row is None:
                if over:
                    raise ValidationError("current mission exceeds context retention capacity")
                break
            removed += self._db.execute("DELETE FROM mission_events WHERE scope=? AND request_id=?", row[:2]).rowcount
        self._db.execute("UPDATE context_retention SET removed=removed+? WHERE id=1", (removed,))
        return removed

    def prune(self) -> int:
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            return self._prune(self._now())

    def record(self, request_id: str, kind: str, payload: dict[str, Any]) -> None:
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 256:
            raise ValidationError("mission request id must contain 1..256 characters")
        if kind not in {"instruction", "status"}:
            raise ValidationError("mission context kind must be instruction or status")
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded) > 32000:
            raise ValidationError("mission context event is too large")
        now = self._now()
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            self._db.execute(
                "INSERT INTO mission_events(scope, request_id, kind, payload, timestamp) VALUES (?, ?, ?, ?, ?)",
                (self._scope, request_id, kind, encoded, now),
            )
            self._prune(now, protected=request_id)

    def recent(self, *, exclude_request_id: str = "", limit: int = 20, max_chars: int = 16000) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValidationError("mission context window must be within 1..100 events")
        if type(max_chars) is not int or not 1000 <= max_chars <= 64000:
            raise ValidationError("mission context budget must be within 1000..64000 characters")
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            self._prune(self._now(), protected=exclude_request_id)
            rows = self._db.execute(
                "SELECT request_id, kind, payload, timestamp FROM mission_events "
                "WHERE scope = ? AND request_id != ? ORDER BY id DESC LIMIT ?",
                (self._scope, exclude_request_id, limit + 1),
            ).fetchall()
            truncated = bool(self._db.execute("SELECT removed FROM context_retention WHERE id=1").fetchone()[0])
        result: list[dict[str, Any]] = []
        # Reserve space for an explicit boundary rather than hide pruning from the agents.
        used = len(json.dumps({"history_boundary": _WINDOW_NOTICE})) + 4
        truncated |= len(rows) > limit
        for request_id, kind, payload, timestamp in rows[:limit]:
            data = json.loads(payload)
            if self._references_available is not None and not self._references_available(data):
                # Remove the affected request and all older context. Never fall back
                # from a deleted latest destination to a still-retained older visit.
                result = [event for event in result if event["request_id"] != request_id]
                truncated = True
                break
            event = {"request_id": request_id, "kind": kind, "data": data, "timestamp": timestamp}
            size = len(json.dumps(event)) + 2
            if used + size > max_chars:
                truncated = True
                break  # Never skip a newer oversized event in favor of an older destination.
            result.append(event)
            used += size
        result.reverse()
        if truncated:
            if result:
                result[0]["history_boundary"] = _WINDOW_NOTICE
            else:
                result.append({"kind": "retention_boundary", "history_boundary": _WINDOW_NOTICE})
        return result

    def stats(self) -> dict[str, int]:
        with self._lock:
            count, size = self._totals()
            removed = self._db.execute("SELECT removed FROM context_retention WHERE id=1").fetchone()[0]
            return {"events": count, "bytes": size, "pruned_events": int(removed)}

    def close(self) -> None:
        with self._lock:
            self._db.close()
