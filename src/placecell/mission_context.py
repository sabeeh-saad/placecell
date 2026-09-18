"""Persistent, scoped conversation and mission events; never a queue of robot commands."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from placecell.errors import ValidationError


class MissionContext:
    """SQLite history with a bounded prompt window, isolated by robot/map/conversation scope."""

    def __init__(self, path: str | Path = ":memory:", *, scope: str = "default") -> None:
        if not scope.strip():
            raise ValidationError("mission context scope must not be empty")
        if str(path) != ":memory:":
            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
        self._scope, self._lock = scope, threading.Lock()
        self._db = sqlite3.connect(str(path), timeout=0.25, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS mission_events ("
            "id INTEGER PRIMARY KEY, scope TEXT NOT NULL, request_id TEXT NOT NULL, "
            "kind TEXT NOT NULL, payload TEXT NOT NULL, timestamp REAL NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS mission_scope ON mission_events(scope, id)")
        self._db.commit()

    def record(self, request_id: str, kind: str, payload: dict[str, Any]) -> None:
        if kind not in {"instruction", "status"}:
            raise ValidationError("mission context kind must be instruction or status")
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded) > 32000:
            raise ValidationError("mission context event is too large")
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO mission_events(scope, request_id, kind, payload, timestamp) VALUES (?, ?, ?, ?, ?)",
                (self._scope, request_id, kind, encoded, time.time()),
            )

    def recent(self, *, exclude_request_id: str = "", limit: int = 20, max_chars: int = 16000) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValidationError("mission context window must be within 1..100 events")
        if type(max_chars) is not int or not 1000 <= max_chars <= 64000:
            raise ValidationError("mission context budget must be within 1000..64000 characters")
        with self._lock:
            rows = self._db.execute(
                "SELECT request_id, kind, payload, timestamp FROM mission_events "
                "WHERE scope = ? AND request_id != ? ORDER BY id DESC LIMIT ?",
                (self._scope, exclude_request_id, limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        used = 2
        for request_id, kind, payload, timestamp in rows:
            event = {"request_id": request_id, "kind": kind, "data": json.loads(payload), "timestamp": timestamp}
            size = len(json.dumps(event)) + 2
            if used + size > max_chars:
                break  # Keep a contiguous recent window; do not skip newer context for older data.
            result.append(event)
            used += size
        return list(reversed(result))

    def close(self) -> None:
        with self._lock:
            self._db.close()
