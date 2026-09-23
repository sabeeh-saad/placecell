"""Bounded, asynchronous mission diagnostics, independent of command persistence.

Trace loss is observable and never authorizes or replays movement. Context is bound
explicitly across worker/action callbacks so late work retains its original identity.
"""

from __future__ import annotations

import contextvars
import functools
import json
import math
import os
import queue
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from placecell.errors import ValidationError

_APPLICATION_ID = 0x50435452
_P = ParamSpec("_P")
_T = TypeVar("_T")
_HEALTH_COUNTS = (
    "dropped_events", "write_errors", "trimmed_events", "truncated_events", "unclean_shutdowns",
    "coalesced_events", "dropped_critical_events",
)
_SECRET_KEYS = {"api_key", "authorization", "password", "secret", "access_token", "refresh_token", "headers", "cookie"}
_SENSITIVE = re.compile(
    r"(?i)(?:bearer\s+\S+|(?:api[_ -]?key|password|secret|access_token)[\"']?\s*[:=]\s*[\"']?[^\s,\"']+"
    r"|\bsk-[a-z0-9_-]+|\bAIza[a-z0-9_-]+|https?://\S+|data:image/\S+)"
)


def _clean(value: Any, secrets: tuple[str, ...], clipped: list[bool], depth: int = 0) -> Any:
    if depth > 6:
        clipped[0] = True
        return "[depth limit]"
    if isinstance(value, str):
        if len(value) > 8192:
            clipped[0] = True
            return "[text exceeds 8192 characters]"
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        value = _SENSITIVE.sub("[redacted]", value)
        clipped[0] |= len(value) > 2000
        return value[:2000]
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        clipped[0] |= len(value) > 64
        return {
            str(k)[:100]: "[redacted]" if str(k).casefold() in _SECRET_KEYS else _clean(v, secrets, clipped, depth + 1)
            for k, v in list(value.items())[:64]
        }
    if isinstance(value, list | tuple):
        clipped[0] |= len(value) > 64
        return [_clean(v, secrets, clipped, depth + 1) for v in value[:64]]
    return f"[unsupported {type(value).__name__}]"


@dataclass(frozen=True)
class TraceContext:
    store: TraceStore
    mission_id: str
    request_id: str
    step: int = 0

    def emit(self, stage: str, *, kind: str = "event", **data: Any) -> None:
        self.store.record(self, stage, kind, data)


_CURRENT: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar("placecell_trace", default=None)
_PARENT: contextvars.ContextVar[str | None] = contextvars.ContextVar("placecell_span", default=None)


def current_trace() -> TraceContext | None:
    return _CURRENT.get()


@contextmanager
def trace_scope(context: TraceContext | None) -> Iterator[None]:
    token, parent = _CURRENT.set(context), _PARENT.set(None)
    try:
        yield
    finally:
        _PARENT.reset(parent)
        _CURRENT.reset(token)


def bind_trace(context: TraceContext | None, fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """Capture immutable mission/step IDs at dispatch time, never from a later active mission."""

    @functools.wraps(fn)
    def bound(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        with trace_scope(context):
            return fn(*args, **kwargs)

    return bound


def trace_event(stage: str, **data: Any) -> None:
    context = current_trace()
    if context:
        context.emit(stage, **data)


@contextmanager
def trace_span(stage: str, **data: Any) -> Iterator[dict[str, Any]]:
    context = current_trace()
    if context is None:
        yield data
        return
    identity, parent, started = uuid.uuid4().hex, _PARENT.get(), time.perf_counter()
    context.emit(stage, kind="start", span_id=identity, parent_span_id=parent, **data)
    token = _PARENT.set(identity)
    outcome = "completed"
    try:
        yield data
    except BaseException as exc:
        outcome = "error"
        data["error_type"] = type(exc).__name__  # Raw exception strings can contain credentials or HTTP bodies.
        raise
    finally:
        _PARENT.reset(token)
        context.emit(
            stage,
            kind="end",
            span_id=identity,
            parent_span_id=parent,
            duration_ms=(time.perf_counter() - started) * 1000,
            outcome=outcome,
            **data,
        )


def traced(stage: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def decorate(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            with trace_span(stage):
                return fn(*args, **kwargs)

        return wrapped

    return decorate


def provider_usage(body: Any) -> dict[str, int | float | None]:
    """Keep reported usage only. Missing, invalid and unspecified-currency costs remain unknown."""
    raw = body.get("usage", body.get("usageMetadata", {})) if isinstance(body, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    result: dict[str, int | float | None] = {}
    for name, aliases in {
        "input_tokens": ("prompt_tokens", "input_tokens", "promptTokenCount"),
        "output_tokens": ("completion_tokens", "output_tokens", "candidatesTokenCount"),
        "total_tokens": ("total_tokens", "totalTokenCount"),
        "cost_usd": ("cost_usd",),
    }.items():
        value = next((raw[key] for key in aliases if key in raw), None)
        valid = type(value) is int and value >= 0
        if name == "cost_usd":
            valid = isinstance(value, int | float) and not isinstance(value, bool) and 0 <= value <= 1e15
        result[name] = value if valid else None
    return result


@dataclass
class _Barrier:
    done: threading.Event = field(default_factory=threading.Event)
    stop: bool = False


@dataclass(frozen=True)
class _QueuedEvent:
    row: tuple[str, str]
    progress_key: tuple[str, str, int, str] | None = None


class _TraceQueue(queue.Queue[_QueuedEvent | _Barrier]):
    """FIFO critical events, bounded latest progress, and ordered flush barriers."""

    def offer(self, event: _QueuedEvent) -> str:
        with self.not_full:
            if event.progress_key is not None:
                for index in range(len(self.queue) - 1, -1, -1):
                    previous = self.queue[index]
                    if isinstance(previous, _Barrier):
                        break  # Never move post-flush work ahead of its barrier.
                    if previous.progress_key == event.progress_key:
                        del self.queue[index]
                        self._put(event)  # Keep the latest sample after intervening transitions.
                        return "coalesced"
            outcome = "added"
            if self._qsize() >= self.maxsize:
                if event.progress_key is not None:
                    return "dropped"
                for index, previous in enumerate(self.queue):
                    if isinstance(previous, _QueuedEvent) and previous.progress_key is not None:
                        del self.queue[index]
                        self.unfinished_tasks -= 1
                        outcome = "evicted_progress"
                        break
                else:
                    return "dropped"  # Critical-only saturation remains observable.
            self._put(event)
            self.unfinished_tasks += 1
            self.not_empty.notify()
            return outcome


class TraceStore:
    """Single-writer SQLite ring of events; callbacks only sanitize and enqueue.

    Retention is by total event count, not mission. Exports report global retention/loss
    counters and incomplete spans. The main database has a SQLite page cap; its rollback
    journal can temporarily use additional space of comparable size.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_events: int = 10000,
        max_bytes: int = 16 * 1024 * 1024,
        queue_size: int = 256,
        secrets: Sequence[str] = (),
    ) -> None:
        import fcntl  # File-backed tracing currently targets the Linux/Unix ROS deployment.

        if any(type(v) is not int or v <= 0 for v in (max_events, max_bytes, queue_size)) or max_bytes < 65536:
            raise ValidationError("trace limits must be positive integers; database budget must be at least 64 KiB")
        if str(path) == ":memory:":
            raise ValidationError("traces require a file path")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        except FileExistsError:
            pass
        self._file_lock = self.path.open("rb")
        try:
            fcntl.flock(self._file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._db = sqlite3.connect(str(self.path), timeout=0.05, check_same_thread=False)
        except Exception:
            self._file_lock.close()
            raise
        try:
            application_id = self._db.execute("PRAGMA application_id").fetchone()[0]
            tables = self._db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if application_id not in (0, _APPLICATION_ID) or (application_id == 0 and tables):
                raise ValidationError("trace path belongs to another database")
            self._db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
            self._db.execute("PRAGMA journal_mode=DELETE")
            pages = max_bytes // self._db.execute("PRAGMA page_size").fetchone()[0]
            if self._db.execute("PRAGMA page_count").fetchone()[0] > pages:
                raise ValidationError("existing trace database exceeds the configured page budget")
            self._db.execute(f"PRAGMA max_page_count={pages}")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS trace_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                "mission_id TEXT NOT NULL, body TEXT NOT NULL)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS trace_mission ON trace_events(mission_id, seq)")
            self._db.execute("CREATE TABLE IF NOT EXISTS trace_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            old = dict(self._db.execute("SELECT key,value FROM trace_meta"))
            if old.get("schema_version", "1") != "1":
                raise ValidationError("unsupported trace schema version")
            self._counts = {
                key: int(old.get(key, "0"))
                for key in _HEALTH_COUNTS
            }
            self._counts["unclean_shutdowns"] += int(old.get("open_session", "false") == "true")
            self._db.execute("INSERT OR REPLACE INTO trace_meta VALUES ('schema_version', '1')")
            self._db.execute("INSERT OR REPLACE INTO trace_meta VALUES ('open_session', 'true')")
            self._db.commit()
        except Exception:
            self._db.close()
            self._file_lock.close()
            raise
        self._max_events = max_events
        self._payload_budget = max_bytes // 2  # Leave room for indexes, metadata and SQLite page overhead.
        self._secrets = tuple(sorted({s for s in secrets if s}, key=len, reverse=True))
        self._queue = _TraceQueue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._accepting = True
        self._last_error = ""
        self._session = uuid.uuid4().hex
        self._thread = threading.Thread(target=self._run, name="placecell-traces", daemon=True)
        self._thread.start()

    def context(self, mission_id: str, request_id: str, step: int = 0) -> TraceContext:
        return TraceContext(self, mission_id, request_id, step)

    def register_secrets(self, secrets: Sequence[str]) -> None:
        """Register configured credential values, never provider text or whole request bodies."""
        with self._lock:
            self._secrets = tuple(sorted(set(self._secrets).union(s for s in secrets if s), key=len, reverse=True))

    def record(self, context: TraceContext, stage: str, kind: str, data: dict[str, Any]) -> None:
        # Diagnostics are best effort. Provider/trace data cannot raise into a movement callback.
        try:
            clipped = [False]
            event = {
                "schema_version": 1,
                "session_id": self._session,
                "mission_id": context.mission_id[:128],
                "request_id": context.request_id[:128],
                "step": context.step,
                "stage": stage[:80],
                "kind": kind[:16],
                "timestamp": time.time(),
                "monotonic_s": time.monotonic(),
                "data": _clean(data, self._secrets, clipped),
            }
            encoded = json.dumps(event, allow_nan=False)
            if len(encoded.encode()) > 16384:
                event["data"] = {"omitted": "event exceeded 16 KiB"}
                clipped[0] = True
            if clipped[0]:
                event["truncated"] = True
                encoded = json.dumps(event, allow_nan=False)
                with self._lock:
                    self._counts["truncated_events"] += 1
            progress = (
                kind == "event"
                and stage in {"nav2.event", "status"}
                and data.get("state") in {"navigating", "canceling"}
                and not data.get("message")
                and data.get("distance_remaining") is not None
            )
            key = (context.mission_id, context.request_id, context.step, stage) if progress else None
            with self._lock:
                if not self._accepting:
                    self._counts["dropped_events"] += 1
                    self._counts["dropped_critical_events"] += int(not progress)
                    return
                outcome = self._queue.offer(_QueuedEvent((context.mission_id[:128], encoded), key))
                if outcome == "coalesced":
                    self._counts["coalesced_events"] += 1
                elif outcome in {"dropped", "evicted_progress"}:
                    self._counts["dropped_events"] += 1
                    self._counts["dropped_critical_events"] += int(outcome == "dropped" and not progress)
        except Exception as exc:
            with self._lock:
                self._counts["dropped_events"] += 1
                self._counts["dropped_critical_events"] += 1
                self._last_error = type(exc).__name__

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._counts,
                "last_error_type": self._last_error,
                "pending": self._queue.qsize(),
                "writer_alive": self._thread.is_alive(),
            }

    def _save_health(self) -> None:
        with self._lock:
            values = [(key, str(value)) for key, value in self._counts.items()]
        self._db.executemany("INSERT OR REPLACE INTO trace_meta(key,value) VALUES (?,?)", values)

    def _write(self, item: tuple[str, str]) -> None:
        with self._db:
            count, size = self._db.execute(
                "SELECT count(*),coalesce(sum(length(CAST(body AS BLOB))),0) FROM trace_events"
            ).fetchone()
            removed = 0
            while count and (count >= self._max_events or size + len(item[1].encode()) > self._payload_budget):
                seq, length = self._db.execute(
                    "SELECT seq,length(CAST(body AS BLOB)) FROM trace_events ORDER BY seq LIMIT 1"
                ).fetchone()
                self._db.execute("DELETE FROM trace_events WHERE seq=?", (seq,))
                count, size, removed = count - 1, size - length, removed + 1
            self._db.execute("INSERT INTO trace_events(mission_id,body) VALUES (?,?)", item)
            self._save_health()
        with self._lock:
            self._counts["trimmed_events"] += removed

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if isinstance(item, _Barrier):
                        with self._db:
                            self._save_health()
                            if item.stop:
                                self._db.execute("UPDATE trace_meta SET value='false' WHERE key='open_session'")
                    else:
                        self._write(item.row)
                except Exception as exc:
                    with self._lock:
                        self._counts["write_errors"] += 1
                        self._last_error = type(exc).__name__
                finally:
                    self._queue.task_done()
                    if isinstance(item, _Barrier):
                        item.done.set()
                if isinstance(item, _Barrier) and item.stop:
                    return
        finally:
            self._db.close()
            self._file_lock.close()

    def flush(self, timeout: float = 2.0) -> bool:
        barrier = _Barrier()
        started = time.monotonic()
        with self._lock:
            accepting = self._accepting
        if not accepting:
            return not self._thread.is_alive()
        try:
            self._queue.put(barrier, timeout=timeout)
        except queue.Full:
            return False
        return barrier.done.wait(max(0.0, timeout - (time.monotonic() - started)))

    def close(self, timeout: float = 2.0) -> bool:
        with self._lock:
            if not self._accepting:
                return not self._thread.is_alive()
            self._accepting = False
        barrier = _Barrier(stop=True)
        started = time.monotonic()
        try:
            self._queue.put(barrier, timeout=timeout)
        except queue.Full:
            # A later explicit close can retry once the writer has made progress.
            with self._lock:
                self._accepting = True
            return False
        self._thread.join(max(0.0, timeout - (time.monotonic() - started)))
        return not self._thread.is_alive()


def read_trace(path: str | Path, mission_id: str | None = None) -> dict[str, Any]:
    """Read a committed snapshot without creating, repairing or resuming anything."""
    uri = Path(path).expanduser().resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=0.1)
    try:
        if db.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID:
            raise ValidationError("not a PlaceCell trace database")
        db.execute("BEGIN")
        meta = dict(db.execute("SELECT key,value FROM trace_meta"))
        if meta.get("schema_version") != "1":
            raise ValidationError("unsupported trace schema version")
        rows = db.execute(
            "SELECT seq,body FROM trace_events WHERE (? IS NULL OR mission_id=?) ORDER BY seq", (mission_id, mission_id)
        ).fetchall()
    finally:
        db.close()
    events = [{"sequence": seq, **json.loads(body)} for seq, body in rows]
    if mission_id is not None and not events:
        raise ValidationError("mission has no retained trace events")
    starts = {e["data"].get("span_id"): e for e in events if e["kind"] == "start"}
    ended = {e["data"].get("span_id") for e in events if e["kind"] == "end"}
    usage = [e["data"].get("usage", {}) for e in events if e["stage"] == "provider_request" and e["kind"] == "end"]
    losses = {
        key: int(meta.get(key, "0"))
        for key in _HEALTH_COUNTS
        if key != "coalesced_events"
    }
    totals = {}
    for field_name in ("input_tokens", "output_tokens", "total_tokens", "cost_usd"):
        known = [u[field_name] for u in usage if u.get(field_name) is not None]
        totals[field_name] = {
            "known_sum": sum(known),
            "reported_calls": len(known),
            "total": sum(known) if usage and len(known) == len(usage) else None,
        }
    return {
        "schema_version": 1,
        "mission_id": mission_id,
        "health": {key: json.loads(value) for key, value in meta.items()},
        "summary": {
            "capture_status": "open_or_unclean" if meta.get("open_session") == "true" else "closed",
            "loss_counters_nonzero": [key for key, value in losses.items() if value],
            "event_count": len(events),
            "mission_ids": list(dict.fromkeys(e["mission_id"] for e in events)),
            "last_status": next((e["data"] for e in reversed(events) if e["stage"] == "status"), None),
            "unfinished_spans": [
                {"stage": e["stage"], "request_id": e["request_id"], "span_id": identity}
                for identity, e in starts.items()
                if identity not in ended
            ],
            "provider_response_usage": totals,
            "unreported_retry_attempts": sum(
                max(0, e["data"].get("attempts", 1) - 1)
                for e in events
                if e["stage"] == "provider_request" and e["kind"] == "end"
            ),
            "usage_scope": "HTTP final responses only; retries/custom providers may add unknown usage",
        },
        "events": events,
    }
