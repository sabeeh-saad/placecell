"""Operator feedback on answers, and how it changes what the memory says next time.

A correction names the memory an answer relied on and says whether it was right or wrong.
Wrong verdicts push a memory down in retrieval and, repeated, get it superseded by the
curator. Right verdicts count as confirmations. Bounded logs refuse new feedback at
capacity instead of silently forgetting negative verdicts for retained memories.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from placecell.errors import ValidationError

VERDICTS = frozenset({"right", "wrong"})


@dataclass(frozen=True, slots=True)
class Correction:
    memory_id: str
    verdict: str
    question: str = ""
    note: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("memory_id", self.memory_id, 256),
            ("question", self.question, 2000),
            ("note", self.note, 2000),
        ):
            if not isinstance(value, str) or len(value) > limit or (name == "memory_id" and not value.strip()):
                raise ValidationError(f"correction {name} is invalid or exceeds {limit} characters")
        if self.verdict not in VERDICTS:
            raise ValidationError(f"verdict must be one of {sorted(VERDICTS)}")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValidationError("timestamp must be a finite, non-negative unix time")


@dataclass(frozen=True, slots=True)
class Verdicts:
    right: int = 0
    wrong: int = 0

    @property
    def weight(self) -> float:
        """Score multiplier: each wrong verdict halves it, right verdicts cancel wrong ones."""
        return 0.5 ** max(0, self.wrong - self.right)


@runtime_checkable
class CorrectionLog(Protocol):
    def record(self, correction: Correction) -> None: ...

    def verdicts(self, memory_ids: Iterable[str]) -> dict[str, Verdicts]:
        """Counts per id, for the ids given. Ids without corrections are absent."""
        ...


class InMemoryCorrectionLog:
    def __init__(self, *, max_records: int = 10000, max_bytes: int = 4_194_304) -> None:
        if type(max_records) is not int or max_records < 1 or type(max_bytes) is not int or max_bytes < 1024:
            raise ValidationError("correction limits need a positive record count and at least 1024 bytes")
        self._max_records, self._max_bytes = max_records, max_bytes
        self._rows: list[Correction] = []
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _encode(correction: Correction) -> str:
        return json.dumps(asdict(correction), allow_nan=False) + "\n"

    def _write(self, rows: list[Correction]) -> None:
        """Persistent variants replace their file before changing in-memory verdicts."""

    def record(self, correction: Correction) -> None:
        size = len(self._encode(correction).encode())
        with self._lock:
            if len(self._rows) >= self._max_records or self._bytes + size > self._max_bytes:
                raise ValidationError(
                    "correction capacity reached; prune feedback for deleted memories or raise limits"
                )
            rows = [*self._rows, correction]
            self._write(rows)
            self._rows, self._bytes = rows, self._bytes + size

    def prune(self, retained_memory_ids: Iterable[str]) -> int:
        """Drop feedback only for deleted memories; retained verdict counts never decay."""
        retained = set(retained_memory_ids)
        with self._lock:
            rows = [row for row in self._rows if row.memory_id in retained]
            removed = len(self._rows) - len(rows)
            if removed:
                self._write(rows)
                self._rows = rows
                self._bytes = sum(len(self._encode(row).encode()) for row in rows)
            return removed

    def verdicts(self, memory_ids: Iterable[str]) -> dict[str, Verdicts]:
        wanted = set(memory_ids)
        out: dict[str, Verdicts] = {}
        with self._lock:
            for c in self._rows:
                if c.memory_id in wanted:
                    v = out.get(c.memory_id, Verdicts())
                    out[c.memory_id] = Verdicts(v.right + (c.verdict == "right"), v.wrong + (c.verdict == "wrong"))
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)


class JsonlCorrectionLog(InMemoryCorrectionLog):
    """Bounded JSON lines, atomically replaced before acknowledging each change.

    A legacy file over the configured limits is refused intact. One writer owns a log;
    multi-process merging belongs to an explicit import workflow.
    """

    def __init__(self, path: str | Path, *, max_records: int = 10000, max_bytes: int = 4_194_304) -> None:
        super().__init__(max_records=max_records, max_bytes=max_bytes)
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            if self._path.stat().st_size > self._max_bytes:
                raise ValidationError("existing correction file exceeds configured byte capacity")
            with self._path.open() as f:
                for line in f:
                    if line.strip():
                        if len(self._rows) >= self._max_records:
                            raise ValidationError("existing correction file exceeds configured record capacity")
                        row = Correction(**json.loads(line))
                        self._rows.append(row)
                        self._bytes += len(self._encode(row).encode())
            if self._bytes > self._max_bytes:
                raise ValidationError("normalized correction records exceed configured byte capacity")
        else:
            # Make an initialized empty log explicit for offline backup inventory.
            self._write([])

    def _write(self, rows: list[Correction]) -> None:
        # Same-directory replacement preserves the old log if serialization or writing fails.
        pending: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self._path.parent, delete=False) as f:
                pending = Path(f.name)
                for row in rows:
                    f.write(self._encode(row))
                f.flush()
                os.fsync(f.fileno())
            pending.replace(self._path)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)


def correction_now(memory_id: str, verdict: str, question: str = "", note: str = "") -> Correction:
    return Correction(memory_id, verdict, question, note, time.time())
