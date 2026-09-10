"""Operator feedback on answers, and how it changes what the memory says next time.

A correction names the memory an answer relied on and says whether it was right or wrong.
Wrong verdicts push a memory down in retrieval and, repeated, get it superseded by the
curator. Right verdicts count as confirmations. The log is append-only, so it can be merged
across robots by concatenation.
"""

from __future__ import annotations

import json
import math
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
        if not self.memory_id:
            raise ValidationError("memory_id must not be empty")
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
    def __init__(self) -> None:
        self._rows: list[Correction] = []
        self._lock = threading.Lock()

    def record(self, correction: Correction) -> None:
        with self._lock:
            self._rows.append(correction)

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
        return len(self._rows)


class JsonlCorrectionLog(InMemoryCorrectionLog):
    """Append-only JSON lines file. Loaded once, every record is flushed immediately."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            with self._path.open() as f:
                for line in f:
                    if line.strip():
                        super().record(Correction(**json.loads(line)))

    def record(self, correction: Correction) -> None:
        super().record(correction)
        with self._lock, self._path.open("a") as f:
            f.write(json.dumps(asdict(correction)) + "\n")


def correction_now(memory_id: str, verdict: str, question: str = "", note: str = "") -> Correction:
    return Correction(memory_id, verdict, question, note, time.time())
