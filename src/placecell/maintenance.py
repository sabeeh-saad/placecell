"""Cooperative maintenance exclusion for the single-controller deployment."""

from __future__ import annotations

import fcntl
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO

from placecell.errors import ValidationError

STORAGE_KEYS = (
    "db_path",
    "keyframe_dir",
    "corrections_path",
    "command_journal_path",
    "mission_context_path",
    "mission_trace_path",
    "navigation_ownership_path",
)


class StorageLease:
    """Hold stable sidecar inodes; never delete a lock file after releasing it."""

    def __init__(self, paths: list[Path]) -> None:
        self._stack = ExitStack()
        try:
            for path in sorted({p.expanduser().resolve() for p in paths}):
                path.parent.mkdir(parents=True, exist_ok=True)
                handle: BinaryIO = self._stack.enter_context(path.open("a+b"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise ValidationError(f"Storage is in use; stop its controller/writers first: {path}") from exc
        except BaseException:
            self.close()
            raise

    @classmethod
    def for_parameters(cls, parameters: Mapping[str, object]) -> StorageLease:
        paths = []
        for key in STORAGE_KEYS:
            value = parameters.get(key)
            if value and value != ":memory:":
                path = Path(str(value)).expanduser().resolve()
                paths.append(path.with_name(path.name + ".maintenance.lock"))
        return cls(paths)

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> StorageLease:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
