"""Version admission before schema or projection mutation.

Version zero is the unversioned Day 14/15 SQLite layout. Version one records that
the existing idempotent column/table migrations have completed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from placecell.errors import ValidationError

STATE_VERSION = 1


def check_connection(connection: sqlite3.Connection) -> int:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version not in (0, STATE_VERSION):
        raise ValidationError(f"Unsupported state schema {version}; use compatible software or restore a backup.")
    if [tuple(row) for row in connection.execute("PRAGMA quick_check")] != [("ok",)]:
        raise ValidationError("State integrity check failed; preserve the damaged files and restore a backup.")
    return version


def check_file(path: Path) -> None:
    if path.exists():
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            check_connection(connection)
        finally:
            connection.close()
