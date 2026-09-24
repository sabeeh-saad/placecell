"""Exclusive, durable ownership of one scoped Nav2 goal, never a replay queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from placecell.errors import ValidationError


@dataclass(frozen=True)
class NavigationScope:
    robot_id: str
    map_id: str
    action_name: str

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() or len(v) > 256 for v in asdict(self).values()):
            raise ValidationError("Navigation ownership requires a robot, versioned map and action name.")
        if not self.action_name.startswith("/"):
            raise ValidationError("Navigation ownership requires a fully resolved action name.")


class NavigationOwnership:
    """One process holds the lock for its entire transport lifetime.

    Missing state starts unknown. Only explicit operator attestation or a terminal
    result for the recorded goal establishes clean ownership. Scope changes fail
    closed instead of hiding another robot/map/server's outstanding goal.
    """

    def __init__(self, path: str | Path, scope: NavigationScope) -> None:
        if not str(path).strip() or str(path) == ":memory:":
            raise ValidationError("Navigation ownership requires a persistent journal path.")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._lease = self.path.with_suffix(self.path.suffix + ".lock").open("a+b")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lease.close()
            raise ValidationError("Navigation ownership is already held by another process.") from exc
        try:
            self._db = sqlite3.connect(str(self.path), timeout=0.05, check_same_thread=False)
            try:
                self._db.execute("PRAGMA synchronous=FULL")
                if self._db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise ValidationError("Navigation ownership integrity check failed.")
                scope_key = json.dumps(asdict(scope), sort_keys=True)
                with self._db:
                    self._db.execute(
                        "CREATE TABLE IF NOT EXISTS ownership (id INTEGER PRIMARY KEY CHECK(id=1), "
                        "version INTEGER NOT NULL, scope TEXT NOT NULL, state TEXT NOT NULL, "
                        "goal_id TEXT NOT NULL, request_id TEXT NOT NULL, outcome TEXT NOT NULL)"
                    )
                    self._db.execute("INSERT OR IGNORE INTO ownership VALUES (1,1,?,'unknown','','','')", (scope_key,))
                row = self._db.execute("SELECT version,scope,state,goal_id FROM ownership WHERE id=1").fetchone()
                if row[:2] != (1, scope_key) or row[2] not in {"unknown", "clean", "pending"}:
                    raise ValidationError("Navigation ownership version, scope or state mismatch.")
                if row[2] == "pending" and (
                    not row[3] or uuid.UUID(hex=row[3]).hex != row[3] or uuid.UUID(hex=row[3]).int == 0
                ):
                    raise ValidationError("Navigation ownership has an invalid goal identity.")
            except BaseException:
                self._db.close()
                raise
        except BaseException:
            self._lease.close()
            raise

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            row = self._db.execute("SELECT state,goal_id,request_id,outcome FROM ownership WHERE id=1").fetchone()
            return dict(zip(("state", "goal_id", "request_id", "outcome"), row, strict=True))

    def reserve(self, request_id: str) -> str:
        """Commit the exact wire UUID before any send request can reach Nav2."""
        goal_id = uuid.uuid4().hex
        with self._lock, self._db:
            changed = self._db.execute(
                "UPDATE ownership SET state='pending',goal_id=?,request_id=?,outcome='' WHERE id=1 AND state='clean'",
                (goal_id, request_id),
            ).rowcount
            if changed != 1:
                raise ValidationError("Previous navigation ownership is unresolved.")
        return goal_id

    def terminal(self, goal_id: str, outcome: str) -> None:
        if outcome not in {"succeeded", "canceled", "failed", "rejected"}:
            raise ValidationError("Only confirmed terminal results can release navigation ownership.")
        with self._lock, self._db:
            changed = self._db.execute(
                "UPDATE ownership SET state='clean',outcome=? WHERE id=1 AND state='pending' AND goal_id=?",
                (outcome, goal_id),
            ).rowcount
            if changed != 1:
                raise ValidationError("Terminal navigation identity does not match the owned goal.")

    def attest_clean(self, reason: str) -> None:
        """Operator has independently stopped/reset this Nav2 server with clients quiescent."""
        if not reason.strip() or len(reason) > 1000:
            raise ValidationError("Record how the Nav2 server was independently stopped/reset (1..1000 characters).")
        with self._lock, self._db:
            self._db.execute("UPDATE ownership SET state='clean',outcome=? WHERE id=1", ("operator: " + reason,))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._db.close()
                self._lease.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect navigation ownership or record an independent server reset.")
    parser.add_argument("operation", choices=("inspect", "attest-clean"))
    parser.add_argument("--journal", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--action-name", required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--confirm-nav2-stopped", action="store_true")
    args = parser.parse_args(argv)
    if args.operation == "attest-clean" and (not args.confirm_nav2_stopped or not args.reason.strip()):
        parser.error("attest-clean requires --confirm-nav2-stopped and --reason; it does not stop Nav2 for you")
    owner = NavigationOwnership(args.journal, NavigationScope(args.robot_id, args.map_id, args.action_name))
    try:
        if args.operation == "attest-clean":
            owner.attest_clean(args.reason)
        print(json.dumps(owner.snapshot(), sort_keys=True))  # noqa: T201 - CLI report
    finally:
        owner.close()


if __name__ == "__main__":  # pragma: no cover
    main()
