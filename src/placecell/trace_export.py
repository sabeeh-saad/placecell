"""Export retained PlaceCell mission traces without moving a robot."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from placecell.errors import ValidationError
from placecell.tracing import read_trace


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--mission-id", help="Omit to export all retained missions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    try:
        report = read_trace(args.database, args.mission_id)
    except (sqlite3.Error, ValidationError) as exc:
        parser.error(f"cannot export trace: {type(exc).__name__}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    sys.stdout.write(json.dumps({"events": len(report["events"]), "missions": report["summary"]["mission_ids"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
