"""Print a failed offline check's report as one GitHub Actions error annotation.

Annotations are readable without signing in, unlike the job log. Runs on the host, stdlib only.
"""

import argparse
import json
import re
from pathlib import Path

LIMIT = 4000  # Bytes of escaped message; GitHub cuts longer annotations.
FRAMES = 3  # Innermost traceback frames shown with their source line.
MARKER = " [truncated]"
FRAME = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<name>.+)$')


def escape_data(text):
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(text):
    return escape_data(text).replace(":", "%3A").replace(",", "%2C")


def fit(text, limit=LIMIT):
    """Longest prefix of text whose escaped form, plus a marker if cut, stays within limit bytes."""
    if len(escape_data(text).encode()) <= limit:
        return text
    budget, size = limit - len(MARKER), 0
    for index, char in enumerate(text):
        size += len(escape_data(char).encode())
        if size > budget:
            return text[:index] + MARKER
    return text


def reason(error):
    """Exception line plus the innermost frames' locations and source lines."""
    # Only the last traceback of a chain raised the reported exception.
    lines = [line for line in error.split("Traceback (most recent call last):")[-1].splitlines() if line.strip()]
    frames, end = [], 0
    for index, line in enumerate(lines):
        match = FRAME.match(line)
        if not match:
            continue
        location = f"{Path(match['path']).name}:{match['line']} in {match['name']}"
        end = index + 1
        # Indented lines after a frame are its source and, on 3.11+, caret markers.
        while end < len(lines) and lines[end].startswith(" ") and not FRAME.match(lines[end]):
            end += 1
        source = lines[index + 1].strip() if index + 1 < end else ""
        frames.append(f"{location}: {source}" if source and source.strip("^~ ") else location)
    exception = "\n".join(line.rstrip() for line in lines[end:]) or "\n".join(lines)
    return "\n".join([exception, *frames[-FRAMES:]])


def last_trial(report):
    trials = report.get("trials") or report.get("cancellation_trials") or []
    failed = [row for row in trials if isinstance(row, dict) and not row.get("passed", True)]
    if not failed:
        return None
    row = failed[-1]
    keys = ("id", "mode", *sorted(key for key in row if key.endswith("_ms")))
    return {key: row[key] for key in keys if key in row}


def summary(report):
    parts = []
    if report.get("error"):
        parts.append(reason(report["error"]))
    if report.get("capture_error"):
        parts.append(f"capture: {report['capture_error']}")
    checks = report.get("checks")
    if isinstance(checks, list):
        parts.append(f"checks passed: {len(checks)}" + (f", last: {checks[-1]}" if checks else ""))
    trial = last_trial(report)
    if trial:
        parts.append(f"failing trial: {json.dumps(trial, sort_keys=True)}")
    for key in ("latency_ms", "measurements", "trace_statuses_match"):
        if key in report:
            parts.append(f"{key}: {json.dumps(report[key], sort_keys=True)}")
    health = {key: value for key, value in (report.get("trace_health") or {}).items() if value}
    if health:
        parts.append(f"trace_health: {json.dumps(health, sort_keys=True)}")
    return "\n".join(parts) or "report.json has no failure details"


def annotation(check, report_path, exit_code):
    try:
        report = json.loads(Path(report_path).read_text())
        message = summary(report) if isinstance(report, dict) else "report.json is not an object"
    except FileNotFoundError:
        message = "no report.json was written" + (" (480 s timeout)" if exit_code == 124 else "")
    except (OSError, ValueError) as error:
        message = f"report.json is unreadable: {error}"
    message = f"exit code {exit_code}\n{message}"
    return f"::error title={escape_property(check)}::{escape_data(fit(message))}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check")
    parser.add_argument("report", type=Path)
    parser.add_argument("exit_code", type=int)
    args = parser.parse_args()
    print(annotation(args.check, args.report, args.exit_code), flush=True)  # noqa: T201 - workflow command


if __name__ == "__main__":
    main()
