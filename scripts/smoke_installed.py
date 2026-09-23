"""Exercise an installed PlaceCell distribution using only its core dependencies and scripted data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import placecell
from placecell import CollectionInfo, Evidence, EvidenceKind, InMemoryStore, Memory, Pose, Recall
from placecell.providers import HashingEmbedder
from placecell.tracing import TraceStore


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    output, fixtures = args.output.resolve(), args.fixtures.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = Path(placecell.__file__).resolve()
    require(package.is_relative_to(Path(sys.prefix)), f"Imported outside the clean environment: {package}")
    require((package.parent / "py.typed").is_file(), "Distribution is missing the typing marker")
    distribution = importlib.metadata.distribution("placecell")
    entrypoints = {entry.name: entry for entry in distribution.entry_points if entry.group == "console_scripts"}
    expected = {
        "placecell-ros2",
        "placecell-listen",
        "placecell-evaluate",
        "placecell-evaluate-objects",
        "placecell-evaluate-missions",
        "placecell-check-faults",
        "placecell-export-trace",
    }
    require(expected <= entrypoints.keys(), "Distribution is missing a supported entry point")
    for name in expected:
        require(callable(entrypoints[name].load()), f"Cannot load entry point {name}")
        require((Path(sys.executable).parent / name).is_file(), f"Missing installed command {name}")

    def cli(name: str, *arguments: str) -> None:
        with (output / "cli.log").open("a") as log:
            subprocess.run(  # noqa: S603 - fixed installed entry points and caller-owned file paths
                [str(Path(sys.executable).parent / name), *arguments],
                check=True,
                timeout=90,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    # ROS and microphone dependencies are optional; loading their entry points above is sufficient here.
    # Real ROS startup is checked by the separate simulation job.
    for name in sorted(expected - {"placecell-ros2", "placecell-listen"}):
        cli(name, "--help")

    embedder = HashingEmbedder(64)
    store = InMemoryStore(CollectionInfo("installed-smoke", embedder.model_name, embedder.dimension))
    try:
        memory = Memory.create("robot", "front", 1000, Pose(1, 2), Evidence(EvidenceKind.FRAME, "frame.jpg"), "printer")
        memory = memory.with_embedding(embedder.embed_text(["printer"])[0], embedder.model_name, kind="caption")
        store.upsert([memory])
        hits = Recall(store, embedder, clock=lambda: 1000).similar("printer", k=1)
        require(bool(hits) and hits[0].memory.id == memory.id, "Installed memory/retrieval smoke failed")
    finally:
        store.close()

    cli(
        "placecell-evaluate-missions",
        "scripted",
        "--dataset",
        str(fixtures / "baseline-v1.json"),
        "--replies",
        str(fixtures / "scripted-replies-v1.json"),
        "--output",
        str(output / "missions.json"),
        "--save-trials",
        str(output / "trials.json"),
    )
    missions = json.loads((output / "missions.json").read_text())
    require(missions["plan"]["assessed"] > 0 and missions["plan"]["failed"] == 0, "Installed mission baseline failed")
    require(missions["total_cost_usd"] == 0, "Scripted baseline unexpectedly has provider costs")
    cli("placecell-check-faults", "--output", str(output / "faults.json"))
    faults = json.loads((output / "faults.json").read_text())
    require(faults["summary"]["runs"] > 0 and faults["summary"]["failed"] == 0, "Installed fault contracts failed")
    require(faults["paid_api_calls"] == 0, "Offline fault runner unexpectedly used a provider")

    traces = TraceStore(output / "traces.sqlite3")
    traces.context("installed-mission", "installed-request").emit("status", state="succeeded")
    require(traces.close(), "Installed trace writer failed to close")
    cli(
        "placecell-export-trace",
        "--database",
        str(output / "traces.sqlite3"),
        "--mission-id",
        "installed-mission",
        "--output",
        str(output / "trace-export.json"),
    )
    exported = json.loads((output / "trace-export.json").read_text())
    require(exported["summary"]["last_status"]["state"] == "succeeded", "Installed trace export lost its outcome")
    report = {
        "passed": True,
        "version": distribution.version,
        "python": sys.version,
        "package_path": str(package),
        "entrypoints": sorted(expected),
        "mission_cases": missions["plan"]["assessed"],
        "fault_runs": faults["summary"]["runs"],
        "paid_api_calls": 0,
        "checks": [
            "installed import",
            "typing marker",
            "entry points",
            "memory/retrieval",
            "mission baseline",
            "fault contracts",
            "persistent trace export",
        ],
    }
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))  # noqa: T201 - validation result


if __name__ == "__main__":
    main()
