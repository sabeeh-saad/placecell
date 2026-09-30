"""Store scaling benchmark: ingest, filtered search, recall and reads beside a writer.

The data is synthetic so runs are reproducible: clustered unit vectors, poses on a
100 x 100 m map per robot and sightings spread over 30 days. Each size is loaded in bulk
and then measured:

- ingest: observations through `Ingester` (reinforcement, contradiction and admission, as
  the ROS node runs it), about 40 % of them revisits that merge into an existing memory,
  and how long each of its write transactions holds the store;
- search: p50/p95 latency per filter, outside and inside a store transaction;
- recall@10 of the store's search against exact brute force over a snapshot;
- reads: search latency while another thread ingests.

Run from the repository root:

    PYTHONPATH=src python benchmarks/store_scaling.py --backend lancedb --sizes 10k,100k
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from placecell import CollectionInfo, Evidence, EvidenceKind, Filter, InMemoryStore, Memory, Pose, Sighting
from placecell.memory import Matrix, Vector
from placecell.observer import Observer
from placecell.pipeline import Ingester, IngestReport, Observation
from placecell.providers import Capabilities, normalise_rows
from placecell.store import StoreLimits, VectorStore
from placecell.store.base import EVERYTHING

DAY = 86400.0
START = 1_767_225_600.0
"""2026-01-01 UTC; the loaded history covers the 30 days after it."""
SPAN = 30 * DAY
MAP_SIZE = 100.0
SPREAD = 0.75
"""Noise around a cluster centre: members of one cluster have cosine similarity near 0.64."""
REVISIT_NOISE = 0.15
"""A revisit stays above the 0.9 merge similarity of its memory."""
RADIUS = 5.0
K = 10
FILTERS = ("unfiltered", "robot", "near", "time")


def _unit(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / norms, dtype=np.float32)


class LookupEmbedder:
    """Stands in for an image model by returning the vector registered for each evidence URI."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self.capabilities = Capabilities(text=True, image=True, video=False)
        self.vectors: dict[str, Vector] = {}

    @property
    def model_name(self) -> str:
        return f"bench-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, texts: Sequence[str]) -> Matrix:
        return normalise_rows(np.ones((len(texts), self._dimension)), len(texts), self._dimension)

    def embed_media(self, items: Sequence[Evidence]) -> Matrix:
        return np.stack([self.vectors[item.uri] for item in items])


class Scene:
    """Deterministic memories, observations and queries for one seed."""

    def __init__(self, dimension: int, robots: int, clusters: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.dimension, self.robots = dimension, robots
        self.centres = _unit(self.rng.standard_normal((clusters, dimension), dtype=np.float32))
        self.clock = START + SPAN
        self.loaded: list[tuple[int, float, float, float]] = []
        self.loaded_vectors: list[Matrix] = []

    def vectors(self, n: int) -> Matrix:
        labels = self.rng.integers(0, len(self.centres), n)
        noise = _unit(self.rng.standard_normal((n, self.dimension), dtype=np.float32))
        return _unit(self.centres[labels] + SPREAD * noise)

    def memories(self, n: int, model: str, batch: int = 1000) -> Iterator[list[Memory]]:
        step = 0.9 * SPAN / n
        for first in range(0, n, batch):
            count = min(batch, n - first)
            vectors = self.vectors(count)
            places = self.rng.uniform(0, MAP_SIZE, (count, 2))
            yaws = self.rng.uniform(-np.pi, np.pi, count)
            self.loaded_vectors.append(vectors)
            out = []
            for j in range(count):
                i = first + j
                robot = i % self.robots
                t = START + i * step + float(self.rng.uniform(0, 0.5 * step))
                later = sorted(t + float(d) for d in self.rng.uniform(1.0, 3 * DAY, int(self.rng.integers(0, 3))))
                pose = Pose(float(places[j, 0]), float(places[j, 1]), float(yaws[j]), "map", f"map-r{robot}")
                self.loaded.append((robot, pose.x, pose.y, pose.yaw))
                memory = Memory.create(
                    f"r{robot}", "front", t, pose, Evidence(EvidenceKind.FRAME, f"bench/r{robot}/{i}.jpg")
                ).with_embedding(vectors[j], model, kind="image")
                sightings = (*memory.sightings, *(Sighting(f"{memory.id}#{k}", s) for k, s in enumerate(later)))
                out.append(
                    replace(
                        memory,
                        sightings=sightings,
                        last_seen=max(t, *later) if later else t,
                        observations=len(sightings),
                        localization_checked=True,
                    )
                )
            yield out

    def observations(self, n: int, embedder: LookupEmbedder, revisits: float = 0.4) -> list[Observation]:
        """New observations after the loaded history; a share of them revisit a loaded memory."""
        fresh = self.vectors(n)
        known = np.concatenate(self.loaded_vectors) if self.loaded_vectors else fresh[:0]
        out = []
        for j in range(n):
            self.clock += 2.0
            if len(self.loaded) and self.rng.random() < revisits:
                index = int(self.rng.integers(0, len(self.loaded)))
                robot, x, y, yaw = self.loaded[index]
                x, y = (float(v) for v in np.array([x, y]) + self.rng.uniform(-0.2, 0.2, 2))
                yaw += float(self.rng.uniform(-0.1, 0.1))
                noise = self.rng.standard_normal(self.dimension).astype(np.float32)
                vector = known[index] + REVISIT_NOISE * noise / np.linalg.norm(noise)
            else:
                robot = int(self.rng.integers(0, self.robots))
                x, y = (float(v) for v in self.rng.uniform(0, MAP_SIZE, 2))
                yaw = float(self.rng.uniform(-np.pi, np.pi))
                vector = fresh[j]
            uri = f"bench/r{robot}/live-{self.clock:.3f}.jpg"
            embedder.vectors[uri] = np.asarray(vector / np.linalg.norm(vector), dtype=np.float32)
            pose = Pose(x, y, yaw, "map", f"map-r{robot}")
            evidence = Evidence(EvidenceKind.FRAME, uri)
            out.append(Observation(f"r{robot}", "front", self.clock, pose, evidence, localization_checked=True))
        return out

    def queries(self, n: int) -> dict[str, list[tuple[Vector, Filter]]]:
        out: dict[str, list[tuple[Vector, Filter]]] = {name: [] for name in FILTERS}
        for name in FILTERS:
            for vector in self.vectors(n):
                robot = int(self.rng.integers(0, self.robots))
                if name == "unfiltered":
                    where = Filter()
                elif name == "robot":
                    where = Filter(robot_id=f"r{robot}")
                elif name == "near":
                    x, y = (float(v) for v in self.rng.uniform(0, MAP_SIZE, 2))
                    where = Filter(near=Pose(x, y, 0.0, "map", f"map-r{robot}"), radius=RADIUS)
                else:
                    t = START + float(self.rng.uniform(0, SPAN - DAY))
                    where = Filter(time_from=t, time_to=t + DAY)
                out[name].append((vector, where))
        return out


@dataclass
class Snapshot:
    """Every memory's vector and filter fields, for exact brute-force search."""

    ids: list[str]
    matrix: Matrix
    superseded: NDArray[np.bool_]
    robots: NDArray[np.str_]
    maps: NDArray[np.str_]
    frames: NDArray[np.str_]
    xy: NDArray[np.float64]
    times: NDArray[np.float64]
    """Sighting times, padded with NaN."""

    @classmethod
    def of(cls, store: VectorStore) -> Snapshot:
        memories = [m for batch in store.iter_query(EVERYTHING, batch_size=2048) for m in batch]
        width = max((len(m.sighting_times) for m in memories), default=1)
        times = np.full((len(memories), width), np.nan)
        for row, m in enumerate(memories):
            times[row, : len(m.sighting_times)] = m.sighting_times
        vectors = [m.embedding for m in memories if m.embedding is not None]
        return cls(
            [m.id for m in memories],
            _unit(np.stack(vectors)) if vectors else np.zeros((0, 1), dtype=np.float32),
            np.array([m.superseded for m in memories], dtype=bool),
            np.array([m.robot_id for m in memories]),
            np.array([m.pose.map_id for m in memories]),
            np.array([m.pose.frame_id for m in memories]),
            np.array([(m.pose.x, m.pose.y) for m in memories], dtype=np.float64).reshape(-1, 2),
            times,
        )

    def search(self, vector: Vector, where: Filter) -> list[str]:
        mask = np.ones(len(self.ids), dtype=bool)
        if not where.include_superseded:
            mask &= ~self.superseded
        if where.robot_id is not None:
            mask &= self.robots == where.robot_id
        if where.near is not None and where.radius is not None:
            p = where.near
            mask &= (self.frames == p.frame_id) & (self.maps == p.map_id)
            mask &= np.hypot(self.xy[:, 0] - p.x, self.xy[:, 1] - p.y) <= where.radius
        if where.time_from is not None or where.time_to is not None:
            low = -np.inf if where.time_from is None else where.time_from
            high = np.inf if where.time_to is None else where.time_to
            with np.errstate(invalid="ignore"):
                mask &= ((self.times >= low) & (self.times < high)).any(axis=1)
        rows = np.flatnonzero(mask)
        scores = self.matrix[rows] @ (vector / np.linalg.norm(vector))
        return [self.ids[i] for i in rows[np.argsort(-scores, kind="stable")[:K]]]


def _ms(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        return {"p50_ms": float("nan"), "p95_ms": float("nan")}
    values = np.asarray(samples) * 1000
    return {"p50_ms": round(float(np.percentile(values, 50)), 3), "p95_ms": round(float(np.percentile(values, 95)), 3)}


def open_store(backend: str, info: CollectionInfo, limits: StoreLimits, directory: Path) -> VectorStore:
    if backend == "memory":
        return InMemoryStore(info, limits=limits)
    from placecell.store.lancedb_store import LanceDBStore

    return LanceDBStore(directory, info, limits=limits)


@contextmanager
def held_transactions(store: VectorStore) -> Iterator[list[float]]:
    """Time how long each outermost store transaction opened inside the block is held."""
    held: list[float] = []
    transaction = store.transaction
    local = threading.local()

    @contextmanager
    def timed() -> Iterator[None]:
        depth = getattr(local, "depth", 0)
        with transaction():
            local.depth = depth + 1
            started = time.perf_counter()
            try:
                yield
            finally:
                local.depth = depth
                if not depth:
                    held.append(time.perf_counter() - started)

    store.transaction = timed  # type: ignore[method-assign]
    try:
        yield held
    finally:
        del store.transaction


def ingest(store: VectorStore, embedder: LookupEmbedder, observations: list[Observation]) -> tuple[float, IngestReport]:
    """Ingest like the ROS node: batches of eight with contradiction checks and no file removal."""
    ingester = Ingester(embedder, store, observer=Observer(store), remover=None, batch_size=8)
    started = time.perf_counter()
    report = ingester.ingest(observations, preselected=True)
    return len(observations) / (time.perf_counter() - started), report


def searches(
    store: VectorStore, queries: list[tuple[Vector, Filter]], *, inside: bool
) -> tuple[list[float], list[list[str]]]:
    latencies, results = [], []
    for vector, where in queries:
        if inside:
            with store.transaction():
                started = time.perf_counter()
                hits = store.search(vector, K, where)
                latencies.append(time.perf_counter() - started)
        else:
            started = time.perf_counter()
            hits = store.search(vector, K, where)
            latencies.append(time.perf_counter() - started)
        results.append([hit.memory.id for hit in hits])
    return latencies, results


def concurrent_reads(
    store: VectorStore, embedder: LookupEmbedder, observations: list[Observation], queries: list[tuple[Vector, Filter]]
) -> dict[str, Any]:
    """Search continuously while another thread ingests, and time both."""
    started, finished = threading.Event(), threading.Event()
    failures: list[BaseException] = []
    write: list[float] = []

    def writer() -> None:
        started.set()
        try:
            write.append(ingest(store, embedder, observations)[0])
        except BaseException as e:  # pragma: no cover - reported below
            failures.append(e)
        finally:
            finished.set()

    thread = threading.Thread(target=writer, name="bench-writer")
    thread.start()
    started.wait()
    latencies = []
    for vector, where in itertools.cycle(queries):
        begin = time.perf_counter()
        store.search(vector, K, where)
        latencies.append(time.perf_counter() - begin)
        if finished.is_set():
            break
    thread.join()
    if failures:
        raise failures[0]
    return {"reads": len(latencies), **_ms(latencies), "writes_per_s": round(write[0], 1)}


def run_size(options: argparse.Namespace, size: int, directory: Path) -> dict[str, Any]:
    scene = Scene(options.dim, options.robots, options.clusters, options.seed)
    embedder = LookupEmbedder(options.dim)
    info = CollectionInfo("bench", embedder.model_name, options.dim)
    limits = StoreLimits(max_memories=2 * size + 4 * (options.ingest + options.writes) + 1000)
    store = open_store(options.backend, info, limits, directory)
    try:
        result: dict[str, Any] = {"size": size}
        started = time.perf_counter()
        for batch in scene.memories(size, embedder.model_name):
            store.upsert(batch)
        result["load_s"] = round(time.perf_counter() - started, 2)
        maintain = getattr(store, "maintain", None)
        if maintain is not None:
            started = time.perf_counter()
            maintain()
            result["maintain_s"] = round(time.perf_counter() - started, 2)
        with held_transactions(store) as held:
            rate, report = ingest(store, embedder, scene.observations(options.ingest, embedder))
        result["ingest"] = {
            "observations": options.ingest,
            "per_s": round(rate, 1),
            "inserted": report.inserted,
            "merged": report.merged,
            "contradicted": report.contradicted,
            "transaction": _ms(held),
        }
        queries = scene.queries(options.queries)
        for vector, where in (queries[name][0] for name in FILTERS):
            store.search(vector, K, where)
        timings: dict[str, Any] = {}
        answers: dict[str, list[list[str]]] = {}
        idle: list[float] = []
        for name in FILTERS:
            outside, answers[name] = searches(store, queries[name], inside=False)
            inside, _ = searches(store, queries[name], inside=True)
            idle.extend(outside)
            timings[name] = {"outside": _ms(outside), "inside": _ms(inside)}
        snapshot = Snapshot.of(store)
        for name in FILTERS:
            found = expected = short = 0
            for (vector, where), answer in zip(queries[name], answers[name], strict=True):
                exact = snapshot.search(vector, where)
                found += len(set(answer) & set(exact))
                expected += len(exact)
                short += len(answer) < len(exact)
            timings[name]["recall_at_10"] = round(found / expected, 4) if expected else None
            timings[name]["short_results"] = short
        result["search"] = timings
        mixed = [q for group in zip(*(queries[name] for name in FILTERS), strict=True) for q in group]
        result["reads"] = {
            "idle": _ms(idle),
            "during_ingest": concurrent_reads(store, embedder, scene.observations(options.writes, embedder), mixed),
        }
        return result
    finally:
        store.close()


def table(report: dict[str, Any]) -> str:
    lines = [
        f"backend {report['backend']}, dim {report['dim']}, {report['robots']} robots, seed {report['seed']}",
        "",
        "| size | load s | maintain s | ingest/s | filter | out p50 | out p95 | in p50 | in p95 | recall@10 |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        head = [result["size"], result["load_s"], result.get("maintain_s", "-"), result["ingest"]["per_s"]]
        for name in FILTERS:
            s = result["search"][name]
            latencies = (f"{s[where][p]:.1f}" for where in ("outside", "inside") for p in ("p50_ms", "p95_ms"))
            recall = "-" if s["recall_at_10"] is None else f"{s['recall_at_10']:.3f}"
            lines.append("| " + " | ".join(str(cell) for cell in (*head, name, *latencies, recall)) + " |")
            head = [""] * 4
    lines += ["", "| size | ingest tx p50 | tx p95 | idle read p95 | busy reads | read p50 | read p95 | writes/s |"]
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in report["results"]:
        held = result["ingest"]["transaction"]
        idle, busy = result["reads"]["idle"], result["reads"]["during_ingest"]
        lines.append(
            f"| {result['size']} | {held['p50_ms']:.1f} | {held['p95_ms']:.1f} | {idle['p95_ms']:.1f} | "
            f"{busy['reads']} | {busy['p50_ms']:.1f} | {busy['p95_ms']:.1f} | {busy['writes_per_s']} |"
        )
    return "\n".join(lines) + "\n"


def _size(text: str) -> int:
    text = text.strip().lower()
    scale = {"k": 1000, "m": 1_000_000}.get(text[-1:], 1)
    return int(float(text.rstrip("km")) * scale)


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--backend", choices=("memory", "lancedb"), default="lancedb")
    parser.add_argument("--sizes", type=lambda v: [_size(s) for s in v.split(",")], default=[10_000, 100_000])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--queries", type=int, default=50, help="queries per filter type")
    parser.add_argument("--ingest", type=int, default=500, help="observations in the ingest measurement")
    parser.add_argument("--writes", type=int, default=300, help="observations ingested beside the readers")
    parser.add_argument("--output", type=Path, default=None, help="write the full report as JSON")
    parser.add_argument("--workdir", type=Path, default=None, help="where LanceDB collections are created")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = parse(argv)
    report: dict[str, Any] = {
        "backend": options.backend,
        "dim": options.dim,
        "robots": options.robots,
        "clusters": options.clusters,
        "seed": options.seed,
        "queries": options.queries,
        "results": [],
    }
    with tempfile.TemporaryDirectory(dir=options.workdir) as scratch:
        for size in options.sizes:
            report["results"].append(run_size(options, size, Path(scratch) / f"db-{size}"))
    if options.output is not None:
        options.output.write_text(json.dumps(report, indent=2) + "\n")
    sys.stdout.write(table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
