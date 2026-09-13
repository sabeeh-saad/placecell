"""Compare retrieval channels against human-labeled destinations in recorded memories."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from placecell.errors import ValidationError
from placecell.providers.base import EmbeddingProvider
from placecell.retrieval import Recall, RetrievalMode
from placecell.store.base import Filter, VectorStore


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValidationError("evaluation query must be a nonempty string")
        if (
            not self.relevant_ids
            or any(not isinstance(i, str) or not i.strip() for i in self.relevant_ids)
            or len(set(self.relevant_ids)) != len(self.relevant_ids)
        ):
            raise ValidationError("relevant_ids must contain distinct, nonempty memory IDs")


def load_cases(path: str | Path) -> list[RetrievalCase]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValidationError("evaluation labels must be a nonempty JSON array")
    cases = []
    for row in data:
        if not isinstance(row, dict) or not isinstance(row.get("relevant_ids"), list):
            raise ValidationError("each label needs query and relevant_ids fields")
        cases.append(RetrievalCase(row.get("query", ""), tuple(row["relevant_ids"])))
    return cases


def evaluate_retrieval(
    store: VectorStore,
    embedder: EmbeddingProvider,
    cases: Sequence[RetrievalCase],
    *,
    k: int = 5,
    where: Filter | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Evaluate the same candidates, labels, clock and ranking policy across all three modes.

    The default clock is the latest retained observation, so archived recordings do not
    disappear through confidence underflow. Latency includes text encoding and retrieval;
    one warm-up query per mode excludes initial model loading and index synchronization.
    """
    if k < 1 or not cases:
        raise ValidationError("evaluation needs positive k and at least one labeled query")
    scope = where or Filter(role="episodic")
    counts = {"memories": 0, "image": 0, "caption": 0, "both": 0, "legacy": 0}
    last_seen = 0.0
    for batch in store.iter_query(scope, 256):
        for memory in batch:
            image, caption = memory.vector_for("image") is not None, memory.vector_for("caption") is not None
            counts["memories"] += 1
            counts["image"] += image
            counts["caption"] += caption
            counts["both"] += image and caption
            counts["legacy"] += memory.embedding_kind == "legacy"
            last_seen = max(last_seen, memory.last_seen)
    if counts["legacy"]:
        raise ValidationError("re-embed legacy memories before comparing channels; their vector modality is unknown")
    if not counts["image"]:
        raise ValidationError("evaluation needs image embeddings; re-embed recordings with a multimodal provider")
    for case in cases:
        for identity in case.relevant_ids:
            labeled = store.get(identity)
            if labeled is None or not scope.matches(labeled):
                raise ValidationError(f"labeled memory {identity!r} is absent or outside the evaluation filter")
    reference_time = last_seen if now is None else now
    if not math.isfinite(reference_time) or reference_time < 0:
        raise ValidationError("evaluation time must be finite and non-negative")
    recall = Recall(store, embedder, clock=lambda: reference_time)
    modes: dict[str, Any] = {}
    for mode in ("caption", "image", "combined"):
        selected: RetrievalMode = mode
        recall.similar(cases[0].query, k=k, where=scope, mode=selected)
        rows: list[dict[str, Any]] = []
        for case in cases:
            started = time.perf_counter()
            hits = recall.similar(case.query, k=k, where=scope, mode=selected)
            elapsed_ms = (time.perf_counter() - started) * 1000
            relevant = set(case.relevant_ids)
            ranks = [i for i, hit in enumerate(hits, 1) if hit.memory.id in relevant]
            rows.append(
                {
                    "query": case.query,
                    "relevant_ids": list(case.relevant_ids),
                    "hit": bool(ranks),
                    "recall_at_k": len(ranks) / len(relevant),
                    "reciprocal_rank_at_k": 1 / ranks[0] if ranks else 0.0,
                    "latency_ms": elapsed_ms,
                    "results": [
                        {
                            "id": hit.memory.id,
                            "score": hit.score,
                            "similarity": hit.similarity,
                            "image_similarity": hit.image_similarity,
                            "caption_similarity": hit.caption_similarity,
                        }
                        for hit in hits
                    ],
                }
            )
        modes[mode] = {
            "hit_rate_at_k": sum(r["hit"] for r in rows) / len(rows),
            "recall_at_k": sum(r["recall_at_k"] for r in rows) / len(rows),
            "mrr_at_k": sum(r["reciprocal_rank_at_k"] for r in rows) / len(rows),
            "mean_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
            "queries": rows,
        }
    return {
        "model": store.info.model,
        "collection": store.info.name,
        "k": k,
        "query_count": len(cases),
        "reference_time": reference_time,
        "filter": asdict(scope),
        "coverage": counts,
        "modes": modes,
        "combined_minus_caption_hit_rate": modes["combined"]["hit_rate_at_k"] - modes["caption"]["hit_rate_at_k"],
    }


def main(args: list[str] | None = None) -> None:
    from placecell.providers.clip import DEFAULT_CLIP_MODEL, ClipEmbedder
    from placecell.providers.gemini import DEFAULT_GEMINI_MODEL, GeminiEmbedder
    from placecell.store.lancedb_store import LanceDBStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--queries", required=True, help="JSON labels: query and relevant_ids")
    parser.add_argument("--output", required=True, help="New JSON report file (never overwrites an existing report)")
    parser.add_argument("--backend", choices=("gemini", "clip"), default="gemini")
    parser.add_argument("--model")
    parser.add_argument("--dimension", type=int, default=0)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-folder")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--robot-id")
    parser.add_argument("--map-id")
    parser.add_argument("--k", type=int, default=5)
    options = parser.parse_args(args)
    cases = load_cases(options.queries)
    embedder: EmbeddingProvider
    if options.backend == "gemini":
        embedder = GeminiEmbedder(
            options.model or DEFAULT_GEMINI_MODEL,
            api_key=os.environ.get(options.api_key_env),
            dimension=options.dimension or 768,
        )
    else:
        embedder = ClipEmbedder(
            options.model or DEFAULT_CLIP_MODEL,
            revision=options.revision,
            device=options.device,
            cache_folder=options.cache_folder,
            local_files_only=options.local_files_only,
        )
        if options.dimension and options.dimension != embedder.dimension:
            raise ValidationError("dimension does not match CLIP checkpoint")
    output = Path(options.output)
    if output.exists():
        raise ValidationError(f"report already exists: {output}")
    store = LanceDBStore.open(Path(options.db_path).expanduser(), options.collection)
    try:
        report = evaluate_retrieval(
            store,
            embedder,
            cases,
            k=options.k,
            where=Filter(role="episodic", robot_id=options.robot_id, map_id=options.map_id),
        )
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    finally:
        store.close()


if __name__ == "__main__":  # pragma: no cover
    main()
