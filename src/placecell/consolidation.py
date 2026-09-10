"""Consolidation: many sightings of one place become one summary memory.

Memories are grouped by map cell, clustered by similarity, and every cluster large enough is
summarised into one sentence by a chat model. The summary is stored as a memory of role
"summary" with the members' combined observation count; the members stay, marked with the
summary's id, so time questions and evidence still work. Runs are idempotent: folded members
are never folded again.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from placecell.chat import ChatMessage, ChatModel
from placecell.errors import ModelMismatchError, ProviderError, ValidationError
from placecell.memory import Memory, Pose, Vector, memory_id
from placecell.providers.base import EmbeddingProvider
from placecell.store.base import Filter, VectorStore


@runtime_checkable
class Summarizer(Protocol):
    def summarize(self, captions: Sequence[str]) -> str: ...


SUMMARY_PROMPT = (
    "These are descriptions a mobile robot recorded at one place on different occasions. "
    "Write one plain sentence stating what is there. Mention only what recurs; ignore people "
    "passing through and one-off details. No preamble."
)


class ChatSummarizer:
    def __init__(self, model: ChatModel, prompt: str = SUMMARY_PROMPT) -> None:
        if not prompt.strip():
            raise ValidationError("prompt must not be empty")
        self._model = model
        self._prompt = prompt

    def summarize(self, captions: Sequence[str]) -> str:
        if not captions:
            raise ValidationError("nothing to summarise")
        listing = "\n".join(f"- {c}" for c in captions)
        reply = self._model.complete([ChatMessage("system", self._prompt), ChatMessage("user", listing)], [])
        text = " ".join((reply.content or "").split())
        if not text:
            raise ProviderError("the model returned an empty summary")
        return text


@dataclass(frozen=True, slots=True)
class ConsolidationPolicy:
    cell_m: float = 2.0
    """Side of the map grid cell that defines 'one place'."""
    min_group: int = 5
    """Clusters smaller than this are left as episodic memories."""
    min_similarity: float = 0.5
    """A memory joins a cluster if it is at least this similar to the cluster's centre."""
    max_captions: int = 30
    """How many captions are shown to the summariser per cluster."""

    def __post_init__(self) -> None:
        if self.cell_m <= 0 or self.min_group < 2 or self.max_captions < 1 or not (-1 <= self.min_similarity <= 1):
            raise ValidationError("consolidation policy out of range")


@dataclass(frozen=True, slots=True)
class ConsolidationReport:
    scanned: int = 0
    clusters: int = 0
    summaries: int = 0
    folded: int = 0


class Consolidator:
    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        summarizer: Summarizer,
        policy: ConsolidationPolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if embedder.model_name != store.info.model:
            raise ModelMismatchError(f"embedder {embedder.model_name!r} does not match collection {store.info.model!r}")
        self._store = store
        self._embedder = embedder
        self._summarizer = summarizer
        self._policy = policy or ConsolidationPolicy()
        self._clock = clock

    def run(self, scope: Filter | None = None) -> ConsolidationReport:
        p = self._policy
        rows = [m for m in self._store.query(scope) if m.role == "episodic" and not m.consolidated_into]
        cells: dict[tuple[str, str, int, int], list[Memory]] = {}
        for m in rows:
            key = (m.pose.frame_id, m.pose.map_id, math.floor(m.pose.x / p.cell_m), math.floor(m.pose.y / p.cell_m))
            cells.setdefault(key, []).append(m)
        clusters = summaries = folded = 0
        for members in cells.values():
            for cluster in _cluster(members, p.min_similarity):
                clusters += 1
                if len(cluster) < p.min_group:
                    continue
                captions = [m.caption for m in cluster if m.caption][: p.max_captions]
                if not captions:
                    continue
                summary = self._summarise(cluster, captions)
                self._store.upsert([summary, *(replace(m, consolidated_into=summary.id) for m in cluster)])
                summaries += 1
                folded += len(cluster)
        return ConsolidationReport(len(rows), clusters, summaries, folded)

    def _summarise(self, cluster: Sequence[Memory], captions: Sequence[str]) -> Memory:
        text = self._summarizer.summarize(captions)
        vector = self._embedder.embed_text([text])[0]
        anchor = max(cluster, key=lambda m: (m.observations, m.last_seen))
        xs = [m.pose.x for m in cluster]
        ys = [m.pose.y for m in cluster]
        yaw = math.atan2(sum(math.sin(m.pose.yaw) for m in cluster), sum(math.cos(m.pose.yaw) for m in cluster))
        pose = Pose(sum(xs) / len(xs), sum(ys) / len(ys), yaw, anchor.pose.frame_id, anchor.pose.map_id)
        newest = max(m.timestamp for m in cluster)
        return Memory(
            id=memory_id(anchor.robot_id, "summary", newest),
            robot_id=anchor.robot_id,
            camera_id="summary",
            timestamp=min(m.timestamp for m in cluster),
            pose=pose,
            evidence=anchor.evidence,
            caption=text,
            embedding=vector,
            model=self._embedder.model_name,
            confidence=1.0,
            observations=sum(m.observations for m in cluster),
            last_seen=max(m.last_seen for m in cluster),
            role="summary",
        )


def _cluster(members: Sequence[Memory], min_similarity: float) -> list[list[Memory]]:
    """Greedy clustering by cosine to a running centre, strongest memories first."""
    ordered = sorted(members, key=lambda m: (-m.observations, m.timestamp))
    clusters: list[list[Memory]] = []
    centres: list[Vector] = []
    for m in ordered:
        v = m.embedding
        if v is None:  # pragma: no cover - stores refuse unembedded memories
            continue
        norm = float(np.linalg.norm(v))
        unit = v / norm if norm else v
        for i, centre in enumerate(centres):
            if float(unit @ centre) >= min_similarity:
                clusters[i].append(m)
                total = centre * (len(clusters[i]) - 1) + unit
                centres[i] = total / (float(np.linalg.norm(total)) or 1.0)
                break
        else:
            clusters.append([m])
            centres.append(unit)
    return clusters
