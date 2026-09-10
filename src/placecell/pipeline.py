"""Ingestion: observations in, memories out.

Four stages, each a pure function over a batch: segment, caption, embed, persist. The
`Ingester` composes them in one process. Because the stages only exchange lists, the same
four can later run as separate workers behind queues without changing what they do.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from placecell.errors import ModelMismatchError, ValidationError
from placecell.lifecycle import Reinforcer
from placecell.memory import Evidence, Memory, Pose
from placecell.providers.base import Captioner, EmbeddingProvider
from placecell.store.base import VectorStore


@dataclass(frozen=True, slots=True)
class Observation:
    """What a source hands to the pipeline: one piece of evidence with where and when."""

    robot_id: str
    camera_id: str
    timestamp: float
    pose: Pose
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class SegmentationPolicy:
    min_interval_s: float = 2.0
    min_travel_m: float = 0.3
    min_turn_rad: float = 0.35

    def __post_init__(self) -> None:
        if min(self.min_interval_s, self.min_travel_m, self.min_turn_rad) < 0:
            raise ValidationError("segmentation thresholds must not be negative")


class Segmenter:
    """Keeps an observation only if enough time passed and the robot moved or turned.

    A robot standing still produces one memory, not one per frame. State is kept per
    robot and camera, so streams can be interleaved.
    """

    def __init__(self, policy: SegmentationPolicy | None = None) -> None:
        self._policy = policy or SegmentationPolicy()
        self._last: dict[tuple[str, str], Observation] = {}

    def accept(self, observation: Observation) -> bool:
        key = (observation.robot_id, observation.camera_id)
        last = self._last.get(key)
        if last is not None:
            p = self._policy
            elapsed = observation.timestamp - last.timestamp
            if elapsed < p.min_interval_s:
                return False
            moved = observation.pose.distance_to(last.pose) >= p.min_travel_m
            turned = observation.pose.heading_difference(last.pose) >= p.min_turn_rad
            if not (moved or turned):
                return False
        self._last[key] = observation
        return True

    def reset(self) -> None:
        self._last.clear()


@dataclass(frozen=True, slots=True)
class IngestReport:
    received: int = 0
    accepted: int = 0
    inserted: int = 0
    merged: int = 0
    unsupported: int = 0
    """Observations no provider could embed: media the embedder rejects and no captioner to describe it."""
    unsupported_ids: tuple[str, ...] = field(default=())


class Ingester:
    """Runs the four stages over batches of observations."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        captioner: Captioner | None = None,
        segmenter: Segmenter | None = None,
        reinforcer: Reinforcer | None = None,
        batch_size: int = 32,
    ) -> None:
        if embedder.model_name != store.info.model or embedder.dimension != store.info.dimension:
            raise ModelMismatchError(
                f"embedder {embedder.model_name!r}/{embedder.dimension} does not match "
                f"collection {store.info.model!r}/{store.info.dimension}"
            )
        if batch_size < 1:
            raise ValidationError("batch_size must be at least 1")
        self._embedder = embedder
        self._captioner = captioner
        self._segmenter = segmenter or Segmenter()
        self._reinforcer = reinforcer or Reinforcer(store)
        self._batch_size = batch_size

    def ingest(self, observations: Iterable[Observation]) -> IngestReport:
        received = accepted = inserted = merged = 0
        unsupported: list[str] = []
        batch: list[Observation] = []

        def flush() -> None:
            nonlocal inserted, merged
            if not batch:
                return
            memories, rejected = self.embed(self.caption(batch))
            unsupported.extend(m.id for m in rejected)
            for m in memories:
                _, was_merged = self.persist(m)
                merged += was_merged
                inserted += not was_merged
            batch.clear()

        for obs in observations:
            received += 1
            if not self.segment(obs):
                continue
            accepted += 1
            batch.append(obs)
            if len(batch) >= self._batch_size:
                flush()
        flush()
        return IngestReport(received, accepted, inserted, merged, len(unsupported), tuple(unsupported))

    # The stages. Each is usable on its own by a queue-based runner.

    def segment(self, observation: Observation) -> bool:
        return self._segmenter.accept(observation)

    def caption(self, batch: Sequence[Observation]) -> list[Memory]:
        """Turn observations into unembedded memories, describing the evidence if a captioner is set."""
        captions = self._captioner.caption([o.evidence for o in batch]) if self._captioner else [""] * len(batch)
        if len(captions) != len(batch):
            raise ValidationError("captioner returned a different number of captions than items")
        return [
            Memory.create(o.robot_id, o.camera_id, o.timestamp, o.pose, o.evidence, caption)
            for o, caption in zip(batch, captions, strict=True)
        ]

    def embed(self, memories: Sequence[Memory]) -> tuple[list[Memory], list[Memory]]:
        """Attach vectors. Media goes to the provider when it can take it, else its caption does.

        Returns (embedded, rejected). Rejected memories have media the provider cannot embed
        and no caption to fall back on.
        """
        caps = self._embedder.capabilities
        media_rows = [i for i, m in enumerate(memories) if m.evidence is not None and caps.supports(m.evidence)]
        text_rows = [i for i, m in enumerate(memories) if i not in set(media_rows) and m.caption and caps.text]
        rejected = [m for i, m in enumerate(memories) if i not in set(media_rows) | set(text_rows)]
        vectors: dict[int, object] = {}
        if media_rows:
            matrix = self._embedder.embed_media([memories[i].evidence for i in media_rows])  # type: ignore[misc]
            vectors.update(zip(media_rows, matrix, strict=True))
        if text_rows:
            matrix = self._embedder.embed_text([memories[i].caption for i in text_rows])
            vectors.update(zip(text_rows, matrix, strict=True))
        embedded = [
            memories[i].with_embedding(vectors[i], self._embedder.model_name)  # type: ignore[arg-type]
            for i in sorted(vectors)
        ]
        return embedded, rejected

    def persist(self, memory: Memory) -> tuple[Memory, bool]:
        return self._reinforcer.reinforce_or_insert(memory)
