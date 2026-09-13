"""Ingestion: observations in, memories out.

Sampling keeps a small checkpoint per stream. Provider work runs outside state transactions;
persistence and contradiction updates commit together for each observation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from placecell.errors import ModelMismatchError, ValidationError
from placecell.lifecycle import EvidenceRemover, Reinforcer, remove_local_file, remove_unreferenced
from placecell.memory import Evidence, Memory, Pose, Vector, memory_id
from placecell.observer import Observer
from placecell.providers.base import Captioner, EmbeddingProvider
from placecell.store.base import VectorStore
from placecell.store.jobs import WorkJournal


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
    max_interval_s: float = 60.0
    """Refresh a stationary view at least this often."""

    def __post_init__(self) -> None:
        values = (self.min_interval_s, self.min_travel_m, self.min_turn_rad, self.max_interval_s)
        if not all(math.isfinite(v) for v in values) or min(values) < 0:
            raise ValidationError("segmentation thresholds must be finite and non-negative")
        if self.max_interval_s <= 0 or self.max_interval_s < self.min_interval_s:
            raise ValidationError("segmentation thresholds must not be negative")


class Segmenter:
    """Keeps an observation only if enough time passed and the robot moved or turned.

    Stationary scenes refresh at the maximum interval. State is kept per
    robot and camera, so streams can be interleaved.
    """

    def __init__(self, policy: SegmentationPolicy | None = None) -> None:
        self._policy = policy or SegmentationPolicy()
        self._last: dict[tuple[str, str], Observation] = {}

    def eligible(self, robot_id: str, camera_id: str, timestamp: float, pose: Pose) -> bool:
        """Cheap admission check before encoding or writing an image; does not advance state."""
        last = self._last.get((robot_id, camera_id))
        if last is None or not last.pose.same_frame(pose):
            return True
        p = self._policy
        elapsed = timestamp - last.timestamp
        if elapsed < p.min_interval_s:
            return False
        return (
            elapsed >= p.max_interval_s
            or pose.distance_to(last.pose) >= p.min_travel_m
            or pose.heading_difference(last.pose) >= p.min_turn_rad
        )

    def accept(self, observation: Observation) -> bool:
        if not self.eligible(observation.robot_id, observation.camera_id, observation.timestamp, observation.pose):
            return False
        self._last[(observation.robot_id, observation.camera_id)] = observation
        return True

    def reset(self) -> None:
        self._last.clear()

    def checkpoint(self) -> dict[tuple[str, str], Observation]:
        """Capture stream positions before processing a batch that may need to be retried."""
        return self._last.copy()

    def restore(self, checkpoint: dict[tuple[str, str], Observation]) -> None:
        self._last = checkpoint.copy()


@dataclass(frozen=True, slots=True)
class IngestReport:
    received: int = 0
    accepted: int = 0
    inserted: int = 0
    merged: int = 0
    unsupported: int = 0
    """Observations no provider could embed: media the embedder rejects and no captioner to describe it."""
    unsupported_ids: tuple[str, ...] = field(default=())
    contradicted: int = 0
    """Memories superseded in this run because repeated visits no longer saw them."""


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
        observer: Observer | None = None,
        remover: EvidenceRemover | None = remove_local_file,
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
        self._reinforcer = reinforcer or Reinforcer(store, remover=remover)
        self._batch_size = batch_size
        self._observer = observer
        self._store = store
        self._remover = remover

    @property
    def jobs(self) -> WorkJournal:
        return self._store.jobs

    def persisted(self, observation: Observation) -> bool:
        identity = memory_id(observation.robot_id, observation.camera_id, observation.timestamp)
        return self._reinforcer.find_observation(identity) is not None

    def ingest(self, observations: Iterable[Observation], *, preselected: bool = False) -> IngestReport:
        received = accepted = inserted = merged = contradicted = 0
        unsupported: list[str] = []
        batch: list[Observation] = []
        discarded: list[Observation] = []
        checkpoint = self._segmenter.checkpoint()

        def discard_rejected() -> None:
            pending = {o.evidence.uri for o in batch}
            self.discard(o for o in discarded if o.evidence.uri not in pending)
            discarded.clear()

        def flush() -> None:
            nonlocal inserted, merged, contradicted, checkpoint
            if not batch:
                return
            pending = []
            for observation in batch:
                identity = memory_id(observation.robot_id, observation.camera_id, observation.timestamp)
                if self._reinforcer.find_observation(identity) is None:
                    pending.append(observation)
                else:
                    # A retry may follow a successful write whose old keyframe has already been replaced.
                    merged += 1
                    discarded.append(observation)
            memories, rejected = self.embed(self.caption(pending)) if pending else ([], [])
            unsupported.extend(m.id for m in rejected)
            for m in memories:
                with self._store.transaction():
                    if self._reinforcer.find_observation(m.id) is not None:
                        merged += 1
                        self._discard_evidence([m.evidence] if m.evidence else [])
                        continue
                    stored, was_merged = self.persist(m)
                    merged += was_merged
                    inserted += not was_merged
                    if self._observer is not None:
                        contradicted += self._observer.observe(m, stored.id).superseded
            self._discard_evidence(m.evidence for m in rejected if m.evidence is not None)
            batch.clear()
            checkpoint = self._segmenter.checkpoint()
            discard_rejected()

        try:
            for obs in observations:
                received += 1
                if not preselected and not self.segment(obs):
                    discarded.append(obs)
                    if len(discarded) >= self._batch_size:
                        discard_rejected()
                    continue
                accepted += 1
                batch.append(obs)
                if len(batch) >= self._batch_size:
                    flush()
            flush()
            discard_rejected()
        except Exception:
            self._segmenter.restore(checkpoint)
            discard_rejected()
            raise
        return IngestReport(received, accepted, inserted, merged, len(unsupported), tuple(unsupported), contradicted)

    def discard(self, observations: Iterable[Observation]) -> None:
        """Release generated images from dropped observations, preserving any stored references."""
        self._discard_evidence(o.evidence for o in observations)

    def _discard_evidence(self, evidence: Iterable[Evidence]) -> None:
        if self._remover is not None:
            remove_unreferenced(self._store, (e for e in evidence if e.managed), self._remover)

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
        vectors: dict[int, Vector] = {}
        if media_rows:
            matrix = self._embedder.embed_media([memories[i].evidence for i in media_rows])  # type: ignore[misc]
            vectors.update(zip(media_rows, matrix, strict=True))
        if text_rows:
            matrix = self._embedder.embed_text([memories[i].caption for i in text_rows])
            vectors.update(zip(text_rows, matrix, strict=True))
        embedded = [memories[i].with_embedding(vectors[i], self._embedder.model_name) for i in sorted(vectors)]
        return embedded, rejected

    def persist(self, memory: Memory) -> tuple[Memory, bool]:
        return self._reinforcer.reinforce_or_insert(memory)
