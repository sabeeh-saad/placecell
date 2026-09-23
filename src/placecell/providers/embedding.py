"""Shared embedding stage for ingestion, migration and evidence refinement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from placecell.memory import SCHEMA_VERSION, EvidenceKind, Memory, Vector
from placecell.providers.base import EmbeddingProvider, normalise_rows


def embed_memories(
    memories: Sequence[Memory], embedder: EmbeddingProvider, *, include_captions: bool = True
) -> tuple[list[Memory], list[Memory]]:
    """Keep media and caption vectors separately; media remains the lifecycle comparison vector."""
    caps = embedder.capabilities
    media_rows = [
        i
        for i, m in enumerate(memories)
        if m.role == "episodic" and m.evidence is not None and caps.supports(m.evidence)
    ]
    text_rows = [i for i, m in enumerate(memories) if include_captions and m.caption.strip() and caps.text]
    media: dict[int, Vector] = {}
    captions: dict[int, Vector] = {}
    if media_rows:
        vectors = embedder.embed_media([memories[i].evidence for i in media_rows])  # type: ignore[misc]
        media.update(zip(media_rows, normalise_rows(vectors, len(media_rows), embedder.dimension), strict=True))
    if text_rows:
        vectors = embedder.embed_text([memories[i].caption for i in text_rows])
        captions.update(zip(text_rows, normalise_rows(vectors, len(text_rows), embedder.dimension), strict=True))
    accepted, rejected = [], []
    for i, memory in enumerate(memories):
        if i in media:
            assert memory.evidence is not None
            kind = "image" if memory.evidence.kind is EvidenceKind.FRAME else "video"
            updated = memory.with_embedding(media[i], embedder.model_name, kind=kind)
            accepted.append(replace(updated, caption_embedding=captions.get(i), schema_version=SCHEMA_VERSION))
        elif i in captions:
            accepted.append(
                replace(
                    memory.with_embedding(captions[i], embedder.model_name, kind="caption"),
                    schema_version=SCHEMA_VERSION,
                )
            )
        else:
            rejected.append(memory)
    return accepted, rejected
