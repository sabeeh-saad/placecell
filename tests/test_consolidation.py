from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from placecell import ChatMessage, ChatReply, Curator, Evidence, EvidenceKind, Filter, InMemoryStore, Recall
from placecell.consolidation import ChatSummarizer, ConsolidationPolicy, Consolidator, Summarizer
from placecell.errors import ModelMismatchError, ProviderError, ValidationError
from placecell.lifecycle import remove_local_file
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
from tests.conftest import DIM, embedded


class JoinSummarizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def summarize(self, captions: Sequence[str]) -> str:
        self.calls.append(list(captions))
        return f"summary of {len(captions)}: {captions[0]}"


def test_consolidator_folds_a_cluster_into_a_summary(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    printer = [
        embedded(hashing, "a printer on a table", t=float(i), x=0.3 + 0.1 * i, y=0.2, observations=2) for i in range(5)
    ]
    chair = [embedded(hashing, "a grey chair", t=100.0 + i, x=0.5, y=0.5, camera="back") for i in range(2)]
    far = [embedded(hashing, "a printer on a table", t=200.0 + i, x=9, y=9) for i in range(5)]
    store.upsert(printer + chair + far)
    summarizer = JoinSummarizer()
    assert isinstance(summarizer, Summarizer)
    consolidator = Consolidator(
        store, hashing, summarizer, ConsolidationPolicy(cell_m=2.0, min_group=5, min_similarity=0.9)
    )
    report = consolidator.run()
    assert (report.scanned, report.clusters, report.summaries, report.folded) == (12, 3, 2, 10)
    summaries = [m for m in store.query(EVERYTHING) if m.role == "summary"]
    assert len(summaries) == 2
    near = next(m for m in summaries if m.pose.x < 5)
    assert (
        near.caption == "summary of 5: a printer on a table" and near.observations == 10 and near.camera_id == "summary"
    )
    assert near.pose.x == pytest.approx(0.5) and near.timestamp == 0.0 and near.last_seen == 4.0
    assert near.evidence == printer[-1].evidence  # anchor: equal observations, latest sighting
    assert all(store.get(m.id).consolidated_into == near.id for m in printer)  # type: ignore[union-attr]
    assert store.get(chair[0].id).consolidated_into == ""  # type: ignore[union-attr]
    # idempotent: folded members are not folded again, small clusters stay
    assert consolidator.run() == consolidator.run().__class__(scanned=2, clusters=1)
    # the summary is retrievable like any other memory
    top = Recall(store, hashing, clock=lambda: 300.0).similar(near.caption, k=1)[0]
    assert top.memory.role == "summary" and top.memory.id in {m.id for m in summaries}


def test_consolidator_skips_captionless_clusters_and_validates(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    rows = [embedded(hashing, "", t=float(i), x=0) for i in range(5)]
    for i, m in enumerate(rows):  # captionless memories embedded from media in a real system
        rows[i] = m.with_embedding(hashing.embed_text(["same picture"])[0], hashing.model_name)
    store.upsert(rows)
    report = Consolidator(store, hashing, JoinSummarizer(), ConsolidationPolicy(min_group=5)).run()
    assert report.clusters == 1 and report.summaries == 0
    with pytest.raises(ModelMismatchError):
        Consolidator(store, HashingEmbedder(DIM * 2), JoinSummarizer())
    with pytest.raises(ValidationError):
        ConsolidationPolicy(min_group=1)


def test_simultaneous_camera_clusters_keep_distinct_summaries(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    rows = [embedded(hashing, "printer", t=t, camera="front", x=0) for t in (100, 200)] + [
        embedded(hashing, "chair", t=t, camera="back", x=4) for t in (100, 200)
    ]
    store.upsert(rows)
    consolidator = Consolidator(store, hashing, JoinSummarizer(), ConsolidationPolicy(min_group=2))
    assert consolidator.run().summaries == 2
    summaries = [m for m in store.query() if m.role == "summary"]
    assert len(summaries) == 2
    for row in rows:
        folded = store.get(row.id)
        assert folded is not None
        summary = store.get(folded.consolidated_into)
        assert summary is not None and summary.pose.x == row.pose.x and row.caption in summary.caption
    assert consolidator.run().summaries == 0
    # Identity depends on the members, including their original camera, regardless of order.
    a = consolidator._summarise(rows[:2], ["printer"])
    b = consolidator._summarise(list(reversed(rows[:2])), ["printer"])
    assert a.id == b.id


def test_summary_keeps_its_image_after_members_are_deleted(
    store: InMemoryStore,
    hashing: HashingEmbedder,
    tmp_path: Path,
) -> None:
    path = tmp_path / "anchor.jpg"
    path.write_bytes(b"image")
    rows = [embedded(hashing, "printer", t=t, evidence=Evidence(EvidenceKind.FRAME, str(path))) for t in (100, 200)]
    store.upsert(rows)
    Consolidator(store, hashing, JoinSummarizer(), ConsolidationPolicy(min_group=2)).run()
    curator = Curator(store, remover=remove_local_file)
    assert curator.forget(Filter(camera_id="front")) == 2
    assert path.exists() and store.query(EVERYTHING)[0].superseded
    assert store.query() == []
    assert curator.forget(Filter(camera_id="summary", include_superseded=True)) == 1
    assert not path.exists()


def test_chat_summarizer() -> None:
    class Chat:
        def __init__(self, text: str | None) -> None:
            self.text = text
            self.messages: list[ChatMessage] = []

        def complete(self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]) -> ChatReply:
            self.messages = list(messages)
            assert tools == []
            return ChatReply(self.text)

    chat = Chat("  A printer\n on a table. ")
    assert ChatSummarizer(chat).summarize(["printer", "printer on table"]) == "A printer on a table."
    assert chat.messages[0].role == "system" and "- printer on table" in (chat.messages[1].content or "")
    with pytest.raises(ProviderError):
        ChatSummarizer(Chat("")).summarize(["x"])
    with pytest.raises(ValidationError):
        ChatSummarizer(chat).summarize([])
    with pytest.raises(ValidationError):
        ChatSummarizer(chat, prompt=" ")
