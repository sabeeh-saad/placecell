from __future__ import annotations

from pathlib import Path

import pytest

from placecell import Curator, InMemoryStore, Recall, RetentionPolicy
from placecell.corrections import (
    Correction,
    CorrectionLog,
    InMemoryCorrectionLog,
    JsonlCorrectionLog,
    Verdicts,
    correction_now,
)
from placecell.errors import ValidationError
from placecell.providers import HashingEmbedder
from placecell.store.base import EVERYTHING
from tests.conftest import embedded


def test_correction_validation_and_weights() -> None:
    assert Correction("m", "right").verdict == "right"
    assert correction_now("m", "wrong", "q").timestamp > 0
    for bad in [("", "right"), ("m", "maybe")]:
        with pytest.raises(ValidationError):
            Correction(*bad)
    with pytest.raises(ValidationError):
        Correction("m", "right", timestamp=-1)
    assert Verdicts().weight == 1.0 and Verdicts(wrong=2).weight == 0.25 and Verdicts(right=3, wrong=1).weight == 1.0


def test_logs_count_verdicts_and_jsonl_persists(tmp_path: Path) -> None:
    log = InMemoryCorrectionLog()
    assert isinstance(log, CorrectionLog)
    log.record(Correction("a", "wrong", "where?"))
    log.record(Correction("a", "wrong"))
    log.record(Correction("a", "right"))
    log.record(Correction("b", "right"))
    assert log.verdicts(["a", "b", "zzz"]) == {"a": Verdicts(1, 2), "b": Verdicts(1, 0)}
    path = tmp_path / "log" / "corrections.jsonl"
    disk = JsonlCorrectionLog(path)
    disk.record(Correction("a", "wrong", "q", "not a chair", 5.0))
    disk.record(Correction("b", "right"))
    reloaded = JsonlCorrectionLog(path)
    assert len(reloaded) == 2 and reloaded.verdicts(["a"]) == {"a": Verdicts(0, 1)}
    assert path.read_text().count("\n") == 2


def test_wrong_verdicts_push_memories_down_in_retrieval(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    a = embedded(hashing, "a fire extinguisher", t=1, x=0)
    b = embedded(hashing, "a fire extinguisher", t=2, x=5, camera="back")
    store.upsert([a, b])
    log = InMemoryCorrectionLog()
    log.record(Correction(a.id, "wrong"))
    recall = Recall(store, hashing, clock=lambda: 2.0, corrections=log)
    ranked = recall.similar("fire extinguisher", k=2)
    assert [r.memory.id for r in ranked] == [b.id, a.id]
    assert ranked[1].confidence == pytest.approx(0.5 * ranked[0].confidence, rel=1e-3)


def test_curator_discredits_memories_judged_wrong_repeatedly(store: InMemoryStore, hashing: HashingEmbedder) -> None:
    a = embedded(hashing, "a", t=0)
    b = embedded(hashing, "b", t=1, x=3)
    store.upsert([a, b])
    log = InMemoryCorrectionLog()
    for _ in range(3):
        log.record(Correction(a.id, "wrong"))
    log.record(Correction(b.id, "wrong"))
    log.record(Correction(b.id, "wrong"))
    log.record(Correction(b.id, "right"))
    curator = Curator(store, RetentionPolicy(wrong_verdicts_to_supersede=3), corrections=log, clock=lambda: 10.0)
    report = curator.run()
    assert report.discredited == 1 and report.removed == 0
    assert store.count() == 1 and store.count(EVERYTHING) == 2
    assert store.get(a.id).superseded and not store.get(b.id).superseded  # type: ignore[union-attr]
    assert curator.run().discredited == 0
    with pytest.raises(ValidationError):
        RetentionPolicy(wrong_verdicts_to_supersede=0)
