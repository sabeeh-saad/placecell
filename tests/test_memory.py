from __future__ import annotations

import math

import numpy as np
import pytest

from placecell import Evidence, EvidenceKind, Memory, Pose, Sighting, memory_id
from placecell.errors import FrameMismatchError, ValidationError


def test_pose_distance_and_heading() -> None:
    a, b = Pose(0, 0, 0.1), Pose(3, 4, -0.1)
    assert a.distance_to(b) == pytest.approx(5.0)
    assert a.heading_difference(b) == pytest.approx(0.2)
    assert Pose(0, 0, math.pi - 0.05).heading_difference(Pose(0, 0, -math.pi + 0.05)) == pytest.approx(0.1)


def test_pose_rejects_other_frames_and_bad_values() -> None:
    with pytest.raises(FrameMismatchError):
        Pose(0, 0).distance_to(Pose(0, 0, frame_id="odom"))
    with pytest.raises(FrameMismatchError):
        Pose(0, 0, map_id="a").distance_to(Pose(0, 0, map_id="b"))
    with pytest.raises(ValidationError):
        Pose(float("nan"), 0)
    with pytest.raises(ValidationError):
        Pose(0, 0, frame_id="")


def test_evidence_rules() -> None:
    assert Evidence(EvidenceKind.FRAME, "a.jpg").duration_s == 0
    assert Evidence(EvidenceKind.CLIP, "a.mp4", duration_s=4.0).kind is EvidenceKind.CLIP
    with pytest.raises(ValidationError):
        Evidence(EvidenceKind.CLIP, "a.mp4")
    with pytest.raises(ValidationError):
        Evidence(EvidenceKind.FRAME, "a.jpg", duration_s=1.0)
    with pytest.raises(ValidationError):
        Evidence(EvidenceKind.FRAME, "")


def test_memory_id_is_deterministic_and_validated() -> None:
    assert memory_id("r1", "front", 1700000000.1234) == "r1:front:1700000000123"
    assert memory_id("r1", "front", 1.0) == memory_id("r1", "front", 1.0)
    for bad in [("", "c"), ("a:b", "c"), ("r", "")]:
        with pytest.raises(ValidationError):
            memory_id(bad[0], bad[1], 1.0)
    with pytest.raises(ValidationError):
        memory_id("r", "c", -1.0)


def test_memory_defaults_and_embedding_contract() -> None:
    m = Memory.create("r1", "front", 10.0, Pose(1, 2), caption="a chair")
    assert m.last_seen == 10.0 and m.confidence == 1.0 and m.observations == 1 and m.embedding is None
    e = m.with_embedding(np.arange(4, dtype=np.float64), "m")
    assert e.embedding is not None and e.embedding.dtype == np.float32 and not e.embedding.flags.writeable
    assert e.model == "m"
    with pytest.raises(ValidationError):
        m.with_embedding(np.zeros((2, 2)), "m")
    with pytest.raises(ValidationError):
        m.with_embedding(np.array([1.0, float("inf")]), "m")
    with pytest.raises(ValidationError):
        m.with_embedding(np.ones(3), "")
    with pytest.raises(ValidationError):
        Memory("id", "r", "c", 5.0, Pose(0, 0), model="m")  # model without vector


@pytest.mark.parametrize("bad", [{"confidence": 1.5}, {"observations": 0}, {"last_seen": 1.0}])
def test_memory_rejects_out_of_range_fields(bad: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Memory("id", "r", "c", 5.0, Pose(0, 0), **bad)  # type: ignore[arg-type]


def test_effective_confidence_halves_per_half_life() -> None:
    m = Memory.create("r", "c", 0.0, Pose(0, 0))
    assert m.effective_confidence(0.0, 10.0) == 1.0
    assert m.effective_confidence(10.0, 10.0) == pytest.approx(0.5)
    assert m.effective_confidence(30.0, 10.0) == pytest.approx(0.125)
    assert m.effective_confidence(-5.0, 10.0) == 1.0  # never grows into the past
    with pytest.raises(ValidationError):
        m.effective_confidence(1.0, 0.0)


def test_memories_compare_without_looking_at_vectors() -> None:
    a = Memory.create("r", "c", 1.0, Pose(0, 0)).with_embedding(np.ones(3), "m")
    b = Memory.create("r", "c", 1.0, Pose(0, 0)).with_embedding(np.zeros(3), "m")
    assert a == b


def test_sightings_and_supersession_timestamps_are_validated() -> None:
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            Sighting("r:c:1", bad)
        with pytest.raises(ValidationError):
            Memory("id", "r", "c", 1.0, Pose(0, 0), superseded_at=bad)
