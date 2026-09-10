from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import pytest

from placecell import EvidenceKind, InMemoryStore, Pose
from placecell.errors import ValidationError
from placecell.pipeline import Ingester
from placecell.providers import HashingEmbedder
from placecell.retrieval import RankedMemory
from placecell.ros2.bridge import (
    KeyframeWriter,
    ObservationBuilder,
    pose_from_transform,
    stamp_to_seconds,
    yaw_from_quaternion,
)
from placecell.ros2.node import IngestWorker, answer_payload, build_embedder, build_store
from tests.conftest import FakeCaptioner, embedded


def test_quaternion_yaw_and_transform_pose() -> None:
    half = math.sin(math.pi / 4)
    assert yaw_from_quaternion(0, 0, half, half) == pytest.approx(math.pi / 2)
    assert yaw_from_quaternion(0, 0, 0, 1) == 0.0
    assert yaw_from_quaternion(0, 0, 1, 0) == pytest.approx(math.pi)
    pose = pose_from_transform(1.5, -2.0, 0, 0, half, half, "map", "office")
    assert (pose.x, pose.y, pose.frame_id, pose.map_id) == (1.5, -2.0, "map", "office")
    assert pose.yaw == pytest.approx(math.pi / 2)
    assert stamp_to_seconds(1700000000, 250_000_000) == pytest.approx(1700000000.25)


def test_keyframe_writer_handles_jpeg_and_raw_encodings(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = KeyframeWriter(tmp_path / "frames", jpeg_quality=70)
    jpeg = writer.write_jpeg("front", 12.5, b"\xff\xd8\xff")
    assert jpeg.kind is EvidenceKind.FRAME and jpeg.uri.endswith("front_12500.jpg") and len(jpeg.digest) == 64
    h, w = 6, 8
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # pure red in RGB
    raw = writer.write_raw("front", 13.0, h, w, "rgb8", w * 3, rgb.tobytes())
    decoded = cv2.imread(raw.uri)
    assert decoded.shape == (h, w, 3) and decoded[0, 0].tolist()[2] > 200 and decoded[0, 0].tolist()[0] < 50
    padded_step = w * 3 + 4
    padded = np.zeros((h, padded_step), dtype=np.uint8)
    padded[:, : w * 3] = rgb.reshape(h, w * 3)
    assert Path(writer.write_raw("front", 14.0, h, w, "rgb8", padded_step, padded.tobytes()).uri).is_file()
    mono = writer.write_raw("front", 15.0, h, w, "mono8", w, np.full((h, w), 90, dtype=np.uint8).tobytes())
    assert cv2.imread(mono.uri, cv2.IMREAD_GRAYSCALE).shape == (h, w)
    bgra = writer.write_raw("front", 16.0, h, w, "bgra8", w * 4, np.zeros((h, w, 4), dtype=np.uint8).tobytes())
    assert cv2.imread(bgra.uri).shape == (h, w, 3)
    with pytest.raises(ValidationError):
        writer.write_raw("front", 17.0, h, w, "16UC1", w * 2, bytes(h * w * 2))
    with pytest.raises(ValidationError):
        writer.write_raw("front", 17.0, h, w, "rgb8", w * 3, bytes(10))
    with pytest.raises(ValidationError):
        writer.write_jpeg("front", 18.0, b"")
    with pytest.raises(ValidationError):
        KeyframeWriter(tmp_path, jpeg_quality=0)


def test_observation_builder(tmp_path: Path) -> None:
    builder = ObservationBuilder("r1", "front", KeyframeWriter(tmp_path))
    obs = builder.from_compressed(20.0, "rgb8; jpeg compressed bgr8", b"\xff\xd8", Pose(1, 2))
    assert builder.from_compressed(20.5, "jpeg", b"\xff\xd8", Pose(1, 2)).evidence.uri.endswith("front_20500.jpg")
    assert obs.robot_id == "r1" and obs.camera_id == "front" and obs.timestamp == 20.0 and obs.pose == Pose(1, 2)
    with pytest.raises(ValidationError):
        builder.from_compressed(21.0, "png", b"x", Pose(0, 0))
    with pytest.raises(ValidationError):
        ObservationBuilder("", "front", KeyframeWriter(tmp_path))


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append("I " + msg)

    def warning(self, msg: str) -> None:
        self.lines.append("W " + msg)

    def error(self, msg: str) -> None:
        self.lines.append("E " + msg)


def test_ingest_worker_batches_in_the_background(tmp_path: Path, hashing: HashingEmbedder) -> None:
    from placecell import CollectionInfo, Observation

    store = InMemoryStore(CollectionInfo("c", hashing.model_name, hashing.dimension))
    ingester = Ingester(hashing, store, captioner=FakeCaptioner("a door"))
    log = _Log()
    worker = IngestWorker(ingester, threading.Lock(), batch_size=4, max_queue=2, log=log)
    writer = KeyframeWriter(tmp_path)
    for i in range(3):  # queue holds two, the third is dropped before the worker starts
        evidence = writer.write_jpeg("front", float(i), b"\xff\xd8")
        worker.submit(Observation("r1", "front", float(i * 5), Pose(i * 2.0, 0), evidence))
    assert worker.dropped == 1 and any(line.startswith("W ingest queue full") for line in log.lines)
    worker.start()
    deadline = threading.Event()
    for _ in range(50):
        if store.count() == 2:
            break
        deadline.wait(0.1)
    worker.stop()
    assert store.count() == 2
    assert any(line.startswith("I ingested 2/2") for line in log.lines)


def test_answer_payload_and_factories(hashing: HashingEmbedder, tmp_path: Path) -> None:
    ranked = RankedMemory(embedded(hashing, "a door", t=5, x=1, y=2), confidence=0.5, similarity=0.9)
    payload = json.loads(answer_payload("where?", "at the door", True, [ranked]))
    assert payload["answer"] == "at the door" and payload["grounded"] is True
    assert payload["evidence"][0] == {
        "id": "r1:front:5000",
        "x": 1.0,
        "y": 2.0,
        "yaw": 0.0,
        "time": 5.0,
        "caption": "a door",
        "confidence": 0.5,
    }
    offline = build_embedder("https://x/v1", "", None, 0)
    assert isinstance(offline, HashingEmbedder)
    online = build_embedder("https://x/v1", "text-embedding-3-small", "k", 256)
    assert online.model_name == "text-embedding-3-small" and online.dimension == 256
    assert isinstance(build_store("", "c", offline), InMemoryStore)
    pytest.importorskip("lancedb")
    persistent = build_store(str(tmp_path / "db"), "c", offline)
    assert persistent.info.model == offline.model_name and persistent.count() == 0
