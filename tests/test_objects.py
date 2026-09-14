from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from placecell import CollectionInfo, Evidence, EvidenceKind, Ingester, Observation, Pose
from placecell.depth import Box, DepthSnapshot
from placecell.errors import ValidationError
from placecell.navigation import DestinationResolver, parse_movement
from placecell.object_types import Detection
from placecell.objects import ObjectPolicy, ObjectRecall, ObjectTracker
from placecell.providers.base import Capabilities, normalise_rows
from placecell.retrieval import Recall
from placecell.store.state import StateStore
from placecell.verification import SceneVerdict

CENTER = Box(0.45, 0.45, 0.55, 0.55)
LEFT = Box(0.10, 0.45, 0.20, 0.55)
RIGHT = Box(0.80, 0.45, 0.90, 0.55)
SCOPE = {"robot_id": "r1", "camera_id": "front", "frame_id": "map", "map_id": "office-v1"}


class PixelEmbedder:
    model_name = "pixels-v1"
    dimension = 3
    capabilities = Capabilities(text=True, image=True)

    def embed_text(self, texts):
        return np.array([[0, 0, 1] if "blue" in t else [1, 0, 0] for t in texts], dtype=np.float32)

    def embed_media(self, items):
        rows = []
        for item in items:
            with Image.open(item.uri) as image:
                rows.append(np.asarray(image.convert("RGB")).mean(axis=(0, 1)))
        return normalise_rows(rows, len(items), 3)


class Detector:
    def __init__(self):
        self.detections = [Detection("printer", "red printer", CENTER)]
        self.absence = True
        self.checks = 0
        self.calls = 0

    def detect(self, _):
        self.calls += 1
        return self.detections

    def absent(self, *_):
        self.checks += 1
        return self.absence


@pytest.fixture
def setup(tmp_path):
    embedder = PixelEmbedder()
    store = StateStore(CollectionInfo("objects", embedder.model_name, 3), tmp_path / "state.sqlite3")
    detector = Detector()
    tracker = ObjectTracker(store, embedder, detector)
    yield store, embedder, detector, tracker
    store.close()


def observation(tmp_path, timestamp=1000, boxes=(CENTER,), colors=("red",), *, depth=True, background=5):
    image = Image.new("RGB", (100, 100), "black")
    draw = ImageDraw.Draw(image)
    array = np.full((100, 100), background, dtype=np.float32)
    for box, color in zip(boxes, colors, strict=True):
        bounds = (int(box.left * 100), int(box.top * 100), int(box.right * 100), int(box.bottom * 100))
        draw.rectangle(bounds, fill=color)
        array[bounds[1] : bounds[3], bounds[0] : bounds[2]] = 2
    path = tmp_path / f"frame-{timestamp}.png"
    image.save(path)
    snapshot = (
        DepthSnapshot.capture(array, (100, 100, 50, 50), np.eye(4), position_error_m=0.02, angular_error_rad=0)
        if depth
        else None
    )
    return Observation(
        "r1",
        "front",
        timestamp,
        Pose(0, 0, map_id="office-v1"),
        Evidence(EvidenceKind.FRAME, str(path), managed=True),
        True,
        snapshot,
    )


def ingest(tracker, obs):
    tracker.commit(tracker.prepare(obs))
    return tracker.store.objects.records(**SCOPE)


def test_revisit_identity_and_bounded_views_survive_restart(setup, tmp_path):
    store, embedder, _, tracker = setup
    first = ingest(tracker, observation(tmp_path))[0]
    for i in range(1, 7):
        current = ingest(tracker, observation(tmp_path, 1000 + i * 601))[0]
    assert current.id == first.id
    assert current.last_seen == 4606
    assert current.position.z == 2
    assert len(store.objects.views(first.id)) == 4
    assert [e.kind for e in store.objects.history(first.id)] == ["created"]
    with Image.open(io.BytesIO(store.objects.views(first.id)[0].crop_png)) as crop:
        assert crop.size == (11, 11) or crop.size == (10, 11)
        assert crop.getpixel((2, 2)) == (255, 0, 0)
    reopened = StateStore(store.info, tmp_path / "state.sqlite3")
    assert reopened.objects.get(first.id) == current
    results = ObjectRecall(reopened, embedder, clock=lambda: 4606).similar("printer", **SCOPE)
    assert results[0].object.id == first.id
    assert results[0].view.memory.timestamp == 4606
    reopened.close()


def test_unique_move_requires_empty_old_location(setup, tmp_path):
    store, _, detector, tracker = setup
    before = ingest(tracker, observation(tmp_path))[0]
    detector.detections = [Detection("printer", "red printer", RIGHT)]
    after = ingest(tracker, observation(tmp_path, 2000, (RIGHT,)))[0]
    assert after.id == before.id
    assert after.position.x > before.position.x + 0.6
    assert detector.checks == 1
    assert [e.kind for e in store.objects.history(after.id)] == ["created", "moved"]


def test_occlusion_does_not_prove_move(setup, tmp_path):
    _, _, detector, tracker = setup
    before = ingest(tracker, observation(tmp_path))[0]
    detector.detections = [Detection("printer", "red printer", RIGHT)]
    records = ingest(tracker, observation(tmp_path, 2000, (RIGHT,), background=1))
    assert len(records) == 2
    assert next(r for r in records if r.id == before.id).misses == 0
    assert detector.checks == 0


def test_two_identical_instances_remain_separate(setup, tmp_path):
    _, _, detector, tracker = setup
    detector.detections = [Detection("printer", "red printer", LEFT), Detection("printer", "red printer", RIGHT)]
    first = ingest(tracker, observation(tmp_path, boxes=(LEFT, RIGHT), colors=("red", "red")))
    second = ingest(tracker, observation(tmp_path, 2000, (LEFT, RIGHT), ("red", "red")))
    assert len(second) == 2
    assert {r.id for r in first} == {r.id for r in second}
    assert all(r.status == "present" for r in second)


def test_association_ties_are_non_navigable_hypotheses(setup, tmp_path):
    _, _, detector, tracker = setup
    first = ingest(tracker, observation(tmp_path, depth=False))[0]
    boxes = (Box(0.44, 0.45, 0.54, 0.55), Box(0.46, 0.45, 0.56, 0.55))
    detector.detections = [Detection("printer", "red printer", box) for box in boxes]
    second = ingest(tracker, observation(tmp_path, 2000, boxes, ("red", "red"), depth=False))
    assert len(second) == 3
    assert sum(r.status == "ambiguous" for r in second) == 2
    assert next(r for r in second if r.id == first.id).last_seen == 1000


@pytest.mark.parametrize("background,absence", [(1, True), (0, True), (5, False)])
def test_occluded_unknown_or_visually_uncertain_never_count_missing(setup, tmp_path, background, absence):
    _, _, detector, tracker = setup
    before = ingest(tracker, observation(tmp_path))[0]
    detector.detections = []
    detector.absence = absence
    for timestamp in (2000, 2700, 3400):
        after = ingest(tracker, observation(tmp_path, timestamp, (), (), background=background))[0]
    assert after == before


def test_missing_requires_separated_visits_and_returns_on_reappearance(setup, tmp_path):
    store, _, detector, tracker = setup
    before = ingest(tracker, observation(tmp_path))[0]
    detector.detections = []
    for timestamp in (2000, 2001, 2600, 3200):
        after = ingest(tracker, observation(tmp_path, timestamp, (), ()))[0]
    assert after.status == "missing" and after.misses == 3
    assert after.last_seen == before.last_seen
    assert detector.checks == 3
    detector.detections = [Detection("printer", "red printer", CENTER)]
    returned = ingest(tracker, observation(tmp_path, 4000))[0]
    assert returned.id == before.id and returned.status == "present" and returned.misses == 0
    assert store.objects.history(before.id)[-1].kind == "returned"


def test_scene_and_objects_commit_atomically_and_replay_skips_providers(setup, tmp_path, monkeypatch):
    store, embedder, detector, tracker = setup
    ingester = Ingester(embedder, store, objects=tracker)
    obs = observation(tmp_path)
    original = store.objects.save

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("disk failure")

    monkeypatch.setattr(store.objects, "save", fail)
    with pytest.raises(RuntimeError):
        ingester.ingest([obs], preselected=True)
    assert store.count() == 0 and store.objects.count() == 0
    monkeypatch.setattr(store.objects, "save", original)
    ingester.ingest([obs], preselected=True)
    assert store.count() == 1 and store.objects.count() == 1
    calls = detector.calls
    ingester.ingest([obs], preselected=True)
    assert detector.calls == calls


def test_prepared_update_rejects_concurrent_change(setup, tmp_path):
    store, _, _, tracker = setup
    obj = ingest(tracker, observation(tmp_path))[0]
    prepared = tracker.prepare(observation(tmp_path, 2000))
    store.objects.delete(obj.id)
    with pytest.raises(ValidationError, match="retry"):
        tracker.commit(prepared)
    assert store.objects.count() == 0


def test_shared_frame_cleanup_waits_for_last_object_reference(setup, tmp_path):
    store, embedder, _, tracker = setup
    obs = observation(tmp_path)
    Ingester(embedder, store, objects=tracker).ingest([obs], preselected=True)
    obj = store.objects.records(**SCOPE)[0]
    store.delete(m.id for m in store.query())
    removed = []
    store.drain_cleanup(removed.append)
    assert removed == []
    store.objects.delete(obj.id)
    store.drain_cleanup(removed.append)
    assert removed == [obs.evidence]


def test_depth_is_durable_with_queued_observation(setup, tmp_path):
    store, _, _, _ = setup
    obs = observation(tmp_path)
    store.jobs.enqueue(obs, 10)
    queued = store.jobs.pending(1)[0].observation
    assert queued == obs
    np.testing.assert_array_equal(queued.depth.array(), obs.depth.array())


def test_capacity_and_retention(setup, tmp_path):
    store, embedder, detector, _ = setup
    tracker = ObjectTracker(store, embedder, detector, ObjectPolicy(max_objects=1))
    obj = ingest(tracker, observation(tmp_path))[0]
    detector.detections = [Detection("cabinet", "blue cabinet", CENTER)]
    with pytest.raises(ValidationError, match="capacity"):
        tracker.prepare(observation(tmp_path, 2000, colors=("blue",)))
    assert store.objects.get(obj.id) == obj
    assert store.objects.prune(1500) == 1


class Matched:
    def verify(self, *args):
        return SceneVerdict("matched", "visible")


def resolver(store, embedder, clock=2000):
    return DestinationResolver(
        store,
        Recall(store, embedder, clock=lambda: clock),
        robot_id="r1",
        camera_id="front",
        map_id="office-v1",
        verifier=Matched(),
        clock=lambda: clock,
        objects=ObjectRecall(store, embedder, clock=lambda: clock),
    )


def test_navigation_uses_latest_view_and_rechecks_identity(setup, tmp_path):
    store, embedder, detector, tracker = setup
    before = ingest(tracker, observation(tmp_path))[0]
    detector.detections = [Detection("printer", "red printer", RIGHT)]
    ingest(tracker, replace(observation(tmp_path, 2000, (RIGHT,)), pose=Pose(0.3, 0, map_id="office-v1")))
    lookup = resolver(store, embedder)
    result = lookup.resolve(parse_movement("go to printer"))
    assert result.state == "resolved"
    destination = result.choices[0]
    assert destination.object_id == before.id
    assert destination.pose.x == 0.3
    assert lookup.current(destination)
    store.objects.delete(before.id)
    assert not lookup.current(destination)


def test_navigation_keeps_two_objects_from_same_viewpoint_separate(setup, tmp_path):
    store, embedder, detector, tracker = setup
    detector.detections = [Detection("printer", "red printer", LEFT), Detection("printer", "red printer", RIGHT)]
    ingest(tracker, observation(tmp_path, 2000, (LEFT, RIGHT), ("red", "red")))
    result = resolver(store, embedder).resolve(parse_movement("go to printer"))
    assert result.state == "ambiguous" and len(result.choices) == 2


def test_missing_object_blocks_scene_fallback(setup, tmp_path):
    store, embedder, detector, tracker = setup
    Ingester(embedder, store, objects=tracker).ingest([observation(tmp_path)], preselected=True)
    detector.detections = []
    ingest(tracker, observation(tmp_path, 2000, (), ()))
    assert resolver(store, embedder).resolve(parse_movement("go to printer")).state == "not_found"


def test_reembedding_preserves_object_ids_crops_positions_and_changes(setup, tmp_path):
    from placecell import reembed

    store, embedder, detector, tracker = setup
    obj = ingest(tracker, observation(tmp_path))[0]
    detector.detections = [Detection("printer", "red printer", RIGHT)]
    ingest(tracker, observation(tmp_path, 2000, (RIGHT,)))
    target = StateStore(CollectionInfo("new", embedder.model_name, 3))
    report = reembed(store, target, embedder)
    assert report.objects_written == 1
    assert target.objects.get(obj.id) == store.objects.get(obj.id)
    assert target.objects.history(obj.id) == store.objects.history(obj.id)
    assert [v.crop_png for v in target.objects.views(obj.id)] == [v.crop_png for v in store.objects.views(obj.id)]
    for before, after in zip(store.objects.views(obj.id), target.objects.views(obj.id), strict=True):
        assert before.memory.same_embeddings(after.memory)
    reembed(store, target, embedder)
    assert len(target.objects.history(obj.id)) == 2
    target.close()


def test_scan_interval_is_durable_and_out_of_order_frames_cannot_create_duplicates(setup, tmp_path):
    store, _, detector, tracker = setup
    obj = ingest(tracker, observation(tmp_path))[0]
    ingest(tracker, observation(tmp_path, 1001))
    ingest(tracker, observation(tmp_path, 900))
    assert detector.calls == 1
    assert store.objects.get(obj.id) == obj


def test_scope_isolation_and_rgb_only_updates(setup, tmp_path):
    store, embedder, _, tracker = setup
    first = ingest(tracker, observation(tmp_path, depth=False))[0]
    ingest(tracker, observation(tmp_path, 2000, depth=False))
    assert store.objects.count() == 1
    assert store.objects.get(first.id).last_seen == 2000
    foreign = replace(observation(tmp_path, 3000), robot_id="r2")
    ingest(tracker, foreign)
    assert store.objects.count() == 2
    hits = ObjectRecall(store, embedder, clock=lambda: 3000).similar("printer", **SCOPE)
    assert len(hits) == 1 and hits[0].object.id == first.id


def test_event_retention_and_reference_cleanup_on_view_rotation(setup, tmp_path):
    store, _, _, tracker = setup
    obj = ingest(tracker, observation(tmp_path))[0]
    for timestamp in (2000, 3000, 4000, 5000):
        ingest(tracker, observation(tmp_path, timestamp))
    removed = []
    store.drain_cleanup(removed.append)
    assert len(removed) == 1
    assert removed[0].uri.endswith("frame-1000.png")
    latest = store.objects.get(obj.id)
    for i in range(40):
        store.objects.save(latest, event="reviewed", event_time=6000 + i)
    assert len(store.objects.history(obj.id)) == 32


def test_invalid_object_config_and_foreign_model_fail_before_provider(setup):
    store, _, detector, _ = setup
    from placecell import ObjectPosition
    from placecell.errors import ModelMismatchError
    from placecell.providers import HashingEmbedder

    with pytest.raises(ModelMismatchError):
        ObjectTracker(store, HashingEmbedder(), detector)
    with pytest.raises(ModelMismatchError):
        ObjectRecall(store, HashingEmbedder())
    with pytest.raises(ValidationError):
        ObjectPolicy(max_views=0)
    with pytest.raises(ValidationError):
        ObjectPolicy(association_margin=2)
    with pytest.raises(ValidationError):
        ObjectPolicy(visit_interval_s=float("nan"))
    with pytest.raises(ValidationError):
        ObjectPosition(0, 0, 2, 0, 1)
    with pytest.raises(ValidationError):
        ObjectPosition(0, 0, float("nan"), 0.1, 1)


def test_lancedb_object_state_and_cleanup_survive_reopen(tmp_path):
    pytest.importorskip("lancedb")
    from placecell.store.lancedb_store import LanceDBStore

    embedder = PixelEmbedder()
    info = CollectionInfo("objects", embedder.model_name, 3)
    store = LanceDBStore(tmp_path / "db", info)
    tracker = ObjectTracker(store, embedder, Detector())
    Ingester(embedder, store, objects=tracker).ingest([observation(tmp_path)], preselected=True)
    obj = store.objects.records(**SCOPE)[0]
    store.close()
    reopened = LanceDBStore(tmp_path / "db", info)
    assert reopened.objects.get(obj.id) == obj
    reopened.delete(m.id for m in reopened.query())
    removed = []
    reopened.drain_cleanup(removed.append)
    assert removed == []
    assert reopened.objects.views(obj.id)[0].image_url().startswith("data:image/png;base64,")
    reopened.close()


def test_explicit_forget_removes_object_crops_and_releases_shared_frames(setup, tmp_path):
    from placecell import Curator, Filter

    store, embedder, _, tracker = setup
    obs = observation(tmp_path)
    Ingester(embedder, store, objects=tracker).ingest([obs], preselected=True)
    assert Curator(store).forget(Filter(robot_id="another")) == 0
    removed = []
    assert Curator(store, remover=removed.append).forget(Filter(robot_id="r1")) == 2
    assert store.objects.count() == 0 and store.count() == 0
    assert removed == [obs.evidence]


def test_schema_six_upgrade_preserves_vectors_and_adds_object_ownership(tmp_path):
    import json

    from placecell import SCHEMA_VERSION
    from placecell.store.lancedb_store import LanceDBStore

    embedder = PixelEmbedder()
    info = CollectionInfo("old", embedder.model_name, 3)
    store = LanceDBStore(tmp_path / "db", info)
    Ingester(embedder, store).ingest([observation(tmp_path)], preselected=True)
    before = store.query()[0]
    store.close()
    metadata = tmp_path / "db" / "old.collection.json"
    data = json.loads(metadata.read_text())
    data["schema_version"] = 6
    metadata.write_text(json.dumps(data))
    upgraded = LanceDBStore(tmp_path / "db", info)
    assert upgraded.info.schema_version == SCHEMA_VERSION == 7
    assert upgraded.get(before.id).same_embeddings(before)
    tracker = ObjectTracker(upgraded, embedder, Detector())
    ingest(tracker, observation(tmp_path, 2000))
    upgraded.rebuild_index()
    assert upgraded.objects.count() == 1
    assert upgraded.get(before.id).same_embeddings(before)
    upgraded.close()


def test_near_threshold_lookalike_still_prevents_false_identity_match(setup, tmp_path):
    store, embedder, detector, tracker = setup
    boxes = (Box(0.35, 0.45, 0.45, 0.55), Box(0.55, 0.45, 0.65, 0.55))
    detector.detections = [Detection("printer", "red printer", box) for box in boxes]
    vectors = np.array([[0.851, np.sqrt(1 - 0.851**2), 0], [0.849, 0, np.sqrt(1 - 0.849**2)]], dtype=np.float32)
    embedder.embed_media = lambda _: vectors
    before = ingest(tracker, observation(tmp_path, boxes=boxes, colors=("red", "red")))
    detector.detections = [Detection("printer", "red printer", CENTER)]
    embedder.embed_media = lambda _: np.array([[1, 0, 0]], dtype=np.float32)
    after = ingest(tracker, observation(tmp_path, 2000))
    assert len(after) == 3
    assert sum(record.status == "ambiguous" for record in after) == 1
    assert all(store.objects.get(record.id).last_seen == 1000 for record in before)
