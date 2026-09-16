from dataclasses import replace

import numpy as np
import pytest

from placecell import CollectionInfo, InMemoryStore, ObjectArrivalPolicy, ObjectArrivalVerifier, ObjectTracker, Pose
from placecell.errors import ValidationError
from placecell.object_types import Detection
from placecell.providers.captioning import data_url
from placecell.verification import SceneVerdict
from tests.test_objects import LEFT, RIGHT, Detector, PixelEmbedder, ingest, observation


class Comparator:
    def __init__(self, result="matched"):
        self.result, self.calls = result, []
        self.after = lambda: None

    def compare(self, references, candidate):
        self.calls.append((references, candidate))
        self.after()
        return SceneVerdict(self.result, "Compared visible distinguishing details.")


def setup_arrival(tmp_path, *, initial=None):
    embedder, detector, comparator = PixelEmbedder(), Detector(), Comparator()
    store = InMemoryStore(CollectionInfo("arrival", embedder.model_name, 3))
    tracker = ObjectTracker(store, embedder, detector)
    obj = ingest(tracker, initial or observation(tmp_path))[0]
    now = [1100]
    verifier = ObjectArrivalVerifier(tracker, comparator, clock=lambda: now[0])
    return verifier, verifier.capture(obj.id), detector, comparator, now


def test_arrival_uses_saved_views_and_new_depth_without_writing_memory(tmp_path):
    verifier, reference, _, comparator, _ = setup_arrival(tmp_path)
    before = verifier.tracker.store.objects.generation
    verdict = verifier.verify(reference, observation(tmp_path, 1100))
    assert verdict.result == "matched" and verdict.position.z == 2
    assert comparator.calls[0][0] == tuple(v.crop_png for v in reference.views)
    assert verifier.tracker.store.objects.generation == before
    assert verifier.tracker.store.objects.get(reference.record.id).last_seen == 1000


@pytest.mark.parametrize("result", ["uncertain", "not_matched"])
def test_high_embedding_similarity_cannot_override_paired_image_check(tmp_path, result):
    verifier, reference, _, comparator, _ = setup_arrival(tmp_path)
    comparator.result = result
    verdict = verifier.verify(reference, observation(tmp_path, 1100))
    assert verdict.result == ("ambiguous" if result == "uncertain" else "unobserved")


def test_multiple_live_lookalikes_are_ambiguous_even_when_category_matches(tmp_path):
    verifier, reference, detector, comparator, _ = setup_arrival(tmp_path)
    detector.detections = [Detection("printer", "red printer", LEFT), Detection("printer", "red printer", RIGHT)]
    obs = observation(tmp_path, 1100, (LEFT, RIGHT), ("red", "red"))
    assert verifier.verify(reference, obs).result == "ambiguous"
    assert not comparator.calls


def test_a_known_lookalike_prevents_unique_match_even_if_only_one_is_detected(tmp_path):
    verifier, reference, detector, comparator, _ = setup_arrival(tmp_path)
    store = verifier.tracker.store
    view = reference.views[0]
    other = replace(reference.record, id="lookalike")
    store.objects.save(other, replace(view, object_id=other.id, memory=replace(view.memory, id="other-view")))
    reference = verifier.capture(reference.record.id)
    assert verifier.verify(reference, observation(tmp_path, 1100)).result == "ambiguous"
    assert not comparator.calls and detector.calls == 2


def test_move_requires_visible_empty_old_location_and_normal_ingestion_records_history(tmp_path):
    verifier, reference, detector, _, _ = setup_arrival(tmp_path)
    detector.detections = [Detection("printer", "red printer", RIGHT)]
    obs = observation(tmp_path, 1100, (RIGHT,))
    verdict = verifier.verify(reference, obs)
    assert verdict.result == "matched" and verdict.position.x > 0.6
    journal = verifier.tracker.store.objects
    assert journal.get(reference.record.id) == reference.record
    moved = ingest(verifier.tracker, obs)[0]
    assert moved.id == reference.record.id and moved.last_seen == 1100
    assert [e.kind for e in journal.history(moved.id)] == ["created", "moved"]
    ingest(verifier.tracker, obs)
    assert len(journal.history(moved.id)) == 2


def test_occlusion_cannot_establish_movement_or_disappearance(tmp_path):
    verifier, reference, detector, comparator, _ = setup_arrival(tmp_path)
    detector.detections = [Detection("printer", "red printer", RIGHT)]
    obs = observation(tmp_path, 1100, (RIGHT,), background=1)
    assert verifier.verify(reference, obs).result == "ambiguous"
    assert detector.checks == 0 and not comparator.calls
    detector.detections = []
    assert verifier.verify(reference, obs).result == "unobserved"
    assert verifier.tracker.store.objects.get(reference.record.id).misses == 0


def test_missing_verdict_is_an_observation_not_a_persistent_missing_state(tmp_path):
    verifier, reference, detector, _, _ = setup_arrival(tmp_path)
    detector.detections = []
    assert verifier.verify(reference, observation(tmp_path, 1100, (), ())).result == "missing"
    assert verifier.tracker.store.objects.get(reference.record.id).status == "present"


def test_wrong_appearance_and_uncertain_depth_do_not_verify(tmp_path):
    verifier, reference, _, comparator, _ = setup_arrival(tmp_path)
    assert verifier.verify(reference, observation(tmp_path, 1100, colors=("blue",))).result == "unobserved"
    assert not comparator.calls
    obs = observation(tmp_path, 1100)
    obs = replace(obs, depth=replace(obs.depth, position_error_m=1))
    assert verifier.verify(reference, obs).result == "unavailable"


def test_rgb_only_requires_same_viewpoint_and_stale_geometry_cannot_authorize_another(tmp_path):
    verifier, reference, _, _, now = setup_arrival(tmp_path)
    obs = observation(tmp_path, 1100, depth=False)
    assert verifier.verify(reference, obs).result == "matched"
    assert verifier.verify(reference, replace(obs, pose=Pose(0.5, 0, map_id="office-v1"))).result == "unavailable"
    now[0] = 1400
    old_depth = replace(observation(tmp_path, 1400), pose=Pose(0.5, 0, map_id="office-v1"))
    assert verifier.verify(reference, old_depth).result == "unavailable"


@pytest.mark.parametrize("change", ["stale", "future", "unlocalized", "camera", "map", "replay"])
def test_invalid_observations_are_rejected_before_provider_work(tmp_path, change):
    verifier, reference, detector, _, _ = setup_arrival(tmp_path)
    obs = observation(tmp_path, 1100)
    if change in {"stale", "future", "replay"}:
        obs = replace(obs, timestamp={"stale": 1090, "future": 1101, "replay": 1000}[change])
    elif change == "unlocalized":
        obs = replace(obs, localization_checked=False)
    elif change == "camera":
        obs = replace(obs, camera_id="rear")
    else:
        obs = replace(obs, pose=Pose(0, 0, map_id="changed"))
    with pytest.raises(ValidationError):
        verifier.verify(reference, obs)
    assert detector.calls == 1


def test_snapshot_survives_source_cleanup_and_keeps_predeparture_reference(tmp_path):
    from pathlib import Path

    verifier, reference, _, comparator, _ = setup_arrival(tmp_path)
    obs = observation(tmp_path, 1100)
    image = data_url(obs.evidence.uri)
    Path(obs.evidence.uri).unlink()
    assert verifier.verify_image(reference, obs, image, lambda: False).result == "matched"
    fresh = observation(tmp_path, 1100, colors=("blue",))
    blue = verifier.tracker.detect_views(fresh)[0]
    verifier.tracker.store.objects.save(
        replace(reference.record, revision=2, last_seen=1100), replace(blue, object_id=reference.record.id)
    )
    assert verifier.verify(reference, fresh).result == "unobserved"
    assert len(comparator.calls) == 1


@pytest.mark.parametrize("change", ["cancel", "delete"])
def test_results_are_rechecked_after_provider_work(tmp_path, change):
    verifier, reference, _, comparator, _ = setup_arrival(tmp_path)
    canceled = [False]
    comparator.after = lambda: (
        canceled.__setitem__(0, True)
        if change == "cancel"
        else verifier.tracker.store.objects.delete(reference.record.id)
    )
    with pytest.raises(ValidationError):
        verifier.verify(reference, observation(tmp_path, 1100), lambda: canceled[0])


def test_invalid_policies_and_snapshot_input_are_rejected(tmp_path):
    with pytest.raises(ValidationError):
        ObjectArrivalPolicy(min_similarity=2)
    with pytest.raises(ValidationError):
        ObjectArrivalPolicy(max_move_m=0.1)
    verifier, reference, _, _, _ = setup_arrival(tmp_path)
    with pytest.raises(ValidationError):
        verifier.verify_image(reference, observation(tmp_path, 1100), "https://example.test/image.png", lambda: False)


def test_movement_requires_a_stronger_appearance_match_than_a_nearby_sighting(tmp_path):
    verifier, reference, detector, comparator, _ = setup_arrival(tmp_path)
    view = reference.views[0]
    weaker = replace(view, memory=replace(view.memory, embedding=np.array([0.9, np.sqrt(0.19), 0])))
    reference = replace(reference, views=(weaker,))
    assert verifier.verify(reference, observation(tmp_path, 1100)).result == "matched"
    detector.detections = [Detection("printer", "red printer", RIGHT)]
    assert verifier.verify(reference, observation(tmp_path, 1100, (RIGHT,))).result == "ambiguous"
    assert detector.checks == 0 and len(comparator.calls) == 1


def test_invalid_comparison_or_clock_reset_during_verification_cannot_match(tmp_path):
    verifier, reference, _, comparator, now = setup_arrival(tmp_path)
    comparator.result = "invalid"
    with pytest.raises(ValidationError, match="comparison verdict"):
        verifier.verify(reference, observation(tmp_path, 1100))
    comparator.result = "matched"
    comparator.after = lambda: now.__setitem__(0, 900)
    with pytest.raises(ValidationError):
        verifier.verify(reference, observation(tmp_path, 1100))


def test_reference_excludes_unlocalized_historical_views_and_requires_saved_crops(tmp_path):
    verifier, reference, _, _, _ = setup_arrival(tmp_path)
    journal = verifier.tracker.store.objects
    view = reference.views[0]
    unchecked = replace(view, memory=replace(view.memory, id="unchecked", localization_checked=False))
    journal.save(reference.record, unchecked)
    assert verifier.capture(reference.record.id).views == reference.views
    journal.delete(reference.record.id)
    journal.save(reference.record, unchecked)
    with pytest.raises(ValidationError, match="localized saved crops"):
        verifier.capture(reference.record.id)
    journal.delete(reference.record.id)
    with pytest.raises(ValidationError, match="unavailable"):
        verifier.capture(reference.record.id)
