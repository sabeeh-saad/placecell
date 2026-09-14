import json
from dataclasses import replace

import pytest

from placecell import CollectionInfo, InMemoryStore, ObjectTracker, RecordingWriter, read_recording
from placecell.errors import ValidationError
from placecell.memory import memory_id
from placecell.object_evaluation import FrameLabels, ObjectLabel, UsageTransport, evaluate_objects, load_object_labels
from tests.conftest import FakeTransport
from tests.test_objects import CENTER, Detector, PixelEmbedder, observation


def tracker():
    embedder = PixelEmbedder()
    store = InMemoryStore(CollectionInfo("evaluation", embedder.model_name, embedder.dimension))
    return ObjectTracker(store, embedder, Detector())


def labels(obs, visibility="visible", physical_id="printer-a", position=(0, 0, 2)):
    identity = memory_id(obs.robot_id, obs.camera_id, obs.timestamp)
    return FrameLabels(
        identity, (ObjectLabel(physical_id, visibility, CENTER if visibility == "visible" else None, position),)
    )


def test_export_copies_images_and_round_trips_stamped_depth(tmp_path):
    obs = observation(tmp_path)
    writer = RecordingWriter(tmp_path / "session")
    writer.append(obs)
    restored = next(iter(read_recording(writer.directory / "observations.jsonl")))
    assert restored.depth == obs.depth
    assert restored.pose == obs.pose and restored.timestamp == obs.timestamp
    assert restored.evidence.managed is False
    assert restored.evidence.uri != obs.evidence.uri
    from pathlib import Path

    Path(obs.evidence.uri).unlink()
    assert Path(restored.evidence.uri).exists()
    with pytest.raises(ValidationError):
        writer.append(obs)
    with pytest.raises(FileExistsError):
        RecordingWriter(writer.directory)


def test_replay_measures_identity_and_position_without_ground_truth_in_provider(tmp_path):
    observations = [observation(tmp_path, 1000), observation(tmp_path, 2000)]
    writer = RecordingWriter(tmp_path / "session")
    for obs in observations:
        writer.append(obs)
    human_labels = {frame.observation_id: frame for frame in map(labels, observations)}
    runner = tracker()
    report = evaluate_objects(runner, read_recording(writer.directory / "observations.jsonl"), human_labels)
    assert report["counts"]["scans"] == 2
    assert report["counts"]["identity_switches"] == 0
    assert report["visible_detection_recall"] == 1
    assert report["mean_surface_position_error_m"] == pytest.approx(0)
    assert report["mean_scan_latency_ms"] >= 0
    assert report["absence_confirmation_rate"] is None
    assert runner.detector.calls == 2


def test_false_merge_is_detected_when_same_track_is_a_different_physical_object(tmp_path):
    observations = [observation(tmp_path, 1000), observation(tmp_path, 2000)]
    frames = [labels(observations[0]), labels(observations[1], physical_id="printer-b")]
    report = evaluate_objects(tracker(), observations, {f.observation_id: f for f in frames})
    assert report["counts"]["false_merges"] == 1
    assert report["identity_failure_examples"][0]["truth_id"] == "printer-b"


def test_omissions_and_unknown_identity_are_measured(tmp_path):
    obs = observation(tmp_path)
    runner = tracker()
    runner.detector.detections = []
    frame = labels(obs)
    report = evaluate_objects(runner, [obs], {frame.observation_id: frame})
    assert report["visible_detection_recall"] == 0
    assert report["counts"]["unknown_identity_checks"] == 1
    assert report["mean_surface_position_error_m"] is None


def test_scan_interval_labels_and_bad_inputs_are_explicit(tmp_path):
    observations = [observation(tmp_path, 1000), observation(tmp_path, 1001)]
    frames = {f.observation_id: f for f in map(labels, observations)}
    report = evaluate_objects(tracker(), observations, frames)
    assert report["labeled_frames_skipped_by_scan_interval"] == 1
    with pytest.raises(ValidationError, match="labels must"):
        evaluate_objects(tracker(), observations[:1], frames)
    with pytest.raises(ValidationError, match="max_frames"):
        evaluate_objects(tracker(), observations, frames, max_frames=1)
    with pytest.raises(ValidationError):
        ObjectLabel("p", "occluded", CENTER)


def test_label_loader_and_recording_path_escape_rejection(tmp_path):
    obs = observation(tmp_path)
    frame = labels(obs)
    path = tmp_path / "labels.jsonl"
    row = {
        "observation_id": frame.observation_id,
        "objects": [
            {"id": "printer-a", "visibility": "visible", "box": [0.45, 0.45, 0.55, 0.55], "position": [0, 0, 2]}
        ],
    }
    path.write_text(json.dumps(row) + "\n")
    assert load_object_labels(path)[frame.observation_id] == frame
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValidationError, match="duplicate"):
        load_object_labels(path)
    writer = RecordingWriter(tmp_path / "session")
    writer.append(obs)
    manifest = writer.directory / "observations.jsonl"
    row = json.loads(manifest.read_text())
    row["observation"]["evidence"]["uri"] = "../outside.png"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValidationError, match="within"):
        list(read_recording(manifest))


def test_usage_reports_missing_billing_data_as_unknown_and_never_saves_prompts():
    transport = FakeTransport(
        [(200, {}, {"usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 10}})]
    )
    meter = UsageTransport(transport, input_usd_per_million=2, output_usd_per_million=5)
    meter.post_json("https://example.test", {}, {"private": "image pixels"}, 1)
    result = meter.report()
    assert result["estimated_cost_usd"] == pytest.approx(0.00035)
    assert result["requests"] == 1 and "private" not in json.dumps(result)
    unknown = UsageTransport(FakeTransport([(200, {}, {})]), input_usd_per_million=2, output_usd_per_million=5)
    unknown.post_json("https://example.test", {}, {}, 1)
    assert unknown.report()["estimated_cost_usd"] is None
    assert unknown.report()["requests_without_token_usage"] == 1


def test_recording_rejects_invalid_or_duplicate_identity_before_copying(tmp_path):
    obs = observation(tmp_path)
    writer = RecordingWriter(tmp_path / "session")
    with pytest.raises(ValidationError):
        writer.append(replace(obs, timestamp=float("nan")))
    assert not list(writer.directory.iterdir())
    writer.append(obs)
    with pytest.raises(ValidationError):
        writer.append(replace(obs, timestamp=1000.0001))
    with pytest.raises(ValidationError):
        writer.append(replace(obs, timestamp=2000, camera_id="other"))


@pytest.mark.parametrize(
    "visibility,background,missing", [("absent", 5, True), ("occluded", 1, False), ("occluded", 5, True)]
)
def test_presence_metrics_distinguish_absence_occlusion_and_false_missing(tmp_path, visibility, background, missing):
    runner = tracker()
    first = observation(tmp_path)
    frames = [first] + [observation(tmp_path, 1000 + i * 1000, (), (), background=background) for i in range(1, 4)]
    truth = [labels(first)] + [labels(obs, visibility, position=None) for obs in frames[1:]]

    def replay():
        yield first
        runner.detector.detections = []
        yield from frames[1:]

    report = evaluate_objects(runner, replay(), {f.observation_id: f for f in truth})
    assert report["counts"]["false_missing"] == int(missing and visibility == "occluded")
    assert report["counts"]["confirmed_absence"] == int(missing and visibility == "absent")
    assert report["counts"]["unknown_identity_checks"] == 0


def test_fragmentation_is_counted_and_incomplete_labels_do_not_count_false_detections(tmp_path):
    runner = tracker()
    frames = [observation(tmp_path), observation(tmp_path, 2000, colors=("blue",))]
    truth = [labels(obs) for obs in frames]
    report = evaluate_objects(runner, frames, {f.observation_id: f for f in truth})
    assert report["counts"]["identity_switches"] == 1
    assert report["track_fragmentations"] == 1
    runner = tracker()
    partial = FrameLabels(truth[0].observation_id, (), complete=False)
    report = evaluate_objects(runner, frames[:1], {partial.observation_id: partial})
    assert report["counts"]["unmatched_detections_in_complete_frames"] == 0


def test_reference_store_close_clears_object_tracks_and_scan_replay_state(tmp_path):
    runner = tracker()
    obs = observation(tmp_path)
    runner.commit(runner.prepare(obs))
    assert runner.store.objects.count() == 1
    runner.store.close()
    assert runner.store.objects.count() == 0
    removed = []
    runner.store.drain_cleanup(removed.append)
    assert removed == [obs.evidence]
    runner.commit(runner.prepare(obs))
    assert runner.store.objects.count() == 1 and runner.detector.calls == 2


def test_evaluation_cli_replays_export_without_live_collection_or_network(tmp_path, monkeypatch):
    from placecell.object_evaluation import main

    monkeypatch.setattr("placecell.providers.GeminiEmbedder", lambda *a, **k: PixelEmbedder())
    monkeypatch.setattr("placecell.providers.GeminiObjectDetector", lambda *a, **k: Detector())
    writer = RecordingWriter(tmp_path / "session")
    obs = observation(tmp_path)
    writer.append(obs)
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "observation_id": labels(obs).observation_id,
                "objects": [{"id": "printer-a", "visibility": "visible", "box": [0.45, 0.45, 0.55, 0.55]}],
            }
        )
        + "\n"
    )
    output = tmp_path / "report.json"
    args = [
        "--recording",
        str(writer.directory / "observations.jsonl"),
        "--labels",
        str(labels_path),
        "--output",
        str(output),
        "--vision-model",
        "test-vision",
    ]
    main(args)
    report = json.loads(output.read_text())
    assert report["visible_detection_recall"] == 1 and report["vision_model"] == "test-vision"
    assert report["api_usage"]["vision"]["requests"] == 0
    with pytest.raises(ValidationError, match="already exists"):
        main(args)
