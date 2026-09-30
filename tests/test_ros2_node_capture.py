"""Camera capture through the production node: pose, localization and depth trust, admission, ordering.

Raw and compressed images take the same path; each case runs for both. The ingest worker is
not started, so accepted observations stay queued in the store where the tests read them back.
"""

from __future__ import annotations

import pytest

from placecell.errors import ValidationError
from placecell.navigation import NavigationCommands
from placecell.pipeline import Segmenter
from placecell.recordings import RecordingWriter
from placecell.ros2.bridge import KeyframeWriter, ObservationBuilder
from placecell.ros2.node import IngestWorker
from tests.conftest import FakeMediaEmbedder
from tests.ros_fakes import (
    CAMERA_FRAME,
    Duration,
    camera_info,
    depth_image,
    image,
    localization,
    spy,
)
from tests.ros_fakes import make_node as make_node
from tests.ros_fakes import ros as ros

IMAGE_TOPIC = "/camera/color/image_raw"
T0 = 1000


@pytest.fixture(params=[False, True], ids=["raw", "compressed"])
def compressed(request):
    return request.param


@pytest.fixture(autouse=True)
def idle_worker(monkeypatch):
    monkeypatch.setattr(IngestWorker, "start", lambda self: None)


@pytest.fixture
def objects(monkeypatch):
    """Parameters for object positions from aligned depth, with an image-capable embedder."""
    monkeypatch.setenv("PLACECELL_TEST_KEY", "offline")
    monkeypatch.setattr("placecell.ros2.node.build_embedder", lambda *a, **k: FakeMediaEmbedder())
    return {
        "objects_enabled": True,
        "object_backend": "chat",
        "object_model": "scripted",
        "object_api_key_env": "PLACECELL_TEST_KEY",
    }


def localize(node, x=0.0, y=0.0, yaw=0.0):
    node.subscription("/amcl_pose").callback(localization(node.clock.seconds, x, y, yaw))
    node._tf.set("map", "base_footprint", x, y, yaw=yaw)


def move(node, step, compressed):
    """A frame from one metre further along, three seconds later: past the admission interval."""
    node.clock.seconds = T0 + 3 * step
    localize(node, float(step))
    send(node, T0 + 3 * step, compressed)


def send(node, sec, compressed, **kwargs):
    node.subscription(IMAGE_TOPIC).callback(image(sec, compressed=compressed, **kwargs))


def queued(node):
    return [job.observation for job in node._store.jobs.pending(100)]


def warnings(node):
    return node.logger.messages("warning")


def test_trusted_capture_is_written_and_queued_with_its_pose(make_node, compressed, tmp_path):
    node = make_node(compressed=compressed)
    localize(node, 1.0, 2.0, 0.5)
    send(node, T0, compressed)
    (observation,) = queued(node)
    pose = observation.pose
    assert (pose.x, pose.y, pose.frame_id, pose.map_id) == (1.0, 2.0, "map", "test-v1")
    assert pose.yaw == pytest.approx(0.5)
    assert observation.timestamp == T0 and (observation.robot_id, observation.camera_id) == ("robot", "front")
    assert observation.localization_checked and observation.depth is None and not observation.refresh_objects
    assert sorted(p.name for p in (tmp_path / "keyframes").iterdir()) == [f"front_{T0 * 1000}.jpg"]
    assert node._sensors.ready(camera=True) and not node._sensors.ready(camera=True, depth=True)
    (lookup,) = node._tf.lookups
    assert (lookup.target, lookup.source, lookup.time.nanoseconds) == ("map", "base_footprint", T0 * 10**9)
    assert lookup.timeout == Duration(seconds=0.2)
    # Admission samples by time and motion: a frame one second later at the same pose is not stored.
    node.clock.seconds += 1
    localize(node, 1.0, 2.0, 0.5)
    send(node, T0 + 1, compressed)
    assert len(queued(node)) == 1 and node._worker.dropped == 0


def test_missing_transform_is_a_failed_capture(make_node, compressed):
    node = make_node(compressed=compressed)
    node.subscription("/amcl_pose").callback(localization(T0))
    send(node, T0, compressed)
    assert not queued(node)
    assert node._sensors.health()["failed"] == 1
    assert any(w.startswith("no pose for image: map -> base_footprint") for w in warnings(node))
    assert "skipping untrusted camera input: capture-time TF/localization is unavailable" in warnings(node)


def test_capture_without_trusted_localization_is_skipped(make_node, compressed):
    node = make_node(compressed=compressed)
    node._tf.set("map", "base_footprint")
    send(node, T0, compressed)
    assert not queued(node) and node._sensors.health()["failed"] == 1
    assert "skipping image: localization is missing, stale or uncertain" in warnings(node)


def test_unrequired_localization_admits_unchecked_captures(make_node, compressed):
    node = make_node(compressed=compressed, localization_required=False)
    node._tf.set("map", "base_footprint", 3.0)
    send(node, T0, compressed)
    (observation,) = queued(node)
    assert observation.pose.x == 3.0 and not observation.localization_checked


def test_zero_stamp_never_asks_tf_for_the_latest_pose(make_node, compressed):
    node = make_node(compressed=compressed)
    localize(node)
    send(node, 0, compressed)
    assert not queued(node) and not node._tf.lookups
    assert node._sensors.health()["failed"] == 1


def test_oversized_frame_is_dropped_before_validation(make_node, compressed):
    node = make_node(compressed=compressed, camera_max_message_bytes=16)
    localize(node)
    send(node, T0, compressed)
    assert not queued(node) and not node._tf.lookups
    assert node._pending_images.health()["oversized"] == 1
    assert node._sensors.health()["failed"] == 0


@pytest.mark.parametrize("fault", ["frame", "stamp", "format"])
def test_malformed_frame_is_a_failed_capture(make_node, compressed, fault):
    node = make_node(compressed=compressed)
    localize(node)
    message = image(T0 if fault != "stamp" else 1000.5, compressed=compressed)
    if fault == "frame":
        message.header.frame_id = ""
    elif fault == "format" and compressed:
        message.format = "png"
    elif fault == "format":
        message.encoding = "yuv422"
    node.subscription(IMAGE_TOPIC).callback(message)
    assert not queued(node) and node._sensors.health()["failed"] == 1
    assert any(w.startswith("skipping untrusted camera input: ") for w in warnings(node))


def test_full_queue_rejects_frames_unless_arrival_needs_one(make_node, compressed, monkeypatch, tmp_path):
    needed, observed = [False], []
    monkeypatch.setattr(NavigationCommands, "needs_observation", property(lambda self: needed[0]))
    monkeypatch.setattr(NavigationCommands, "observe", lambda self, obs: observed.append(obs))
    node = make_node(compressed=compressed, max_queue=1, navigation_enabled=True)
    for step in range(2):
        move(node, step, compressed)
    assert len(queued(node)) == 1 and node._worker.dropped == 1
    assert len(list((tmp_path / "keyframes").glob("*.jpg"))) == 1  # rejected before encoding
    assert [o.refresh_objects for o in observed] == [False]
    # A forced frame skips the admission interval and the capacity check, then meets the durable queue bound.
    needed[0] = True
    node.clock.seconds = T0 + 3.5
    localize(node, 1.0)
    message = image(T0 + 3, compressed=compressed)
    message.header.stamp.nanosec = 500_000_000
    node.subscription(IMAGE_TOPIC).callback(message)
    assert [o.refresh_objects for o in observed] == [False, True]
    assert observed[-1].timestamp == T0 + 3.5 and observed[-1].localization_checked
    assert len(queued(node)) == 1 and node._worker.dropped == 2


def test_accepted_frame_is_recorded_observed_submitted_admitted_then_confirmed(
    make_node, compressed, monkeypatch, ros, tmp_path
):
    for owner, method, name in (
        (RecordingWriter, "append", "record"),
        (NavigationCommands, "observe", "observe"),
        (IngestWorker, "submit", "submit"),
        (Segmenter, "accept", "accept"),
        (KeyframeWriter, "confirm", "confirm"),
    ):
        monkeypatch.setattr(owner, method, spy(ros.calls, name, getattr(owner, method)))
    node = make_node(compressed=compressed, navigation_enabled=True, recording_dir=str(tmp_path / "recording"))
    localize(node)
    send(node, T0, compressed)
    assert ros.calls == ["record", "observe", "submit", "accept", "confirm"]
    assert len(queued(node)) == 1
    assert not list((tmp_path / "keyframes").glob("*.pending"))


def test_rejected_submission_is_still_confirmed_but_not_admitted(make_node, compressed, monkeypatch, ros):
    monkeypatch.setattr(IngestWorker, "submit", spy(ros.calls, "submit", IngestWorker.submit, result=False))
    monkeypatch.setattr(Segmenter, "accept", spy(ros.calls, "accept", Segmenter.accept))
    monkeypatch.setattr(KeyframeWriter, "confirm", spy(ros.calls, "confirm", KeyframeWriter.confirm))
    node = make_node(compressed=compressed)
    localize(node)
    send(node, T0, compressed)
    assert ros.calls == ["submit", "confirm"]


def test_recording_failure_stops_recording_but_not_capture(make_node, compressed, monkeypatch, tmp_path):
    attempts = []

    def fail(self, observation):
        attempts.append(observation)
        raise OSError("disk full")

    monkeypatch.setattr(RecordingWriter, "append", fail)
    node = make_node(compressed=compressed, recording_dir=str(tmp_path / "recording"))
    for step in range(2):
        move(node, step, compressed)
    assert len(attempts) == 1 and len(queued(node)) == 2
    assert node.logger.messages("error") == ["recording stopped after an export failure: disk full"]


def test_clock_fault_while_encoding_discards_the_unconfirmed_frame(make_node, compressed, monkeypatch, tmp_path):
    node = make_node(compressed=compressed, recording_dir=str(tmp_path / "recording"))
    method = "from_compressed" if compressed else "from_raw"
    original = getattr(ObservationBuilder, method)

    def encode_then_fault(self, *args):
        observation = original(self, *args)
        node._sensors.clock_changed.set()
        return observation

    monkeypatch.setattr(ObservationBuilder, method, encode_then_fault)
    localize(node)
    send(node, T0, compressed)
    assert not queued(node) and not list((tmp_path / "recording").iterdir())
    # The creation marker stays, so restart recovery removes the orphaned keyframe.
    assert [p.name for p in (tmp_path / "keyframes").glob("*.pending")] == [f"front_{T0 * 1000}.jpg.pending"]


def test_encoding_error_skips_the_frame(make_node, compressed, monkeypatch):
    def broken(self, *args):
        raise ValidationError("bad pixels")

    monkeypatch.setattr(ObservationBuilder, "from_compressed" if compressed else "from_raw", broken)
    node = make_node(compressed=compressed)
    localize(node)
    send(node, T0, compressed)
    assert not queued(node) and "skipped image: bad pixels" in warnings(node)
    assert node._sensors.ready(camera=True)  # the capture itself was trusted


def rgbd(node, sec, compressed, *, depth=True, info=True, **kwargs):
    send(node, sec, compressed)
    if depth:
        node.subscription("/camera/aligned_depth_to_color/image_raw").callback(depth_image(sec))
    if info:
        node.subscription("/camera/color/camera_info").callback(camera_info(sec, **kwargs))
    node.timer(0.04).callback()


def test_aligned_depth_waits_for_its_pair_and_positions_the_capture(make_node, compressed, objects):
    node = make_node(compressed=compressed, rgbd_wait_s=60.0, **objects)
    localize(node)
    node._tf.set("map", CAMERA_FRAME, 1.0)
    send(node, T0, compressed)
    node.timer(0.04).callback()
    assert not queued(node) and node._pending_images.health()["rgb_queued"] == 1
    node.subscription("/camera/aligned_depth_to_color/image_raw").callback(depth_image(T0))
    node.subscription("/camera/color/camera_info").callback(camera_info(T0))
    node.timer(0.04).callback()
    (observation,) = queued(node)
    assert observation.depth is not None and observation.depth.map_from_camera[3] == 1.0
    assert node._sensors.ready(camera=True, depth=True)
    lookup = node._tf.lookups[-1]
    assert (lookup.target, lookup.source, lookup.time.nanoseconds) == ("map", CAMERA_FRAME, T0 * 10**9)
    assert lookup.timeout == Duration(seconds=0.2)


def test_unpaired_rgb_falls_back_to_a_scene_capture_after_the_wait(make_node, compressed, objects):
    node = make_node(compressed=compressed, rgbd_wait_s=1e-9, **objects)
    localize(node)
    rgbd(node, T0, compressed, depth=False, info=False)
    (observation,) = queued(node)
    assert observation.depth is None and not node._sensors.ready(depth=True)
    assert "object positions unavailable: waiting for aligned depth and CameraInfo" in warnings(node)


@pytest.mark.parametrize("fault", ["dimensions", "transform"])
def test_untrusted_depth_leaves_a_scene_capture(make_node, compressed, objects, fault):
    node = make_node(compressed=compressed, rgbd_wait_s=60.0, **objects)
    localize(node)
    if fault != "transform":
        node._tf.set("map", CAMERA_FRAME)
    rgbd(node, T0, compressed, **({"width": 8} if fault == "dimensions" else {}))
    (observation,) = queued(node)
    assert observation.depth is None
    expected = "RGB and aligned depth dimensions differ" if fault == "dimensions" else f"map -> {CAMERA_FRAME}"
    assert any(w.startswith(f"object positions unavailable: {expected}") for w in warnings(node))


def test_depth_needs_a_localization_uncertainty(make_node, compressed, objects):
    node = make_node(compressed=compressed, rgbd_wait_s=60.0, localization_required=False, **objects)
    node._tf.set("map", "base_footprint")
    node._tf.set("map", CAMERA_FRAME)
    rgbd(node, T0, compressed)
    (observation,) = queued(node)
    assert observation.depth is None and not any("object positions" in w for w in warnings(node))


@pytest.mark.parametrize("stamp", [T0 - 10, "bad"])
def test_invalid_or_stale_rgbd_input_is_a_failed_capture(make_node, compressed, objects, stamp):
    node = make_node(compressed=compressed, **objects)
    localize(node)
    send(node, stamp, compressed)
    assert node._pending_images.health()["rgb_queued"] == 0
    assert node._sensors.health()["failed"] == 1


def test_clock_jump_latches_a_fault_that_blocks_and_clears_rgbd_input(make_node, compressed, objects):
    node = make_node(compressed=compressed, rgbd_wait_s=60.0, **objects)
    (jump,) = node.clock.jumps
    assert jump.threshold.min_forward is None and jump.threshold.on_clock_change
    assert jump.threshold.min_backward == Duration(nanoseconds=-1)
    localize(node)
    send(node, T0, compressed)
    node.subscription("/camera/aligned_depth_to_color/image_raw").callback(depth_image(T0))
    node.clock.jump(10)
    send(node, T0 + 1, compressed)
    assert node._pending_images.health()["rgb_queued"] == 1  # nothing new is queued
    node.timer(0.04).callback()
    node.timer(0.04).callback()
    health = node._pending_images.health()
    assert health["rgb_queued"] == health["depth_queued"] == 0 and not queued(node)
    assert node.logger.messages("error") == [
        "Clock changed or reset: navigation and new captures are blocked. "
        "Confirm Nav2 is stopped, then restart with a fresh collection and keyframe directory."
    ]
    node.destroy_node()
    assert not jump.registered


def test_clock_fault_blocks_direct_capture(make_node, compressed):
    node = make_node(compressed=compressed)
    localize(node)
    node.clock.jump(T0)
    send(node, T0, compressed)
    assert not queued(node) and not node._tf.lookups


def test_localization_messages_drive_the_gate(make_node):
    node = make_node()
    topic = node.subscription("/amcl_pose")
    topic.callback(localization(T0))
    assert node._localization.ready()
    node.clock.seconds += 2
    topic.callback(localization(T0 + 1, frame="odom"))
    assert not node._localization.ready()
    topic.callback(localization(T0 + 1.5))
    assert node._localization.ready()
    topic.callback(object())
    assert not node._localization.ready()


def test_stationary_odometry_extends_localization_while_idle(make_node):
    node = make_node()
    odometry = node.timers[0]  # created before the operator's snapshot timer, which shares its period
    assert odometry.period == 0.2
    assert odometry.clock.clock_type == "STEADY_TIME"
    odometry.callback()
    assert node.logger.records[-1] == (
        "warning",
        "no odometry: idle localization expires after localization_max_age_s: odom -> base_footprint does not exist",
        {"throttle_duration_sec": 60.0},
    )
    node.subscription("/amcl_pose").callback(localization(T0))
    for stamp in (T0, T0 + 4):
        node.clock.seconds = stamp
        node._tf.set("odom", "base_footprint", stamp=stamp)
        odometry.callback()
    lookup = node._tf.lookups[-1]
    assert (lookup.target, lookup.source, lookup.time.nanoseconds, lookup.timeout) == (
        "odom",
        "base_footprint",
        0,
        None,
    )
    node.clock.seconds = T0 + 7  # older than localization_max_age_s, but four seconds were at rest
    assert node._localization.ready()
    node.clock.seconds = T0 + 9.5
    assert not node._localization.ready()


def test_without_an_odometry_frame_no_timer_polls_tf(make_node):
    node = make_node(odom_frame="")
    assert [t.period for t in node.timers].count(0.2) == 1  # the operator snapshot only
