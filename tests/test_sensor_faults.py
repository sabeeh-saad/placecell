"""Sensor provenance must survive failures, recovery and callback interleavings."""

import math
from types import SimpleNamespace as Obj

import pytest

from placecell import Destination, NavigationEvent, Pose
from placecell.errors import ValidationError
from placecell.localization import LocalizationGate
from placecell.ros2.bridge import image_dimensions, pose_from_transform, stamp_to_seconds
from placecell.ros2.depth import PendingImages, aligned_snapshot
from placecell.sensors import SensorHealth
from tests.test_depth import messages
from tests.test_localization import covariance
from tests.test_missions import mission as mission
from tests.test_missions import start


def test_clock_rewind_within_localization_age_revokes_trust():
    now = [100.0]
    gate = LocalizationGate("map", "office", clock=lambda: now[0])
    assert gate.update(98, Pose(0, 0, map_id="office"), covariance())
    now[0] = 99  # Still newer than the sample, but in a different clock epoch.
    assert not gate.ready()


def test_expired_localization_cannot_revive_when_clock_catches_up():
    now = [100.0]
    gate = LocalizationGate("map", "office", clock=lambda: now[0])
    assert gate.update(100, Pose(0, 0, map_id="office"), covariance())
    now[0] = 106
    assert not gate.ready()
    now[0] = 101
    assert not gate.ready()


def test_named_goal_success_after_localization_loss_does_not_advance(mission):
    m = mission
    start(m)
    m.ready[0] = False
    m.nav.sent[0][2](NavigationEvent("succeeded"))  # Result overtakes the poll timer.
    assert m.events[-1].state == "canceled"
    assert not m.tasks and not m.commands.busy
    assert not any(e.state in {"succeeded", "step_succeeded"} for e in m.events)


@pytest.mark.parametrize("quaternion", [(0, 0, 0, 0), (0, 0, 0, 2), (math.inf, 0, 0, 1)])
def test_invalid_base_tf_quaternion_cannot_supply_capture_pose(quaternion):
    with pytest.raises(ValidationError):
        pose_from_transform(0, 0, *quaternion)


@pytest.mark.parametrize("stamp", [math.nan, math.inf, -1.0])
def test_invalid_rgb_timestamp_cannot_pair_with_depth(stamp):
    with pytest.raises(ValidationError):
        aligned_snapshot(*messages(), rgb_stamp=stamp, rgb_frame="camera_optical")


@pytest.mark.parametrize("phase", ["planning", "motion", "result"])
def test_localization_recovery_does_not_resume_interrupted_mission(mission, phase):
    m = mission
    gate = LocalizationGate("map", "office", clock=lambda: m.now[0])
    pose = Pose(0, 0, map_id="office")
    assert gate.update(m.now[0], pose, covariance())
    m.commands._localization_check = gate.ready
    m.commands._localization_generation = lambda: gate.generation
    if phase == "planning":
        m.commands.handle("Visit the printer then cupboard")
    else:
        start(m)
    gate.invalidate()
    m.now[0] += 0.1
    assert gate.update(m.now[0], pose, covariance()) and gate.ready()
    if phase == "planning":
        m.tasks.pop(0)()
        assert not m.nav.sent and not m.commands.busy
    elif phase == "motion":
        m.commands.poll()
        assert m.nav.canceled and m.commands.busy
        m.nav.sent[0][2](NavigationEvent("canceled"))
    else:
        m.nav.sent[0][2](NavigationEvent("succeeded"))
        assert m.events[-1].state == "canceled"
    assert not m.tasks and not m.commands.busy
    assert not any(e.state in {"succeeded", "step_succeeded"} for e in m.events)


@pytest.fixture
def sensor():
    now, mono = [100.0], [0.0]
    health = SensorHealth(2, clock=lambda: now[0], monotonic=lambda: mono[0])
    return Obj(health=health, now=now, mono=mono)


@pytest.mark.parametrize("fault", ["camera_missing", "depth_missing", "paused", "replayed", "forward", "future"])
def test_sensor_faults_revoke_only_affected_capabilities(sensor, fault):
    s = sensor
    assert not s.health.ready(camera=True)
    assert s.health.observe(100, camera=True, depth=True)
    before = s.health.generation(depth=True)
    if fault == "depth_missing":
        s.now[0] += 2.1
        assert s.health.observe(s.now[0], camera=True, depth=False)
    elif fault == "future":
        assert not s.health.observe(101, camera=True, depth=True)
    elif fault == "camera_missing":
        s.now[0] += 0.1
        assert not s.health.observe(s.now[0], camera=False, depth=False)
    else:
        s.mono[0] = 3
        if fault == "replayed":
            assert not s.health.observe(100, camera=True, depth=True)
        elif fault == "forward":
            s.now[0] += 20
    assert not s.health.ready(camera=True, depth=True)
    assert s.health.ready()  # Named places do not depend on the RGB-D stream.
    assert s.health.ready(camera=True) == (fault == "depth_missing")
    assert s.health.generation(depth=True) != before
    s.now[0] += 1
    assert s.health.observe(s.now[0], camera=True, depth=True)
    assert s.health.ready(camera=True, depth=True)


@pytest.mark.parametrize("fault", ["rewind", "jump_callback", "nonfinite"])
def test_clock_fault_is_latched_even_after_new_valid_sensor_messages(sensor, fault):
    s = sensor
    assert s.health.observe(100, camera=True, depth=True)
    if fault == "rewind":
        s.now[0] = 99
    elif fault == "nonfinite":
        s.now[0] = math.nan
    else:
        s.health.clock_changed.set()
    assert not s.health.ready()
    s.now[0] = 101
    assert not s.health.observe(101, camera=True, depth=True)
    assert not s.health.ready()


@pytest.mark.parametrize("object_goal", [False, True])
@pytest.mark.parametrize("phase", ["dispatch", "motion", "result", "recovered_result"])
def test_live_sensor_loss_blocks_dispatch_or_success(mission, sensor, object_goal, phase, monkeypatch):
    m, s = mission, sensor
    goal = Destination("printer", Pose(1, 2, map_id="office"), "memory", object_id="object" if object_goal else "")
    # Isolate the controller capability boundary from retrieval and model behavior.
    from placecell.navigation import Resolution

    monkeypatch.setattr(m.resolver, "resolve", lambda _: Resolution("resolved", "Selected", (goal,)))
    monkeypatch.setattr(m.resolver, "current", lambda _: True)
    monkeypatch.setattr(m.resolver, "prepare_destination", lambda d, _: d)
    m.commands._sensor_ready = lambda d: s.health.ready(camera=d.source == "memory", depth=bool(d.object_id))
    m.commands._sensor_generation = lambda d: s.health.generation(depth=bool(d.object_id))
    if phase != "dispatch":
        assert s.health.observe(s.now[0], camera=True, depth=True)
    start(m)
    if phase == "dispatch":
        assert m.events[-1].state == "unavailable" and not m.nav.sent
        return
    s.now[0] += 2.1 if object_goal else 0.1
    s.health.observe(s.now[0], camera=object_goal, depth=False)
    if phase == "recovered_result":
        s.now[0] += 0.1
        assert s.health.observe(s.now[0], camera=True, depth=True)
    if phase == "motion":
        m.commands.poll()
        assert m.nav.canceled and m.commands.busy
        m.nav.sent[0][2](NavigationEvent("canceled"))
        assert m.events[-1].state == "canceled"
    else:
        m.nav.sent[0][2](NavigationEvent("succeeded"))
        assert m.events[-1].state == "destination_unverified"
    assert not m.tasks and not m.commands.busy
    assert not any(e.state in {"succeeded", "step_succeeded"} for e in m.events)


def test_missing_depth_pair_does_not_refresh_or_revoke_recent_valid_depth(sensor):
    s = sensor
    s.health.observe(100, camera=True, depth=True)
    generation = s.health.generation(depth=True)
    for stamp in (100.2, 100.8, 101.5):
        s.now[0] = stamp
        assert s.health.observe(stamp, camera=True, depth=False)
        assert s.health.ready(camera=True, depth=True)
        assert s.health.generation(depth=True) == generation
    s.now[0] = 102.1
    s.health.observe(102.1, camera=True, depth=False)
    assert s.health.ready(camera=True) and not s.health.ready(depth=True)
    assert s.health.generation(depth=True) != generation


@pytest.mark.parametrize("stamp", [(-1, 0), (1, -1), (1, 1_000_000_000), (math.nan, 0), (1.1, 0)])
def test_invalid_ros_stamps_fail_closed(stamp):
    with pytest.raises(ValidationError):
        stamp_to_seconds(*stamp)


def test_clearing_pending_images_discards_all_old_epoch_inputs():
    pending = PendingImages()
    depth, info, _ = messages()
    pending.depth.append(depth)
    pending.info.append(info)
    pending.add(depth, False, 0)
    pending.clear()
    assert not pending.depth and not pending.info and pending.pop(1) is None


@pytest.mark.parametrize("fault", ["empty", "dimensions", "stride", "encoding", "frame", "jpeg"])
def test_invalid_camera_buffers_do_not_refresh_health(fault):
    msg = Obj(header=Obj(frame_id="camera"), width=2, height=2, encoding="rgb8", step=6, data=b"a" * 12)
    assert image_dimensions(msg) == (2, 2)
    if fault == "empty":
        msg.data = b""
    elif fault == "dimensions":
        msg.width = 5_000_000
    elif fault == "stride":
        msg.step = 7
    elif fault == "encoding":
        msg.encoding = "invalid"
    elif fault == "frame":
        msg.header.frame_id = ""
    else:
        msg.format = "jpeg"
    with pytest.raises(ValidationError):
        image_dimensions(msg, compressed=fault == "jpeg")


def test_static_calibration_still_requires_capture_time_depth():
    depth, info, transform = messages()
    info.header = Obj(frame_id="camera_optical", stamp=Obj(sec=0, nanosec=0))
    assert aligned_snapshot(depth, info, transform, rgb_stamp=10, rgb_frame="camera_optical")
    with pytest.raises(ValidationError):
        aligned_snapshot(depth, info, transform, rgb_stamp=11, rgb_frame="camera_optical")


def test_malformed_covariance_revokes_trust_without_escaping_callback():
    gate = LocalizationGate("map", "office", clock=lambda: 100)
    assert gate.update(99, Pose(0, 0, map_id="office"), covariance())
    assert not gate.update(100, Pose(0, 0, map_id="office"), ["bad"] * 36)
    assert not gate.ready() and gate.generation == 1


def test_zero_clock_localization_is_not_an_initialized_capture():
    gate = LocalizationGate("map", "office", clock=lambda: 0)
    assert not gate.update(0, Pose(0, 0, map_id="office"), covariance())
    assert not gate.ready()


@pytest.mark.parametrize("stamp", [0, 90, 101, 1000000])
def test_invalid_rgb_stamp_cannot_poison_synchronizer_high_watermark(stamp):
    pending = PendingImages()
    msg = Obj(header=Obj(stamp=Obj(sec=stamp, nanosec=0), frame_id="camera"))
    assert not pending.add(msg, False, 0, source_now=100)
    msg.header.stamp.sec = 100
    assert pending.add(msg, False, 0, source_now=100)
    assert pending.pop(1) == (msg, False)


@pytest.mark.parametrize("stamp", [-1, 0])
def test_invalid_depth_stamps_cannot_break_synchronizer(stamp):
    pending = PendingImages()
    msg = Obj(header=Obj(stamp=Obj(sec=stamp, nanosec=0), frame_id="camera"))
    assert not pending.add_depth(msg)
    assert not pending.depth
    assert pending.add_depth(msg, calibration=True) == (stamp == 0)


def test_sensor_loss_during_submitting_status_blocks_the_actual_send(mission):
    m = mission
    publish = m.commands._publish_callback

    def revoke(update):
        publish(update)
        if update.state == "submitting":
            m.ready[0] = False

    m.commands._publish_callback = revoke
    start(m)
    assert not m.nav.sent and not m.commands.busy
    assert m.events[-1].state == "unavailable"


def test_new_instruction_can_use_recovered_localization_after_interruption(mission):
    m = mission
    gate = LocalizationGate("map", "office", clock=lambda: m.now[0])
    pose = Pose(0, 0, map_id="office")
    assert gate.update(m.now[0], pose, covariance())
    m.commands._localization_check = gate.ready
    m.commands._localization_generation = lambda: gate.generation
    start(m)
    gate.invalidate()
    m.commands.poll()
    m.nav.sent[0][2](NavigationEvent("canceled"))
    m.now[0] += 0.1
    assert gate.update(m.now[0], pose, covariance())
    m.commands._mission_planner = None
    m.commands.handle("go to printer")
    m.tasks.pop(0)()
    assert len(m.nav.sent) == 2
    m.nav.sent[1][2](NavigationEvent("succeeded"))
    assert m.events[-1].state == "succeeded"
