from __future__ import annotations

import math
from types import SimpleNamespace as Obj

import pytest

from placecell import Pose
from placecell.errors import ValidationError
from placecell.localization import LocalizationGate, LocalizationPolicy
from placecell.ros2.bridge import update_localization, update_odometry


def covariance(x=0.01, y=0.01, yaw=0.01):
    values = [0.0] * 36
    values[0], values[7], values[35] = x, y, yaw
    return values


def test_recent_quality_is_required_at_capture_and_dispatch():
    now, mono = [100.0], [0.0]
    gate = LocalizationGate("map", "office", clock=lambda: now[0], monotonic=lambda: mono[0])
    pose = Pose(1, 2, map_id="office")
    assert not gate.ready()
    assert gate.update(100, pose, covariance())
    assert gate.ready() and gate.accepts(pose, 100)
    assert not gate.accepts(Pose(2, 2, map_id="office"), 100)
    assert not gate.accepts(Pose(1, 2, 1.5, map_id="office"), 100)
    assert not gate.accepts(Pose(1, 2, map_id="old-office"), 100)
    assert not gate.accepts(pose, 101)  # Future camera stamp.
    mono[0] = 6  # A paused ROS clock cannot keep a pose ready forever.
    assert not gate.ready()
    assert not gate.update(100, pose, covariance())  # Replaying a stamp cannot refresh it.
    assert not gate.ready()
    now[0], mono[0] = 101, 7
    assert gate.update(101, pose, covariance())
    now[0] = 107
    assert not gate.ready() and not gate.accepts(pose, 107)
    now[0] = 10  # A new simulation epoch can recover after a fresh localization.
    assert gate.update(10, pose, covariance()) and gate.ready()


@pytest.mark.parametrize(
    "values",
    [
        covariance(x=1),
        covariance(yaw=1),
        covariance(x=-1),
        covariance(x=float("nan")),
        covariance(y=float("inf")),
        [0.0] * 36,
        [0.1],
    ],
)
def test_bad_covariance_revokes_previous_good_estimate(values):
    gate = LocalizationGate("map", "office", clock=lambda: 100)
    pose = Pose(0, 0, map_id="office")
    assert gate.update(99, pose, covariance())
    assert not gate.update(100, pose, values) and not gate.ready()


def test_planar_covariance_must_be_symmetric_and_positive_semidefinite():
    gate = LocalizationGate("map", "office", clock=lambda: 100)
    pose = Pose(0, 0, map_id="office")
    values = covariance()
    values[1] = 0.1
    assert not gate.update(99, pose, values)
    values[6] = 0.1
    assert not gate.update(100, pose, values)
    with pytest.raises(ValidationError):
        LocalizationPolicy(max_age_s=float("nan"))


def test_wrong_frame_and_invalid_message_do_not_authorize_captures():
    gate = LocalizationGate("map", "office", clock=lambda: 100)
    assert not gate.update(100, Pose(0, 0, frame_id="odom", map_id="office"), covariance())
    message = Obj(
        header=Obj(stamp=Obj(sec=100, nanosec=0), frame_id="map"),
        pose=Obj(pose=Obj(position=Obj(x=0, y=0), orientation=Obj(x=0, y=0, z=0, w=1)), covariance=covariance()),
    )
    gate = LocalizationGate("map", "office", clock=lambda: 100)
    assert update_localization(gate, message, "office") and gate.ready()
    message.pose.pose.orientation.w = 0
    assert not update_localization(gate, message, "office") and not gate.ready()
    assert not update_localization(gate, Obj(), "office")
    assert not gate.update(float("nan"), Pose(0, 0), covariance())


def test_delayed_localization_does_not_replace_a_newer_estimate():
    gate = LocalizationGate("map", "office", clock=lambda: 100)
    pose = Pose(0, 0, map_id="office")
    assert gate.update(100, pose, covariance())
    assert not gate.update(99, pose, covariance(x=100))
    assert gate.ready()


def test_capture_uncertainty_uses_measured_covariance_and_expires():
    now = [100.0]
    gate = LocalizationGate("map", "office", clock=lambda: now[0])
    pose = Pose(0, 0, map_id="office")
    assert gate.uncertainty_at(100) is None
    assert gate.update(100, pose, covariance(x=0.0025, y=0.01, yaw=0.0004))
    assert gate.uncertainty_at(100) == pytest.approx((0.1, 0.02))
    assert gate.uncertainty_at(101) is None
    assert gate.uncertainty_at(float("nan")) is None
    now[0] = 106
    assert gate.uncertainty_at(106) is None
    assert not gate.update(106, pose, covariance(yaw=100))
    assert gate.uncertainty_at(106) is None


def test_capture_stamps_within_future_tolerance_are_checked_like_current_ones():
    gate = LocalizationGate("map", "office", clock=lambda: 100)
    pose = Pose(0, 0, map_id="office")
    assert gate.update(100, pose, covariance())
    assert gate.accepts(pose, 100.05) and gate.uncertainty_at(100.05) is not None
    assert not gate.accepts(pose, 100.2) and gate.uncertainty_at(100.2) is None
    strict = LocalizationGate("map", "office", LocalizationPolicy(max_capture_future_s=0), clock=lambda: 100)
    assert strict.update(100, pose, covariance()) and not strict.accepts(pose, 100.05)


@pytest.mark.parametrize("tolerance", [-0.1, 5.0, float("inf")])
def test_capture_future_tolerance_must_be_below_maximum_age(tolerance):
    with pytest.raises(ValidationError):
        LocalizationPolicy(max_capture_future_s=tolerance)


class Odometry:
    """Advance both clocks and feed odometry-frame poses every `step` seconds."""

    def __init__(self, gate, now, mono):
        self.gate, self.now, self.mono, self.x, self.yaw = gate, now, mono, 0.0, 0.0

    def run(self, seconds, *, speed=0.0, turn=0.0, step=0.5):
        for _ in range(round(seconds / step)):
            self.now[0] += step
            self.mono[0] += step
            self.x += speed * step
            self.yaw += turn * step
            self.gate.update_odometry(self.now[0], Pose(self.x, 0, self.yaw, frame_id="odom"))


@pytest.fixture
def idle():
    now, mono = [100.0], [0.0]
    gate = LocalizationGate("map", "office", clock=lambda: now[0], monotonic=lambda: mono[0])
    pose = Pose(1, 2, map_id="office")
    assert gate.update(100, pose, covariance())
    odometry = Odometry(gate, now, mono)
    assert gate.update_odometry(100, Pose(0, 0, frame_id="odom"))
    return Obj(gate=gate, now=now, pose=pose, odometry=odometry)


def test_idle_robot_keeps_an_old_estimate_while_odometry_confirms_rest(idle):
    i = idle
    i.odometry.run(600, speed=0.00005)  # Sub-threshold creep still counts as rest.
    assert i.gate.ready() and i.gate.generation == 0
    assert i.gate.accepts(i.pose, i.now[0]) and i.gate.accepts(i.pose, i.now[0] - 4)
    assert i.gate.uncertainty_at(i.now[0]) == pytest.approx((0.1, 0.1))
    assert not i.gate.accepts(i.pose, i.now[0] - 6)  # Captures still need their own freshness.


@pytest.mark.parametrize("motion", [{"speed": 0.2}, {"turn": 0.2}])
def test_moving_time_after_a_long_rest_still_expires_the_estimate(idle, motion):
    i = idle
    i.odometry.run(60)
    i.odometry.run(4.5, **motion)  # A trip starting after the rest still has time to relocalize.
    assert i.gate.ready()
    i.odometry.run(1, **motion)
    assert not i.gate.ready() and i.gate.generation == 1


def test_new_estimate_during_motion_restores_the_full_age(idle):
    i = idle
    i.odometry.run(60)
    i.odometry.run(4.5, speed=0.2)
    assert i.gate.update(i.now[0], i.pose, covariance())
    i.odometry.run(4.5, speed=0.2)
    assert i.gate.ready() and i.gate.generation == 0


def test_stop_and_go_motion_consumes_one_cumulative_age_budget(idle):
    i = idle
    i.odometry.run(3, speed=0.2)  # Stops short of the localizer's update distance.
    i.odometry.run(60)
    assert i.gate.ready()
    i.odometry.run(2.5, speed=0.2)
    assert not i.gate.ready()


@pytest.mark.parametrize("fault", ["none", "repeated", "stale", "stopped", "paused", "frame", "nonfinite"])
def test_missing_or_invalid_odometry_keeps_the_ordinary_age_limit(fault):
    now, mono = [100.0], [0.0]
    gate = LocalizationGate("map", "office", clock=lambda: now[0], monotonic=lambda: mono[0])
    assert gate.update(100, Pose(0, 0, map_id="office"), covariance())
    for step in range(1, 17):
        now[0] += 0 if fault == "paused" else 0.5
        mono[0] += 0.5
        stamp = {"repeated": 100.0, "stale": now[0] - 10, "nonfinite": math.nan}.get(fault, now[0])
        frame = "other_odom" if fault == "frame" and step % 2 else "odom"
        if fault != "none" and not (fault == "stopped" and step > 4):
            gate.update_odometry(stamp, Pose(0, 0, frame_id=frame))
    assert not gate.ready() and gate.generation == 1


def test_odometry_gap_longer_than_the_remaining_age_expires_the_estimate(idle):
    i = idle
    i.odometry.run(60)
    i.now[0] += 6
    i.odometry.mono[0] += 6
    assert i.gate.update_odometry(i.now[0], Pose(0, 0, frame_id="odom"))
    assert not i.gate.ready()


def test_optional_stationary_age_cap_bounds_total_estimate_age():
    now, mono = [100.0], [0.0]
    policy = LocalizationPolicy(max_stationary_age_s=30)
    gate = LocalizationGate("map", "office", policy, clock=lambda: now[0], monotonic=lambda: mono[0])
    assert gate.update(100, Pose(0, 0, map_id="office"), covariance())
    odometry = Odometry(gate, now, mono)
    odometry.run(29.5)
    assert gate.ready()
    odometry.run(1)
    assert not gate.ready()
    with pytest.raises(ValidationError):
        LocalizationPolicy(max_stationary_age_s=1)
    with pytest.raises(ValidationError):
        LocalizationPolicy(stationary_translation_m=0)


def test_clock_rewind_discards_old_epoch_odometry(idle):
    i = idle
    i.odometry.run(10)
    i.now[0] = 50
    assert not i.gate.ready()
    assert i.gate.update(50, i.pose, covariance())
    assert i.gate.update_odometry(50.1, Pose(3, 0, frame_id="odom"))


def test_odometry_transform_message_feeds_the_gate():
    gate = LocalizationGate("map", "office", clock=lambda: 100)
    message = Obj(
        header=Obj(stamp=Obj(sec=100, nanosec=0), frame_id="odom"),
        transform=Obj(translation=Obj(x=1.0, y=2.0, z=0.0), rotation=Obj(x=0.0, y=0.0, z=0.0, w=1.0)),
    )
    assert update_odometry(gate, message)
    message.header.stamp.sec = 101
    message.transform.rotation.w = 0.0
    assert not update_odometry(gate, message)
    assert not update_odometry(gate, Obj())
