from __future__ import annotations

from types import SimpleNamespace as Obj

import pytest

from placecell import Pose
from placecell.errors import ValidationError
from placecell.localization import LocalizationGate, LocalizationPolicy
from placecell.ros2.bridge import update_localization


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
