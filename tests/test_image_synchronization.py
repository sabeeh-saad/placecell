from types import SimpleNamespace

import pytest

from placecell.errors import ValidationError
from placecell.ros2.depth import PendingImages


def message(stamp, frame="camera"):
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp, nanosec=0), frame_id=frame))


def test_rgb_can_arrive_before_depth_and_calibration_without_losing_position():
    pending = PendingImages()
    rgb = message(10)
    pending.add(rgb, False, 1.0)
    assert pending.pop(1.05) is None
    pending.depth.append(message(10))
    assert pending.pop(1.1) is None
    pending.info.append(message(10))
    assert pending.pop(1.15) == (rgb, False)
    assert pending.pop(1.2) is None


def test_wrong_capture_or_frame_waits_then_falls_back_without_fabricating_depth():
    pending = PendingImages()
    rgb = message(10)
    pending.depth.append(message(9))
    pending.info.append(message(10, "other_camera"))
    pending.add(rgb, True, 1.0)
    assert pending.pop(1.1) is None
    assert pending.pop(1.31) == (rgb, True)


def test_longer_transport_wait_still_requires_the_matching_capture():
    pending = PendingImages(wait_s=1.0)
    rgb = message(10)
    pending.add(rgb, False, 1.0)
    pending.depth.append(message(9))
    pending.info.append(message(10))
    assert pending.pop(1.8) is None
    pending.depth.append(message(10))
    assert pending.pop(1.9) == (rgb, False)


def test_queue_is_bounded_and_rejects_out_of_order_or_duplicate_images():
    pending = PendingImages(capacity=2)
    a, b, c = message(10), message(11), message(12)
    for image in (a, b, c, b):
        pending.add(image, False, 0)
    assert pending.pop(1) == (b, False)
    pending.add(a, False, 1)
    assert pending.pop(1) == (c, False)
    pending.add(c, False, 1)
    assert pending.pop(2) is None


def test_static_calibration_and_depth_before_rgb_work():
    pending = PendingImages()
    pending.depth.append(message(10))
    pending.info.append(message(0))
    rgb = message(10)
    pending.add(rgb, False, 1)
    assert pending.pop(1) == (rgb, False)


def test_dropped_depth_frame_does_not_hide_a_newer_complete_capture():
    pending = PendingImages(wait_s=1)
    old, current = message(10), message(11)
    pending.add(old, False, 0)
    pending.add(current, False, 0.2)
    pending.add_depth(message(11))
    pending.add_depth(message(11), calibration=True)
    assert pending.pop(0.5) is None  # Still allow the earlier pair to arrive.
    assert pending.pop(1.1) == (current, False)
    assert pending.pop(2) is None


def test_no_newer_complete_capture_keeps_bounded_scene_only_fallback():
    pending = PendingImages(wait_s=1)
    old, current = message(10), message(11)
    pending.add(old, False, 0)
    pending.add(current, False, 0.2)
    pending.add_depth(message(11))  # Calibration is missing for both.
    assert pending.pop(1.1) == (old, False)


@pytest.mark.parametrize("options", [{"wait_s": -1}, {"max_skew_s": float("nan")}, {"capacity": 0}])
def test_invalid_synchronization_bounds_are_rejected(options):
    with pytest.raises(ValidationError):
        PendingImages(**options)
