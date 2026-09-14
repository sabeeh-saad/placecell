from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from placecell.depth import Box, DepthSnapshot, ObjectPosition
from placecell.errors import ValidationError
from placecell.ros2.depth import aligned_snapshot

BOX = Box(0.4, 0.4, 0.6, 0.6)


def snapshot(depth=2, transform=None):
    array = np.full((100, 100), depth, dtype=np.float32) if np.isscalar(depth) else depth
    return DepthSnapshot.capture(
        array,
        (100, 100, 50, 50),
        np.eye(4) if transform is None else transform,
        position_error_m=0.02,
        angular_error_rad=0,
    )


def test_projects_optical_depth_through_full_camera_transform():
    # Optical z forward -> map x forward; optical x right -> map -y; optical y down -> map -z.
    matrix = np.array([[0, 0, 1, 3], [-1, 0, 0, 4], [0, -1, 0, 1], [0, 0, 0, 1]])
    position = snapshot(transform=matrix).locate(BOX)
    assert (position.x, position.y, position.z) == (5, 4, 1)
    assert position.uncertainty_m >= 0.06 and position.radius_m == pytest.approx(0.2)


def test_downsampling_preserves_metric_projection_and_limits_payload():
    frame = DepthSnapshot.capture(np.full((600, 800), 2), (800, 800, 400, 300), np.eye(4))
    assert frame.width <= 320 and frame.height <= 320
    assert frame.image_width == 800 and frame.image_height == 600
    location = frame.locate(BOX)
    assert location.z == 2
    assert abs(location.x) < 0.01 and abs(location.y) < 0.01


@pytest.mark.parametrize("depth", [0, -1, float("nan"), float("inf"), 9])
def test_unknown_depth_does_not_locate_or_prove_absence(depth):
    frame = snapshot(depth)
    assert frame.locate(BOX) is None
    assert frame.clear_region(ObjectPosition(0, 0, 2, 0.05, 0.1)) is None


def test_mixed_depth_patch_has_no_trustworthy_location():
    array = np.full((100, 100), 2.0)
    array[:, 50:] = 3
    assert snapshot(array).locate(BOX) is None
    assert snapshot().locate(Box(0.49, 0.49, 0.5, 0.5)) is None


def test_visibility_requires_every_ray_valid_and_background_beyond_old_extent():
    position = snapshot().locate(BOX)
    assert snapshot(5).clear_region(position) is not None
    assert snapshot(1).clear_region(position) is None
    array = np.full((100, 100), 5.0)
    array[50, 50] = 0
    assert snapshot(array).clear_region(position) is None
    assert snapshot(5).clear_region(replace(position, x=10)) is None
    assert snapshot(5).clear_region(replace(position, z=-2)) is None
    assert snapshot(5).clear_region(replace(position, z=8)) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"fx": 0},
        {"cx": 101},
        {"fy": float("nan")},
        {"width": 0},
        {"angular_error_rad": -1},
        {"map_from_camera": (1,)},
        {"map_from_camera": tuple(np.zeros(16))},
        {"data": "no"},
        {"data": "x" * 600_001},
    ],
)
def test_rejects_invalid_calibration_transform_and_payload(changes):
    with pytest.raises(ValidationError):
        replace(snapshot(), **changes)


@pytest.mark.parametrize("bounds", [(-0.1, 0, 1, 1), (0, 0, 0, 1), (0, 0, float("nan"), 1)])
def test_rejects_invalid_boxes(bounds):
    with pytest.raises(ValidationError):
        Box(*bounds)


def messages(encoding="16UC1", endian=False, pad=0):
    dtype = (">" if endian else "<") + ("u2" if encoding == "16UC1" else "f4")
    value = 2000 if encoding == "16UC1" else 2.0
    row = np.full(100, value, dtype=dtype).tobytes() + b"\0" * pad
    header = SimpleNamespace(frame_id="camera_optical", stamp=SimpleNamespace(sec=10, nanosec=0))
    depth = SimpleNamespace(
        header=header, width=100, height=100, encoding=encoding, step=len(row), is_bigendian=endian, data=row * 100
    )
    info = SimpleNamespace(
        header=header,
        width=100,
        height=100,
        d=[0] * 5,
        r=np.eye(3).flatten(),
        k=[100, 0, 50, 0, 100, 50, 0, 0, 1],
        binning_x=0,
        binning_y=0,
        roi=SimpleNamespace(x_offset=0, y_offset=0),
    )
    transform = SimpleNamespace(
        translation=SimpleNamespace(x=1, y=2, z=3), rotation=SimpleNamespace(x=0, y=0, z=0, w=1)
    )
    return depth, info, transform


@pytest.mark.parametrize("encoding,endian,pad", [("16UC1", True, 4), ("32FC1", False, 8)])
def test_ros_units_endianness_padding_and_extrinsics(encoding, endian, pad):
    frame = aligned_snapshot(*messages(encoding, endian, pad), rgb_stamp=10, rgb_frame="camera_optical")
    location = frame.locate(BOX)
    assert (location.x, location.y, location.z) == (1, 2, 5)


@pytest.mark.parametrize(
    "change", ["time", "frame", "dimensions", "distortion", "binning", "encoding", "stride", "quaternion"]
)
def test_ros_misaligned_inputs_are_rejected(change):
    depth, info, transform = messages()
    if change == "time":
        depth.header = SimpleNamespace(frame_id="camera_optical", stamp=SimpleNamespace(sec=9, nanosec=0))
    elif change == "frame":
        info.header = SimpleNamespace(frame_id="another_camera", stamp=SimpleNamespace(sec=10, nanosec=0))
    elif change == "dimensions":
        info.width = 99
    elif change == "distortion":
        info.d = [0.01]
    elif change == "binning":
        info.binning_x = 2
    elif change == "encoding":
        depth.encoding = "mono8"
    elif change == "stride":
        depth.step = 10
    elif change == "quaternion":
        transform.rotation.w = 2
    with pytest.raises(ValidationError):
        aligned_snapshot(depth, info, transform, rgb_stamp=10, rgb_frame="camera_optical")
