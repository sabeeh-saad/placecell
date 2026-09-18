"""Aligned RGB-D message validation; independent of rclpy for replay and tests."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from placecell.depth import DepthSnapshot
from placecell.errors import ValidationError
from placecell.ros2.bridge import stamp_to_seconds


class PendingImages:
    """Bounded wait for depth/calibration callbacks that may arrive after RGB.

    Pop in capture order, falling back to a scene-only observation after the wall-time
    deadline. Missing depth never receives a fabricated position. Call from one callback group.
    """

    def __init__(self, max_skew_s: float = 0.08, wait_s: float = 0.3, capacity: int = 8) -> None:
        if not math.isfinite(max_skew_s) or not math.isfinite(wait_s) or min(max_skew_s, wait_s, capacity) <= 0:
            raise ValidationError("invalid image synchronization bounds")
        self.depth: deque[Any] = deque(maxlen=capacity)
        self.info: deque[Any] = deque(maxlen=capacity)
        self._images: deque[tuple[Any, bool, float]] = deque(maxlen=capacity)
        self._skew, self._wait = max_skew_s, wait_s
        self._last_stamp = -math.inf

    @staticmethod
    def stamp(message: Any) -> float:
        return stamp_to_seconds(message.header.stamp.sec, message.header.stamp.nanosec)

    def add(self, message: Any, compressed: bool, now: float) -> None:
        timestamp = self.stamp(message)
        if timestamp <= self._last_stamp or (self._images and timestamp <= self.stamp(self._images[-1][0])):
            return
        self._images.append((message, compressed, now))

    def pop(self, now: float) -> tuple[Any, bool] | None:
        if not self._images:
            return None
        message, compressed, received = self._images[0]
        timestamp = self.stamp(message)
        frame = message.header.frame_id
        depth_ready = any(
            d.header.frame_id == frame and abs(self.stamp(d) - timestamp) <= self._skew for d in self.depth
        )
        info_ready = any(
            i.header.frame_id == frame and (self.stamp(i) == 0 or abs(self.stamp(i) - timestamp) <= self._skew)
            for i in self.info
        )
        if not (depth_ready and info_ready) and 0 <= now - received < self._wait:
            return None
        self._images.popleft()
        self._last_stamp = timestamp
        return message, compressed


def aligned_snapshot(
    depth: Any,
    info: Any,
    transform: Any,
    *,
    rgb_stamp: float,
    rgb_frame: str,
    max_skew_s: float = 0.08,
    position_error_m: float = 0.1,
    angular_error_rad: float = 0.05,
) -> DepthSnapshot:
    """Accept only rectified, calibrated depth aligned to the RGB optical frame.

    TF must map that optical frame into the current map at the RGB timestamp. 16UC1
    is millimetres and 32FC1 is metres, including ROS row padding and endianness.
    """
    if not math.isfinite(max_skew_s) or max_skew_s <= 0:
        raise ValidationError("depth skew must be finite and positive")
    depth_stamp = stamp_to_seconds(depth.header.stamp.sec, depth.header.stamp.nanosec)
    info_stamp = stamp_to_seconds(info.header.stamp.sec, info.header.stamp.nanosec)
    if (
        not rgb_frame
        or depth.header.frame_id != rgb_frame
        or info.header.frame_id != rgb_frame
        or abs(rgb_stamp - depth_stamp) > max_skew_s
        or (info_stamp != 0 and abs(rgb_stamp - info_stamp) > max_skew_s)
    ):
        raise ValidationError("RGB, depth and CameraInfo must share optical frame and capture time")
    if depth.width != info.width or depth.height != info.height or min(depth.width, depth.height) < 1:
        raise ValidationError("aligned depth and calibration dimensions differ")
    if any(not math.isfinite(v) or abs(float(v)) > 1e-8 for v in info.d) or not np.allclose(
        np.asarray(info.r).reshape(3, 3), np.eye(3)
    ):
        raise ValidationError("object depth requires rectified RGB with zero-distortion CameraInfo")
    if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
        raise ValidationError("object depth does not accept cropped or binned CameraInfo")
    if depth.encoding not in {"16UC1", "32FC1"}:
        raise ValidationError("depth encoding must be 16UC1 or 32FC1")
    size = 2 if depth.encoding == "16UC1" else 4
    if (
        depth.width * depth.height > 4_000_000
        or depth.step < depth.width * size
        or depth.step * depth.height != len(depth.data)
    ):
        raise ValidationError("invalid depth buffer size or stride")
    dtype = (">" if depth.is_bigendian else "<") + ("u2" if size == 2 else "f4")
    array: NDArray[np.float32] = np.ndarray(
        (depth.height, depth.width), dtype=dtype, buffer=bytes(depth.data), strides=(depth.step, size)
    ).astype(np.float32)
    if size == 2:
        array /= 1000
    q, t = transform.rotation, transform.translation
    x, y, z, w = (float(v) for v in (q.x, q.y, q.z, q.w))
    if not math.isclose(x * x + y * y + z * z + w * w, 1, abs_tol=1e-5):
        raise ValidationError("camera quaternion must be normalized")
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), t.x],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), t.y],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), t.z],
            [0, 0, 0, 1],
        ]
    )
    return DepthSnapshot.capture(
        array,
        (info.k[0], info.k[4], info.k[2], info.k[5]),
        matrix,
        position_error_m=position_error_m,
        angular_error_rad=angular_error_rad,
    )
