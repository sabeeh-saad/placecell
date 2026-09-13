"""Conversions between ROS 2 message contents and placecell observations. No rclpy here."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from placecell.errors import PlacecellError, ValidationError
from placecell.memory import Evidence, EvidenceKind, Pose
from placecell.pipeline import Observation

_JPEG_FORMATS = ("jpeg", "jpg")
_CHANNELS = {"mono8": 1, "8UC1": 1, "rgb8": 3, "bgr8": 3, "8UC3": 3, "rgba8": 4, "bgra8": 4, "8UC4": 4}


def stamp_to_seconds(sec: int, nanosec: int) -> float:
    return sec + nanosec * 1e-9


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Heading about +z from a unit quaternion, in radians."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pose_from_transform(
    tx: float, ty: float, qx: float, qy: float, qz: float, qw: float, frame_id: str = "map", map_id: str = ""
) -> Pose:
    """The planar pose of a child frame from a `geometry_msgs/Transform` parent->child."""
    return Pose(tx, ty, yaw_from_quaternion(qx, qy, qz, qw), frame_id, map_id)


class KeyframeWriter:
    """Writes keyframes as JPEG files and returns the evidence that points at them."""

    def __init__(self, out_dir: str | Path, jpeg_quality: int = 85) -> None:
        if not (1 <= jpeg_quality <= 100):
            raise ValidationError("jpeg_quality must be within 1..100")
        self._dir = Path(out_dir).resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._quality = jpeg_quality

    def write_jpeg(self, camera_id: str, timestamp: float, data: bytes) -> Evidence:
        if not data:
            raise ValidationError("empty image data")
        path = self._dir / f"{camera_id}_{round(timestamp * 1000)}.jpg"
        path.write_bytes(data)
        return Evidence(EvidenceKind.FRAME, str(path), hashlib.sha256(data).hexdigest(), managed=True)

    def write_raw(
        self, camera_id: str, timestamp: float, height: int, width: int, encoding: str, step: int, data: bytes
    ) -> Evidence:
        """Encode a `sensor_msgs/Image` buffer as JPEG. Supports 8-bit mono, RGB(A) and BGR(A)."""
        try:
            import cv2
            import numpy as np
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise PlacecellError("raw images need OpenCV: pip install placecell[video]") from e
        channels = _CHANNELS.get(encoding)
        if channels is None:
            raise ValidationError(f"unsupported image encoding {encoding!r}")
        if height < 1 or width < 1 or step < width * channels or len(data) < height * step:
            raise ValidationError("image dimensions do not match the buffer")
        rows = np.frombuffer(data, dtype=np.uint8)[: height * step].reshape(height, step)
        pixels = rows[:, : width * channels].reshape(height, width, channels) if channels > 1 else rows[:, :width]
        conversions = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR, "bgra8": cv2.COLOR_BGRA2BGR}
        bgr: Any = cv2.cvtColor(pixels, conversions[encoding]) if encoding in conversions else pixels
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            raise PlacecellError("JPEG encoding failed")
        return self.write_jpeg(camera_id, timestamp, encoded.tobytes())


class ObservationBuilder:
    """Turns the contents of an image message plus the robot pose into an observation."""

    def __init__(self, robot_id: str, camera_id: str, writer: KeyframeWriter) -> None:
        if not robot_id or not camera_id:
            raise ValidationError("robot_id and camera_id must not be empty")
        self._robot_id = robot_id
        self._camera_id = camera_id
        self._writer = writer

    def from_compressed(self, timestamp: float, image_format: str, data: bytes, pose: Pose) -> Observation:
        """From a `sensor_msgs/CompressedImage`. Only JPEG payloads are accepted as-is."""
        # image_transport writes e.g. "rgb8; jpeg compressed bgr8"; a bare "jpeg" also occurs
        if not any(tag in image_format.lower() for tag in _JPEG_FORMATS):
            raise ValidationError(f"compressed image format {image_format!r} is not JPEG")
        evidence = self._writer.write_jpeg(self._camera_id, timestamp, data)
        return Observation(self._robot_id, self._camera_id, timestamp, pose, evidence)

    def from_raw(
        self, timestamp: float, height: int, width: int, encoding: str, step: int, data: bytes, pose: Pose
    ) -> Observation:
        """From a `sensor_msgs/Image`."""
        evidence = self._writer.write_raw(self._camera_id, timestamp, height, width, encoding, step, data)
        return Observation(self._robot_id, self._camera_id, timestamp, pose, evidence)
