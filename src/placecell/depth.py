"""Small, durable snapshots of rectified, aligned depth in the RGB optical frame."""

from __future__ import annotations

import base64
import math
import zlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from placecell.errors import ValidationError


@dataclass(frozen=True)
class Box:
    """Normalized RGB bounds, in x/y order."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if not (0 <= self.left < self.right <= 1 and 0 <= self.top < self.bottom <= 1):
            raise ValidationError("object bounds must be nonempty and within the image")

    def overlap(self, other: Box) -> float:
        intersection = max(0, min(self.right, other.right) - max(self.left, other.left)) * max(
            0, min(self.bottom, other.bottom) - max(self.top, other.top)
        )
        area = (self.right - self.left) * (self.bottom - self.top)
        other_area = (other.right - other.left) * (other.bottom - other.top)
        return intersection / (area + other_area - intersection)


@dataclass(frozen=True)
class ObjectPosition:
    x: float
    y: float
    z: float
    uncertainty_m: float
    radius_m: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.z, self.uncertainty_m, self.radius_m)):
            raise ValidationError("object position must be finite")
        if self.uncertainty_m <= 0 or self.radius_m <= 0:
            raise ValidationError("position uncertainty and extent must be positive")

    def distance(self, other: ObjectPosition) -> float:
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


@dataclass(frozen=True)
class DepthSnapshot:
    """Metres of optical-axis depth; map_from_camera is a row-major rigid 4x4 transform.

    Intrinsics belong to this downsampled grid. The JSON-safe payload travels in the
    ingestion journal, so retries never use a later depth frame or a later TF lookup.
    Coordinates estimate a visible surface, not the hidden centre of an object.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    map_from_camera: tuple[float, ...]
    data: str
    position_error_m: float = 0.1
    angular_error_rad: float = 0.05
    image_width: int = 0
    image_height: int = 0

    def __post_init__(self) -> None:
        if not (1 <= self.width <= 320 and 1 <= self.height <= 320):
            raise ValidationError("depth snapshots are limited to 320 pixels per side")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy, self.position_error_m)):
            raise ValidationError("invalid depth calibration")
        if min(self.fx, self.fy, self.position_error_m) <= 0:
            raise ValidationError("depth focal lengths and error must be positive")
        if not 0 <= self.angular_error_rad <= math.pi / 2:
            raise ValidationError("invalid camera angular error")
        if not 0 <= self.cx < self.width or not 0 <= self.cy < self.height:
            raise ValidationError("depth principal point must be inside image")
        transform = np.asarray(self.map_from_camera)
        if transform.shape != (16,) or not np.all(np.isfinite(transform)):
            raise ValidationError("depth requires a finite 4x4 transform")
        matrix = transform.reshape(4, 4)
        rotation = matrix[:3, :3]
        if (
            not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not math.isclose(float(np.linalg.det(rotation)), 1, abs_tol=1e-5)
        ):
            raise ValidationError("camera transform must be rigid")
        self.array()

    @classmethod
    def capture(
        cls,
        depth_m: ArrayLike,
        intrinsics: tuple[float, float, float, float],
        map_from_camera: ArrayLike,
        *,
        position_error_m: float = 0.1,
        angular_error_rad: float = 0.05,
    ) -> DepthSnapshot:
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2 or not depth.size:
            raise ValidationError("aligned depth must be a nonempty 2D array")
        stride = max(1, math.ceil(max(depth.shape) / 320))
        sampled = depth[::stride, ::stride].copy()
        sampled[~np.isfinite(sampled) | (sampled < 0.2) | (sampled > 8)] = 0
        fx, fy, cx, cy = intrinsics
        return cls(
            sampled.shape[1],
            sampled.shape[0],
            fx / stride,
            fy / stride,
            cx / stride,
            cy / stride,
            tuple(float(v) for v in np.asarray(map_from_camera).flatten()),
            base64.b64encode(zlib.compress(sampled.astype("<f4").tobytes())).decode("ascii"),
            position_error_m,
            angular_error_rad,
            depth.shape[1],
            depth.shape[0],
        )

    def array(self) -> NDArray[np.float32]:
        size = self.width * self.height * 4
        if len(self.data) > 600_000:
            raise ValidationError("oversized depth payload")
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(base64.b64decode(self.data, validate=True), size + 1)
            if len(raw) != size or not decoder.eof or decoder.unused_data:
                raise ValueError("invalid depth size")
            result = np.frombuffer(raw, dtype="<f4").reshape(self.height, self.width)
            if not np.all(np.isfinite(result)) or np.any(result < 0) or np.any(result > 8):
                raise ValueError("invalid depth values")
            return result
        except (ValueError, zlib.error) as e:
            raise ValidationError("invalid depth payload") from e

    def _patch(self, box: Box) -> NDArray[np.float32]:
        left, right = int(box.left * self.width), math.ceil(box.right * self.width)
        top, bottom = int(box.top * self.height), math.ceil(box.bottom * self.height)
        return self.array()[top:bottom, left:right]

    def locate(self, box: Box) -> ObjectPosition | None:
        # The central half excludes many background pixels near an imprecise detector boundary.
        dx, dy = (box.right - box.left) / 4, (box.bottom - box.top) / 4
        patch = self._patch(Box(box.left + dx, box.top + dy, box.right - dx, box.bottom - dy))
        valid = patch[patch > 0]
        if len(valid) < 9 or len(valid) < patch.size * 0.8:
            return None
        z = float(np.median(valid))
        spread = float(np.quantile(valid, 0.9) - np.quantile(valid, 0.1))
        radius = (
            max((box.right - box.left) * self.width * z / self.fx, (box.bottom - box.top) * self.height * z / self.fy)
            / 2
        )
        # Compact 3D landmarks can have several surfaces (e.g. a printer body and tray).
        # Bound their depth spread by visible extent, then retain that spread in uncertainty.
        # Large foreground/background discontinuities still produce no position.
        if spread > max(0.15, radius):
            return None
        u, v = (box.left + box.right) * self.width / 2, (box.top + box.bottom) * self.height / 2
        point = np.asarray(self.map_from_camera).reshape(4, 4) @ [
            (u - self.cx) * z / self.fx,
            (v - self.cy) * z / self.fy,
            z,
            1,
        ]
        return ObjectPosition(
            float(point[0]),
            float(point[1]),
            float(point[2]),
            self.position_error_m + spread / 2 + z * (0.02 + math.sin(self.angular_error_rad)),
            max(0.05, radius),
        )

    def clear_region(self, position: ObjectPosition) -> Box | None:
        """A fully framed old extent with depth behind it; foreground/unknown depth means no evidence."""
        camera = np.linalg.inv(np.asarray(self.map_from_camera).reshape(4, 4)) @ [position.x, position.y, position.z, 1]
        x, y, z = (float(v) for v in camera[:3])
        radius = (
            position.radius_m
            + position.uncertainty_m
            + self.position_error_m
            + abs(z) * math.sin(self.angular_error_rad)
        )
        if z <= radius + 0.2 or z + radius >= 8:
            return None
        u, v = (self.fx * x / z + self.cx) / self.width, (self.fy * y / z + self.cy) / self.height
        du, dv = self.fx * radius / (z - radius) / self.width, self.fy * radius / (z - radius) / self.height
        if not (0.02 < u - du < u + du < 0.98 and 0.02 < v - dv < v + dv < 0.98):
            return None
        box = Box(u - du, v - dv, u + du, v + dv)
        patch = self._patch(box)
        # Every sampled ray must have valid background depth. Conservative by design.
        if patch.size < 9 or not np.all(patch > z + radius + 0.15):
            return None
        return box
