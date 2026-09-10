"""Observations from a recorded video plus a pose track, for building a memory without a robot.

Needs OpenCV: `pip install placecell[video]`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from placecell.errors import PlacecellError, ValidationError
from placecell.memory import Evidence, EvidenceKind
from placecell.pipeline import Observation
from placecell.sources.pose_log import PoseTrack


def iter_video_observations(
    video_path: str | Path,
    track: PoseTrack,
    robot_id: str,
    camera_id: str,
    start_time: float,
    out_dir: str | Path,
    every_s: float = 1.0,
    jpeg_quality: int = 85,
) -> Iterator[Observation]:
    """Yield one observation every `every_s` seconds of video, writing the keyframe as a JPEG.

    `start_time` is the unix time at which the video begins; it aligns frames with the track.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise PlacecellError("video sources need OpenCV: pip install placecell[video]") from e
    if every_s <= 0:
        raise ValidationError("every_s must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PlacecellError(f"cannot open video {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            raise PlacecellError(f"video {video_path} reports no frame rate")
        step = max(1, round(every_s * fps))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            if index % step == 0:
                timestamp = start_time + index / fps
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if not ok:
                    raise PlacecellError(f"cannot encode frame {index} of {video_path}")
                data = encoded.tobytes()
                path = out / f"{camera_id}_{round(timestamp * 1000)}.jpg"
                path.write_bytes(data)
                evidence = Evidence(EvidenceKind.FRAME, str(path), hashlib.sha256(data).hexdigest())
                yield Observation(robot_id, camera_id, timestamp, track.at(timestamp), evidence)
            index += 1
    finally:
        capture.release()
