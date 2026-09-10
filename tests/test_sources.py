from __future__ import annotations

import math
from pathlib import Path

import pytest

from placecell import Pose
from placecell.errors import PlacecellError, ValidationError
from placecell.sources import PoseTrack


def test_pose_track_interpolates_position_and_shortest_heading() -> None:
    track = PoseTrack([(10.0, Pose(0, 0, math.pi - 0.1)), (0.0, Pose(0, 0, 0)), (20.0, Pose(10, 0, -math.pi + 0.1))])
    assert len(track) == 3 and track.span == (0.0, 20.0)
    mid = track.at(15.0)
    assert mid.x == pytest.approx(5.0) and mid.y == 0
    assert abs((mid.yaw - math.pi + math.pi) % (2 * math.pi) - math.pi) == pytest.approx(0.0, abs=1e-9)
    assert track.at(0.0) == Pose(0, 0, 0) and track.at(20.0).x == 10
    assert track.at(-0.4) == Pose(0, 0, 0) and track.at(20.4).x == 10  # within tolerance, clamped
    with pytest.raises(ValidationError):
        track.at(21.0)
    with pytest.raises(ValidationError):
        PoseTrack([])
    with pytest.raises(ValidationError):
        PoseTrack([(0.0, Pose(0, 0)), (1.0, Pose(0, 0, frame_id="odom"))])


def test_pose_track_from_csv(tmp_path: Path) -> None:
    csv = tmp_path / "poses.csv"
    csv.write_text("timestamp,x,y,yaw,frame_id,map_id\n0,0,0,0,map,office\n2,2,0,0,map,office\n")
    track = PoseTrack.from_csv(csv)
    assert track.at(1.0) == Pose(1, 0, 0, "map", "office")
    (tmp_path / "bad.csv").write_text("t,x\n1,2\n")
    with pytest.raises(ValidationError):
        PoseTrack.from_csv(tmp_path / "bad.csv")


def test_video_source_yields_keyframes_with_interpolated_poses(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from placecell import EvidenceKind
    from placecell.sources.video import iter_video_observations

    video = tmp_path / "walk.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 12, dtype=np.uint8))
    writer.release()
    track = PoseTrack([(100.0, Pose(0, 0)), (102.0, Pose(2, 0))])

    observations = list(iter_video_observations(video, track, "r1", "front", 100.0, tmp_path / "frames", every_s=0.5))

    assert [o.timestamp for o in observations] == [100.0, 100.5, 101.0, 101.5]
    assert [round(o.pose.x, 2) for o in observations] == [0.0, 0.5, 1.0, 1.5]
    assert all(o.evidence.kind is EvidenceKind.FRAME and Path(o.evidence.uri).is_file() for o in observations)
    assert len({o.evidence.digest for o in observations}) == 4
    assert observations[0].evidence.uri.endswith("front_100000.jpg")
    assert all(o.robot_id == "r1" and o.camera_id == "front" for o in observations)


def test_video_source_rejects_missing_files_and_bad_intervals(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    from placecell.sources.video import iter_video_observations

    track = PoseTrack([(0.0, Pose(0, 0))])
    with pytest.raises(PlacecellError):
        next(iter_video_observations(tmp_path / "missing.avi", track, "r", "c", 0.0, tmp_path))
    with pytest.raises(ValidationError):
        next(iter_video_observations(tmp_path / "missing.avi", track, "r", "c", 0.0, tmp_path, every_s=0))
