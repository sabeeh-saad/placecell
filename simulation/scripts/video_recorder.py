"""Record actual ROS camera frames and telemetry alongside the simulation check."""

import json
import math
import textwrap
from pathlib import Path

import cv2
import numpy as np
from make_map import rasterize


class VideoRecorder:
    """Write a silent, simulation-time video plus an auditable frame manifest."""

    def __init__(self, output, probe, navigation_only):
        self.probe = probe
        self.navigation_only = navigation_only
        self.phase = "Starting navigation check"
        self.memory = "No live model calls" if navigation_only else "Waiting for the first stored observation"
        self.writer = cv2.VideoWriter(str(output / "walkthrough.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 5, (1280, 720))
        if not self.writer.isOpened():
            raise RuntimeError("Could not open the simulation video writer")
        self.manifest = (output / "video_frames.jsonl").open("w")
        self.frames = 0
        self.route = []
        self.last_frame = None
        pixels = rasterize(Path(__file__).resolve().parents[1] / "worlds/office.sdf")
        self.map = np.zeros((*pixels.shape, 3), dtype=np.uint8)
        self.map[pixels == 254] = (53, 42, 31)
        self.map[pixels == 205] = (31, 24, 18)
        self.map[pixels == 0] = (150, 139, 125)
        self.timer = probe.create_timer(0.2, self.capture)

    @staticmethod
    def text(frame, value, xy, scale=0.6, color=(230, 235, 241)):
        cv2.putText(frame, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    @staticmethod
    def map_xy(x, y):
        return round((x + 6) / 0.025), round((5 - y) / 0.025)

    def capture(self):
        if not self.probe.rgb:
            return
        rgb = self.probe.rgb[-1]
        if rgb.encoding != "rgb8":
            raise RuntimeError(f"Unsupported video camera encoding: {rgb.encoding}")
        rows = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.step)
        camera = rows[:, : rgb.width * 3].reshape(rgb.height, rgb.width, 3)
        frame = np.full((720, 1280, 3), (26, 18, 12), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (1280, 5), (213, 190, 48), -1)
        self.text(frame, "PLACECELL", (28, 44), 0.95)
        title = "Gazebo / Nav2 route recording" if self.navigation_only else "Gazebo / visual memory navigation"
        self.text(frame, title, (300, 43), 0.73)
        mode = "NAVIGATION ONLY - NO MODEL CALLS" if self.navigation_only else "LIVE HOSTED MODELS / TEXT COMMAND"
        self.text(frame, mode, (28, 77), 0.52, (213, 190, 48))
        self.text(frame, f"Simulation time {self.probe.sim_time:6.1f}s", (923, 77), 0.53)
        self.text(frame, "ROBOT CAMERA / RGB-D SENSOR", (28, 117), 0.57)
        frame[134:614, 28:668] = cv2.resize(cv2.cvtColor(camera, cv2.COLOR_RGB2BGR), (640, 480))
        self.text(frame, "MAP + LOCALIZED ROBOT POSE", (744, 117), 0.57)
        map_frame = self.map.copy()
        pose = self.probe.pose()
        if pose is not None:
            point = self.map_xy(pose.x, pose.y)
            if not self.route or self.route[-1] != point:
                self.route.append(point)
            if len(self.route) > 1:
                cv2.polylines(map_frame, [np.array(self.route, dtype=np.int32)], False, (213, 190, 48), 2)
            cv2.circle(map_frame, point, 8, (213, 190, 48), -1)
            tip = self.map_xy(pose.x + 0.5 * math.cos(pose.yaw), pose.y + 0.5 * math.sin(pose.yaw))
            cv2.arrowedLine(map_frame, point, tip, (255, 255, 255), 2, tipLength=0.4)
        status = self.probe.statuses[-1] if self.probe.statuses else {}
        destination = status.get("destination") or {}
        if "x" in destination and "y" in destination:
            cv2.drawMarker(
                map_frame, self.map_xy(destination["x"], destination["y"]), (90, 195, 255), cv2.MARKER_CROSS, 16, 2
            )
        frame[134:534, 744:1224] = map_frame
        if pose is not None:
            self.text(frame, f"x {pose.x:+.2f}m    y {pose.y:+.2f}m", (744, 562), 0.62)
        self.text(frame, f"Travelled {self.probe.distance:.2f}m", (744, 591), 0.6, (213, 190, 48))
        self.text(frame, self.phase, (28, 649), 0.73)
        state = status.get("state", "route test")
        self.text(frame, f"Status: {state}", (744, 623), 0.55)
        for index, line in enumerate(textwrap.wrap(self.memory, 65)[:2]):
            self.text(frame, line, (28, 677 + index * 23), 0.51, (176, 181, 191))
        self.text(frame, "5 fps / simulation-time playback", (744, 678), 0.49, (176, 181, 191))
        self.text(frame, "Text input; microphone not tested", (744, 701), 0.49, (176, 181, 191))
        self.writer.write(frame)
        self.last_frame = frame
        self.manifest.write(
            json.dumps(
                {
                    "frame": self.frames,
                    "simulation_time": self.probe.sim_time,
                    "camera_time": rgb.header.stamp.sec + rgb.header.stamp.nanosec / 1e9,
                    "phase": self.phase,
                    "status": state,
                    "distance_m": self.probe.distance,
                    "pose": None if pose is None else {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
                }
            )
            + "\n"
        )
        self.frames += 1

    def close(self, passed):
        self.probe.destroy_timer(self.timer)
        try:
            self.phase = "CHECK PASSED" if passed else "CHECK FAILED - see report.json"
            self.capture()
            # A clearly labelled final still gives viewers time to read the result.
            if self.last_frame is not None:
                self.text(self.last_frame, "FINAL STILL", (1065, 43), 0.5)
                for _ in range(15):
                    self.writer.write(self.last_frame)
        finally:
            self.writer.release()
            self.manifest.close()
