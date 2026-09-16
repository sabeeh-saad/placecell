"""Exercise actual Nav2, or the live vision-to-memory-to-Nav2 pipeline in Gazebo."""

import argparse
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from smoke_test import Probe
from std_msgs.msg import String
from tf2_ros import TransformException

from placecell.localization import LocalizationGate
from placecell.ros2.bridge import pose_from_transform, update_localization


class PipelineProbe(Probe):
    def __init__(self):
        super().__init__()
        self.set_parameters([Parameter("use_sim_time", value=True)])
        self.gate = LocalizationGate("map", "office-v1", clock=lambda: self.sim_time)
        self.statuses = []
        self.distance = 0.0
        self.last_xy = None
        self.current_goal = None
        self.client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.lifecycle = self.create_client(GetState, "/bt_navigator/get_state")
        self.lifecycle_request = None
        self.navigation_active = False
        self.command = self.create_publisher(String, "/placecell/command", 1)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.localization, qos_profile_sensor_data)
        self.create_subscription(String, "/placecell/navigation_status", self.status, 100)
        self.create_timer(0.1, self.track)

    def localization(self, message):
        update_localization(self.gate, message, "office-v1")

    def status(self, message):
        value = json.loads(message.data)
        self.statuses.append({"simulation_time": self.sim_time, **value})
        if value["state"] != "navigating" or not any(v["state"] == "navigating" for v in self.statuses[:-1]):
            print(json.dumps(value), flush=True)  # noqa: T201 - test progress

    def track(self):
        if self.odom is None:
            return
        p = self.odom.pose.pose.position
        if self.last_xy is not None:
            self.distance += math.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1])
        self.last_xy = p.x, p.y

    def pose(self):
        try:
            value = self.tf.lookup_transform("map", "base_footprint", Time()).transform
        except TransformException:
            return None
        t, q = value.translation, value.rotation
        return pose_from_transform(t.x, t.y, q.x, q.y, q.z, q.w, "map", "office-v1")

    def ready(self):
        if not self.navigation_active:
            if self.lifecycle_request is not None and self.lifecycle_request.done():
                self.navigation_active = self.lifecycle_request.result().current_state.id == 3
                self.lifecycle_request = None
            if self.lifecycle_request is None and self.lifecycle.service_is_ready():
                self.lifecycle_request = self.lifecycle.call_async(GetState.Request())
        pose = self.pose()
        return (
            self.navigation_active
            and pose is not None
            and self.gate.accepts(pose, self.sim_time)
            and self.client.server_is_ready()
        )

    def navigate(self, x, y, yaw=0.0):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        future = self.client.send_goal_async(goal)
        self.wait(future.done, 15)
        self.current_goal = future.result()
        if not self.current_goal.accepted:
            raise RuntimeError("Nav2 rejected the test route goal")
        result = self.current_goal.get_result_async()
        self.wait(result.done, 150)
        self.current_goal = None
        if result.result().status != 4 or result.result().result.error_code:
            raise RuntimeError(f"Nav2 route failed: {result.result()}")
        reached = self.pose()
        if reached is None or math.hypot(reached.x - x, reached.y - y) > 0.3:
            raise RuntimeError("Nav2 reported success without reaching the requested pose")
        print(f"Nav2 reached ({reached.x:.2f}, {reached.y:.2f})", flush=True)  # noqa: T201
        return {"x": reached.x, "y": reached.y, "yaw": reached.yaw}


def snapshot(database):
    if not database.exists():
        return {"memories": [], "objects": []}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=2) as connection:
        try:
            return {
                "memories": [json.loads(r[0]) for r in connection.execute("SELECT payload FROM memories")],
                "objects": [json.loads(r[0]) for r in connection.execute("SELECT payload FROM objects")],
            }
        except sqlite3.OperationalError:
            return {"memories": [], "objects": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navigation-only", action="store_true", help="Test real AMCL/Nav2 without any model calls")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record-video", action="store_true", help="Record camera and telemetry to walkthrough.avi")
    options = parser.parse_args()
    if not options.navigation_only and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("live mode requires OPENROUTER_API_KEY; it never substitutes mocked perception")
    output = options.output
    output.mkdir(parents=True, exist_ok=False)
    report = {"mode": "navigation" if options.navigation_only else "live_pipeline", "passed": False}
    process = None
    video = None
    rclpy.init()
    probe = PipelineProbe()
    log = (output / "placecell.log").open("w")
    try:
        probe.wait(probe.ready, 120)
        report["localization_ready"] = True
        if options.record_video:
            from video_recorder import VideoRecorder

            video = VideoRecorder(output, probe, options.navigation_only)
            video.phase = "1 / Drive to the observation viewpoint"
        report["observation_pose"] = probe.navigate(1.0, 0.0)
        if not options.navigation_only:
            if video:
                video.phase = "2 / Observe and store the printer"
            root = Path(__file__).resolve().parents[1]
            config = yaml.safe_load((root / "config/placecell.yaml").read_text())
            params = config["placecell"]["ros__parameters"]
            params.update(
                {
                    "db_path": str(output / "db"),
                    "collection": "office",
                    "keyframe_dir": str(output / "keyframes"),
                    "recording_dir": str(output / "recording"),
                    "corrections_path": str(output / "corrections.jsonl"),
                }
            )
            config_path = output / "placecell.yaml"
            config_path.write_text(yaml.safe_dump(config))
            process = subprocess.Popen(  # noqa: S603 - fixed executable/module and generated local config
                [sys.executable, "-m", "placecell.ros2.node", "--ros-args", "--params-file", str(config_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            database = output / "db/office.state.sqlite3"

            def stored_printer():
                if process.poll() is not None:
                    raise RuntimeError("Placecell exited; inspect placecell.log")
                return any(
                    "printer" in o["label"].lower() and o.get("position") and o["position"]["uncertainty_m"] <= 0.35
                    for o in snapshot(database)["objects"]
                )

            probe.wait(stored_printer, 180)
            report["before"] = snapshot(database)
            if video:
                video.memory = (
                    f"Stored {len(report['before']['memories'])} scene memories; printer localized from depth"
                )
            print("Stored printer from live images with depth coordinates", flush=True)  # noqa: T201
        # Face away: the command must recall a previous view, not a freshly seen printer here.
        if video:
            video.phase = "3 / Move away and turn away from the printer"
        report["departure_pose"] = probe.navigate(-1.0, 0.3, math.pi)
        if not options.navigation_only:
            probe.wait(lambda: probe.command.get_subscription_count() > 0, 20)
            distance_before = probe.distance
            if video:
                video.phase = '4 / Command: "go to the printer"'
            probe.command.publish(String(data="go to the printer"))
            terminal = {
                "succeeded",
                "failed",
                "rejected",
                "ambiguous",
                "canceled",
                "destination_unverified",
                "not_found",
                "unavailable",
                "invalid",
                "uncertain",
                "busy",
            }
            probe.wait(lambda: bool(probe.statuses) and probe.statuses[-1]["state"] in terminal, 240)
            final = probe.statuses[-1]
            report["navigation"] = final
            report["semantic_trip_distance_m"] = probe.distance - distance_before
            if final["state"] != "succeeded" or final.get("object_result") != "matched":
                raise RuntimeError(f"Semantic navigation did not verify arrival: {final['state']}")
            if report["semantic_trip_distance_m"] < 1.0:
                raise RuntimeError("Semantic command did not cause a real trip")
            object_id = final["destination"]["object_id"]
            arrival_time = next(s["simulation_time"] for s in probe.statuses if s["state"] == "awaiting_observation")
            probe.wait(
                lambda: any(
                    o["id"] == object_id and o["last_seen"] > arrival_time for o in snapshot(database)["objects"]
                ),
                120,
            )
            report["after"] = snapshot(database)
            report["memory_updated_after_command"] = True
            if video:
                video.phase = "5 / Arrival verified; object memory updated"
                before = next((o for o in report["before"]["objects"] if o["id"] == object_id), None)
                after = next(o for o in report["after"]["objects"] if o["id"] == object_id)
                revision = f"{before['revision']} -> {after['revision']}" if before else str(after["revision"])
                video.memory = (
                    f"Arrival verified; printer revision {revision}; {len(report['after']['memories'])} scene memories"
                )
            report["arrival_pose"] = {"x": probe.pose().x, "y": probe.pose().y}
            previous_statuses, stopped_at = len(probe.statuses), probe.distance
            if video:
                video.phase = '6 / Unknown destination: "purple helicopter"'
            probe.command.publish(String(data="go to the purple helicopter"))
            probe.wait(lambda: len(probe.statuses) > previous_statuses and probe.statuses[-1]["state"] in terminal, 120)
            report["unknown_destination"] = probe.statuses[-1]
            report["unknown_destination_distance_m"] = probe.distance - stopped_at
            if probe.statuses[-1]["state"] != "not_found" or probe.distance - stopped_at > 0.05:
                raise RuntimeError("An unknown destination was not rejected without movement")
        report["total_distance_m"] = probe.distance
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        if video:
            try:
                video.close(report["passed"])
            except Exception as error:
                # A recording failure must not leave the model process or a goal running.
                report["video_error"] = str(error)
        if probe.current_goal is not None:
            canceled = probe.current_goal.cancel_goal_async()
            with suppress(RuntimeError):
                probe.wait(canceled.done, 5)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
        report["statuses"] = probe.statuses
        (output / "report.json").write_text(json.dumps(report, indent=2))
        log.close()
        probe.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
