"""Record text commands handled by the real Placecell ROS node and Nav2."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import rclpy
import yaml
from pipeline_test import PipelineProbe
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from video_recorder import VideoRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    output = options.output
    output.mkdir(parents=True, exist_ok=False)
    # Coordinate resolution uses the normal command handler but needs no perception.
    # Keep the camera off the memory node; the recorder subscribes to the actual feeds.
    params = {
        "use_sim_time": True,
        "robot_id": "office_robot",
        "map_id": "office-v1",
        "base_frame": "base_footprint",
        "navigation_enabled": True,
        "localization_required": True,
        "image_topic": "/demo/memory_disabled",
        "db_path": "",
        "keyframe_dir": str(output / "keyframes"),
        "corrections_path": str(output / "corrections.jsonl"),
        "refine_interval_s": 0.0,
        "curator_interval_s": 0.0,
    }
    config = output / "placecell.yaml"
    config.write_text(yaml.safe_dump({"placecell": {"ros__parameters": params}}))
    report = {"mode": "coordinate_commands", "passed": False, "commands": []}
    process, video = None, None
    rclpy.init()
    probe = PipelineProbe()
    probe.overview = None
    probe.create_subscription(
        Image, "/demo/overview", lambda msg: setattr(probe, "overview", msg), qos_profile_sensor_data
    )

    def pause(seconds):
        until = probe.sim_time + seconds
        probe.wait(lambda: probe.sim_time >= until, 30)

    with (output / "placecell.log").open("w") as log:
        try:
            probe.wait(lambda: probe.ready() and probe.overview is not None and bool(probe.rgb), 120)
            process = subprocess.Popen(  # noqa: S603 - fixed node executable and local generated config
                [sys.executable, "-m", "placecell.ros2.node", "--ros-args", "--params-file", str(config)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            probe.wait(lambda: probe.command.get_subscription_count() > 0, 30)
            # Allow the newly started node to receive a fresh AMCL estimate.
            pause(3)
            video = VideoRecorder(output, probe, True, command_demo=True)
            video.memory = "Actual /placecell/command input. Coordinate navigation; visual memory disabled."
            terminal = {
                "succeeded",
                "failed",
                "rejected",
                "not_found",
                "unavailable",
                "invalid",
                "uncertain",
                "canceled",
            }
            for text, x, y in (("go to 1.0, 0.0", 1.0, 0.0), ("go to -1.0, 0.3, 3.14", -1.0, 0.3)):
                video.phase = f'Ready to send: "{text}"'
                pause(2)
                start_distance, status_index = probe.distance, len(probe.statuses)
                sent_at = probe.sim_time
                probe.command.publish(String(data=text))
                video.command, video.command_at = text, sent_at
                video.phase = f'SENT: "{text}"'
                print(f"Published to /placecell/command: {text}", flush=True)  # noqa: T201

                def finished(status_index=status_index):
                    if process.poll() is not None:
                        raise RuntimeError("Placecell exited during the demo")
                    return len(probe.statuses) > status_index and probe.statuses[-1]["state"] in terminal

                probe.wait(finished, 150)
                status = probe.statuses[-1]
                pose = probe.pose()
                if status["state"] != "succeeded" or status["destination"]["source"] != "coordinates":
                    raise RuntimeError(f"Command failed: {status}")
                if pose is None:
                    raise RuntimeError("No localized pose at arrival")
                error = math.hypot(pose.x - x, pose.y - y)
                travelled = probe.distance - start_distance
                if error > 0.15 or travelled < 0.5:
                    raise RuntimeError("Command did not produce the expected physical trip")
                report["commands"].append(
                    {
                        "text": text,
                        "sent_at": sent_at,
                        "result": status,
                        "arrived_at": probe.sim_time,
                        "arrival_pose": {"x": pose.x, "y": pose.y},
                        "position_error_m": error,
                        "travelled_m": travelled,
                    }
                )
                video.phase = f'ARRIVED: "{text}"'
                pause(3)
            report["passed"] = True
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            if video:
                try:
                    video.close(report["passed"])
                except Exception as error:
                    report["video_error"] = str(error)
            if process is not None and process.poll() is None:
                # Stop through the same command input before closing the node.
                if not report["passed"]:
                    probe.command.publish(String(data="stop"))
                    pause(1)
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
            report["statuses"] = probe.statuses
            (output / "report.json").write_text(json.dumps(report, indent=2))
            probe.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
