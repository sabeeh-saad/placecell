"""Gazebo mission/fault checkpoint with deterministic provider fixtures.

The production node, storage, command journal, RGB-D/TF/AMCL, planning environment
and Nav2 transport remain real. Only language/vision/embedding providers are fixtures.
"""

import argparse
import copy
import json
import math
import sqlite3
import subprocess
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import rclpy
import yaml
from checkpoint_fixtures import CaptionFixture, DetectorFixture, PixelFixture, PlanFixture, VisionFixture
from geometry_msgs.msg import PoseWithCovarianceStamped
from pipeline_test import PipelineProbe, snapshot
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from placecell import MissionPlanner, PlanReviewAgent
from placecell.providers._http import UrllibTransport
from placecell.ros2.node import create_node
from placecell.tracing import read_trace

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {
    "succeeded",
    "failed",
    "rejected",
    "ambiguous",
    "canceled",
    "destination_unverified",
    "destination_ambiguous",
    "not_found",
    "unavailable",
    "invalid",
    "uncertain",
}
CASES = (
    "ordered_duplicate",
    "single_object",
    "cancel_motion",
    "cancel_planning",
    "invalid_model",
    "depth_loss",
    "camera_loss",
    "removed",
    "moved",
    "occluded",
    "lookalike",
)


class World:
    @staticmethod
    def service(name, kind, request, *, required=True):
        result = subprocess.run(  # noqa: S603 - fixed Gazebo CLI, authored test world only
            [  # noqa: S607 - executable provided by the pinned ROS/Gazebo image
                "gz",
                "service",
                "-s",
                f"/world/office/{name}",
                "--reqtype",
                f"gz.msgs.{kind}",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "5000",
                "--req",
                request,
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if required and (result.returncode or "data: true" not in result.stdout):
            raise RuntimeError(f"Gazebo {name} failed: {result.stdout} {result.stderr}")

    def pose(self, y=0.0, z=0.78):
        self.service(
            "set_pose",
            "Pose",
            f'name:"printer" position:{{x:3 y:{y} z:{z}}} orientation:{{z:-0.7071067812 w:0.7071067812}}',
        )

    def remove(self, name):
        self.service("remove", "Entity", f'name:"{name}" type:MODEL', required=False)

    def reset(self):
        self.remove("checkpoint_occluder")
        self.remove("checkpoint_lookalike")
        self.pose()

    def change(self, case):
        if case == "removed":
            self.pose(z=-10)  # Remove the target from all camera/depth views; keep it restorable.
        elif case == "moved":
            self.pose(y=0.8)
        elif case == "occluded":
            sdf = (
                '<sdf version="1.9"><model name="checkpoint_occluder"><static>true</static>'
                '<pose>2.4 0 1.1 0 0 0</pose><link name="body"><visual name="screen">'
                "<geometry><box><size>0.15 1.2 1.1</size></box></geometry>"
                "<material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>"
                '<collision name="screen"><geometry><box><size>0.15 1.2 1.1</size></box></geometry>'
                "</collision></link></model></sdf>"
            )
            self.service("create", "EntityFactory", "sdf:" + json.dumps(sdf))
        elif case == "lookalike":
            printer = copy.deepcopy(ET.parse(ROOT / "worlds/office.sdf").find(".//model[@name='printer']"))  # noqa: S314 - authored local world
            printer.set("name", "checkpoint_lookalike")
            printer.find("pose").text = "3 0.85 0.78 0 0 -1.5707963268"
            sdf = '<sdf version="1.9">' + ET.tostring(printer, encoding="unicode") + "</sdf>"
            self.service("create", "EntityFactory", "sdf:" + json.dumps(sdf))


class Probe(PipelineProbe):
    def __init__(self):
        super().__init__()
        self.forward = {"rgb": True, "depth": True, "info": True, "localization": True}
        self.receipts = []
        self.json_command = self.create_publisher(String, "/placecell/command_json", 10)
        self.create_subscription(
            String, "/placecell/command_receipt", lambda m: self.receipts.append(json.loads(m.data)), 10
        )
        for key, topic, message_type in (
            ("rgb", "/camera/color/image_raw", Image),
            ("depth", "/camera/aligned_depth_to_color/image_raw", Image),
            ("info", "/camera/color/camera_info", CameraInfo),
            ("localization", "/amcl_pose", PoseWithCovarianceStamped),
        ):
            pub = self.create_publisher(message_type, f"/checkpoint/{key}", qos_profile_sensor_data)
            self.create_subscription(
                message_type,
                topic,
                lambda m, key=key, pub=pub: pub.publish(m) if self.forward[key] else None,
                qos_profile_sensor_data,
            )

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        self.wait(lambda: time.monotonic() >= deadline, seconds + 5)

    def submit(self, text, identity):
        command = {
            "schema_version": 2,
            "command_id": identity,
            "scope": {"robot_id": "office_robot", "map_id": "office-v1", "conversation_id": "checkpoint"},
            "issued_at_unix_s": time.time(),
            "command": "instruction",
            "text": text,
        }
        self.json_command.publish(String(data=json.dumps(command)))
        return command

    def picture(self, path):
        from PIL import Image as PillowImage

        image = self.rgb[-1]
        PillowImage.frombytes("RGB", (image.width, image.height), bytes(image.data)).save(path)


def run_case(probe, world, case, output, *, isolate_arrival=False):
    output.mkdir(parents=True, exist_ok=False)
    world.reset()
    probe.forward = dict.fromkeys(probe.forward, True)
    probe.wait(probe.ready, 60)
    probe.navigate(1.0, 0.0)
    config = yaml.safe_load((ROOT / "config/placecell.yaml").read_text())
    p = config["placecell"]["ros__parameters"]
    p.update(
        db_path=str(output / "db"),
        collection="checkpoint",
        keyframe_dir=str(output / "keyframes"),
        corrections_path=str(output / "corrections.jsonl"),
        mission_enabled=True,
        mission_model="fixture",
        mission_context_path=str(output / "missions.sqlite3"),
        command_journal_path=str(output / "commands.sqlite3"),
        mission_trace_path=str(output / "traces.sqlite3"),
        mission_conversation_id="checkpoint",
        navigation_lookup_timeout_s=30.0,
        image_topic="/checkpoint/rgb",
        depth_topic="/checkpoint/depth",
        camera_info_topic="/checkpoint/info",
        localization_topic="/checkpoint/localization",
        object_min_interval_s=2.0,
        min_interval_s=2.0,
        max_interval_s=4.0,
        object_search_enabled=False,
    )
    places = output / "places.json"
    places.write_text(
        json.dumps(
            {
                "home": {"x": -1.0, "y": 0.3, "yaw": math.pi, "map_id": "office-v1"},
                "view": {"x": 1.0, "y": 0.0, "yaw": 0.0, "map_id": "office-v1"},
            }
        )
    )
    p["places_file"] = str(places)
    # Set this node's parameters without resetting the shared ROS/Gazebo clock.
    from rclpy.node import Node
    from rclpy.parameter import Parameter

    original_init = Node.__init__

    def initialize(node, name, *args, **kwargs):
        if name == "placecell":
            kwargs["parameter_overrides"] = [Parameter(k, value=v) for k, v in p.items()]
        original_init(node, name, *args, **kwargs)

    (output / "parameters.yaml").write_text(yaml.safe_dump(config))
    model, reviewer = PlanFixture(), PlanFixture(True)
    report = {
        "case_id": case,
        "passed": False,
        "provider_mode": "deterministic pixel/language fixtures",
        "isolate_arrival": isolate_arrival,
    }
    node = executor = thread = None
    dispatches = []
    started_status = len(probe.statuses)
    try:
        with ExitStack() as stack:
            for target, value in (
                ("placecell.ros2.node.build_embedder", lambda *a, **k: PixelFixture()),
                ("placecell.providers.OpenAICompatibleCaptioner", CaptionFixture),
                ("placecell.providers.object_detection.ChatObjectDetector", DetectorFixture),
                ("placecell.ros2.node.VisionVerifier", VisionFixture),
                (
                    "placecell.ros2.node.build_mission_planner",
                    lambda *a: MissionPlanner(model, PlanReviewAgent(reviewer)),
                ),
                ("rclpy.node.Node.__init__", initialize),
            ):
                stack.enter_context(patch(target, value))
            node = create_node()
        original_send = node._navigator._client.send_goal_async

        def send(goal, **kwargs):
            pose = goal.pose.pose.position
            dispatches.append({"x": pose.x, "y": pose.y, "monotonic_s": time.monotonic()})
            return original_send(goal, **kwargs)

        node._navigator._client.send_goal_async = send
        executor = MultiThreadedExecutor(4)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        database = output / "db/checkpoint.state.sqlite3"
        object_case = case in {
            "single_object",
            "depth_loss",
            "camera_loss",
            "removed",
            "moved",
            "occluded",
            "lookalike",
        }
        probe.wait(lambda: node._localization.ready() and node._sensors.ready(camera=True, depth=True), 30)
        if object_case:
            probe.wait(
                lambda: any(o.get("position") and o["status"] == "present" for o in snapshot(database)["objects"]), 60
            )
            report["before"] = snapshot(database)
            if isolate_arrival:
                # Preserve real scene ingestion and independent fresh arrival detection,
                # but defer background object refresh until this short trial has ended.
                # This profile cannot qualify continuous object identity association.
                tracker = node._worker._ingester._objects
                tracker.policy = replace(tracker.policy, min_interval_s=3600)
                report["background_object_refresh_interval_after_learning_s"] = 3600
            probe.picture(output / "learned-camera.png")
            probe.navigate(-1.0, 0.3, math.pi)
        probe.wait(lambda: probe.json_command.get_subscription_count() > 0 and node._localization.ready(), 30)
        start_distance = probe.distance
        if case == "cancel_planning":
            model.release.clear()
        model.malformed = case == "invalid_model"
        instruction = "go to printer" if object_case else "go to home then view"
        command = probe.submit(instruction, "checkpoint-" + case)
        if case == "ordered_duplicate":
            probe.wait(lambda: any(r.get("command_id") == command["command_id"] for r in probe.receipts), 10)
            probe.json_command.publish(String(data=json.dumps(command)))
            probe.wait(
                lambda: any(
                    r.get("command_id") == command["command_id"] and r["disposition"] == "duplicate"
                    for r in probe.receipts
                ),
                10,
            )
        elif case == "cancel_planning":
            probe.wait(model.blocked.is_set, 10)
            probe.command.publish(String(data="stop"))
            probe.wait(lambda: node._commands.snapshot().status.state == "canceled", 10)
            model.release.set()
            probe.spin_for(1)
        elif case not in {"single_object", "invalid_model"}:
            probe.wait(lambda: node._commands.snapshot().status.state in {"navigating", *TERMINAL}, 60)
            assert node._commands.snapshot().status.state == "navigating", node._commands.snapshot().status
            if case == "cancel_motion":
                probe.wait(lambda: probe.distance - start_distance > 0.15, 30)
                probe.command.publish(String(data="stop"))
            elif case in {"depth_loss", "camera_loss"}:
                probe.wait(lambda: probe.distance - start_distance > 0.10, 30)
                probe.forward["depth" if case == "depth_loss" else "rgb"] = False
            else:
                world.change(case)
                report["world_change_after_dispatch"] = case
        probe.wait(lambda: node._commands.snapshot().status.state in TERMINAL, 150)
        probe.spin_for(0.4)
        final = node._commands.snapshot().status
        report.update(
            terminal_state=final.state,
            object_result=final.object_result,
            failure_stage=final.failure_stage,
            message=final.message,
            mission_distance_m=probe.distance - start_distance,
        )
        probe.picture(output / "terminal-camera.png")
        expected = {
            "ordered_duplicate": {"succeeded"},
            "single_object": {"succeeded"},
            "moved": {"succeeded"},
            "cancel_motion": {"canceled"},
            "cancel_planning": {"canceled"},
            "invalid_model": {"rejected"},
            "depth_loss": {"canceled", "destination_unverified"},
            "camera_loss": {"canceled", "destination_unverified"},
            "removed": {"destination_unverified"},
            "occluded": {"destination_unverified"},
            "lookalike": {"destination_ambiguous"},
        }[case]
        assert final.state in expected, (case, final)
        if case in {"single_object", "moved"}:
            assert final.object_result == "matched" and report["mission_distance_m"] > 1
        if case == "removed":
            assert final.object_result == "missing" and final.failure_stage == "geometry"
        if case == "occluded":
            assert final.object_result == "unobserved" and final.failure_stage == "geometry"
        if case == "lookalike":
            assert final.object_result == "ambiguous" and final.failure_stage == "identity"
        if case in {"cancel_motion", "depth_loss", "camera_loss"}:
            probe.wait(
                lambda: abs(probe.odom.twist.twist.linear.x) < 0.02 and abs(probe.odom.twist.twist.angular.z) < 0.02, 10
            )
            report["simulated_robot_stopped"] = True
        if case in {"cancel_planning", "invalid_model"}:
            assert report["mission_distance_m"] < 0.05
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        model.release.set()
        if node is not None:
            node.stop_navigation()
            try:
                probe.wait(lambda: not node.navigation_busy(), 20)
            except RuntimeError:
                report["passed"] = False
                report["cleanup_error"] = "Navigation ownership did not clear before shutdown"
            report["owned_at_shutdown"] = node.navigation_busy()
        if executor is not None:
            executor.shutdown(timeout_sec=15)
        if thread is not None:
            thread.join(5)
        if node is not None:
            if node._mission_traces is not None:
                report["trace_flushed"] = node._mission_traces.flush(timeout=30)
                report["trace_health"] = node._mission_traces.health()
            node.destroy_node()
        report.update(statuses=probe.statuses[started_status:], planner_calls=model.calls, review_calls=reviewer.calls)
        trace_path = output / "traces.sqlite3"
        if trace_path.exists():
            try:
                trace = read_trace(trace_path)
                report["retained_trace_dispatches"] = sum(e["stage"] == "nav2.dispatch" for e in trace["events"])
                report["cancel_requests"] = sum(e["stage"] == "nav2.cancel_requested" for e in trace["events"])
                (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
            except sqlite3.OperationalError as error:
                report["trace_read_error"] = str(error)
        report["dispatches"] = dispatches
        expected_goals = 2 if case == "ordered_duplicate" else 0 if case in {"cancel_planning", "invalid_model"} else 1
        if len(dispatches) != expected_goals:
            report["passed"] = False
            report["dispatch_error"] = f"Expected {expected_goals} goals, observed {len(dispatches)}"
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in {"statuses", "before"}}, indent=2), flush=True)  # noqa: T201
        probe.forward = dict.fromkeys(probe.forward, True)
        world.reset()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument(
        "--isolate-arrival",
        action="store_true",
        help="Defer background object refresh after learning; separate profile",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    probe, world, reports = Probe(), World(), []
    try:
        with patch.object(
            UrllibTransport, "post_json", side_effect=AssertionError("Provider network access forbidden")
        ):
            for case in args.case or CASES:
                reports.append(run_case(probe, world, case, args.output / case, isolate_arrival=args.isolate_arrival))
                if reports[-1].get("owned_at_shutdown"):
                    break  # Do not start another setup goal while transport ownership is uncertain.
    finally:
        probe.destroy_node()
        rclpy.try_shutdown()
        result = {
            "scope": "Gazebo sensors and actual Nav2; scripted provider fixtures",
            "paid_api_calls": 0,
            "isolate_arrival": args.isolate_arrival,
            "cases": [{k: v for k, v in r.items() if k not in {"before", "statuses"}} for r in reports],
            "passed": len(reports) == len(args.case or CASES) and all(r["passed"] for r in reports),
        }
        (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
