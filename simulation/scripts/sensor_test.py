"""Offline DDS sensor/clock faults through the production node; scripted navigation."""

import argparse
import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import rclpy
import yaml
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from placecell import Destination, Memory, NavigationEvent, Pose
from placecell.navigation import Resolution
from placecell.providers import HashingEmbedder
from placecell.providers.base import Capabilities
from placecell.ros2.node import IngestWorker, create_node


class Navigator:
    def __init__(self):
        self.sent, self.canceled = [], []

    def send(self, request_id, destination, callback):
        self.sent.append((request_id, destination, callback))

    def cancel(self, request_id):
        self.canceled.append(request_id)


class Check:
    def __init__(self, output):
        embedder = HashingEmbedder()
        embedder._capabilities = Capabilities(text=True, image=True)
        with patch("placecell.ros2.node.build_embedder", return_value=embedder):
            self.node = create_node()
        self.probe = rclpy.create_node("sensor_fault_probe")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.probe)
        self.clock = self.probe.create_publisher(Clock, "/clock", 10)
        self.localization = self.probe.create_publisher(
            PoseWithCovarianceStamped, "/amcl_pose", qos_profile_sensor_data
        )
        image_qos = QoSProfile(depth=8, reliability=ReliabilityPolicy.RELIABLE)
        self.rgb = self.probe.create_publisher(Image, "/camera/color/image_raw", image_qos)
        self.depth = self.probe.create_publisher(
            Image, "/camera/aligned_depth_to_color/image_raw", image_qos
        )
        self.info = self.probe.create_publisher(CameraInfo, "/camera/color/camera_info", image_qos)
        self.command = self.probe.create_publisher(String, "/placecell/command", 1)
        self.tf = TransformBroadcaster(self.probe)
        self.checks, self.captures, self.tasks = [], [], []
        self.nav = Navigator()
        self.commands = self.node._commands
        self.commands._navigator = self.nav
        self.commands._submit_callback = lambda task: self.tasks.append(task) is None
        # Preserve actual RGB validation, synchronization, TF and evidence writes.
        # Replace only provider ingestion and action transport; no inference is run.
        assert self.node._worker.stop()
        self.node._worker.has_capacity = lambda: True
        self.node._worker.submit = lambda obs: self.captures.append(obs) is None
        self.goal = Destination("fixture", Pose(0, 0, map_id="test-v1"), "named_place")
        self.commands._resolver.resolve = lambda _: Resolution("resolved", "Scripted fixture", (self.goal,))
        self.commands._resolver.current = lambda _: True
        self.commands._resolver.arrival_available = lambda *_: True
        self.commands._resolver.prepare_destination = lambda d, _: d
        self.commands._resolver.verify = lambda *_: (_ for _ in ()).throw(AssertionError("unexpected model call"))
        self.until(
            lambda: all(
                p.get_subscription_count()
                for p in (self.clock, self.localization, self.rgb, self.depth, self.info, self.command)
            )
        )
        self.seconds = 100
        self.set_time(100)

    def until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, "sensor check timed out"
            self.executor.spin_once(timeout_sec=0.01)

    def spin(self, duration=0.12):
        deadline = time.monotonic() + duration
        self.until(lambda: time.monotonic() >= deadline)

    def passed(self, name):
        self.checks.append(name)

    def set_time(self, seconds):
        self.seconds = seconds
        self.clock.publish(Clock(clock=Time(sec=seconds)))
        self.until(lambda: self.node._memory_time() == seconds)

    def locate(self, *, invalid=False):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp, msg.header.frame_id = Time(sec=self.seconds), "map"
        msg.pose.pose.orientation.w = 1.0
        for index in (0, 7, 35):
            msg.pose.covariance[index] = 1.0 if invalid else 0.01
        self.localization.publish(msg)
        self.until(lambda: self.node._localization.ready() != invalid)

    def transforms(self):
        transforms = []
        for child in ("base_footprint", "camera_optical"):
            tf = TransformStamped()
            tf.header.stamp, tf.header.frame_id, tf.child_frame_id = Time(sec=self.seconds), "map", child
            tf.transform.rotation.w = 1.0
            transforms.append(tf)
        self.tf.sendTransform(transforms)
        from rclpy.time import Time as RosTime

        self.until(lambda: self.node._tf.can_transform("map", "camera_optical", RosTime(seconds=self.seconds)))

    def frame(self, *, depth=True, skew=0, tf=True, malformed=False):
        self.set_time(self.seconds + 1)
        self.locate()
        if tf:
            self.transforms()
        info = CameraInfo()
        info.header.frame_id = "camera_optical"  # Stamp zero is static calibration.
        info.width = info.height = 10
        info.k = [10.0, 0.0, 5.0, 0.0, 10.0, 5.0, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.publish(info)
        if depth:
            msg = Image()
            msg.header.stamp, msg.header.frame_id = Time(sec=self.seconds + skew), "camera_optical"
            msg.width = msg.height = 10
            msg.encoding, msg.step, msg.data = "16UC1", 20, b"\xd0\x07" * 100
            self.depth.publish(msg)
        rgb = Image()
        rgb.header.stamp, rgb.header.frame_id = Time(sec=self.seconds), "camera_optical"
        rgb.width = rgb.height = 10
        rgb.encoding, rgb.step, rgb.data = "rgb8", 30, b"\x80" * (1 if malformed else 300)
        self.rgb.publish(rgb)
        self.spin(0.3)
        return rgb

    def start(self):
        before = self.commands.snapshot().sequence
        self.command.publish(String(data="go to fixture"))
        self.until(
            lambda: (
                self.tasks
                or (
                    self.commands.snapshot().sequence > before
                    and self.commands.snapshot().status.state == "unavailable"
                )
            )
        )
        if self.tasks:
            self.tasks.pop(0)()

    def finish(self, state="canceled"):
        self.nav.sent[-1][2](NavigationEvent(state))
        assert not self.commands.busy

    def run(self):
        self.start()
        assert not self.nav.sent and self.commands.snapshot().status.state == "unavailable"
        self.passed("missing localization blocks DDS instruction admission")
        self.locate()
        self.start()
        assert len(self.nav.sent) == 1
        self.finish("succeeded")
        self.passed("named place needs localization but no RGB-D")

        rgb = self.frame()
        assert self.captures and self.captures[-1].depth is not None
        self.passed("aligned RGB-D with static calibration reaches actual node observation path")
        count = len(self.captures)
        self.rgb.publish(rgb)
        self.spin()
        assert len(self.captures) == count
        self.passed("duplicate RGB is not delivered twice or used to refresh receipt age")
        for stamp in (0, self.seconds + 1_000_000):
            rgb.header.stamp = Time(sec=stamp)
            self.rgb.publish(rgb)
            self.spin()
            assert len(self.captures) == count and not self.node._sensors.ready(camera=True)
            self.frame()
            count += 1
            assert len(self.captures) == count and self.node._sensors.ready(camera=True, depth=True)
        self.passed("zero and future RGB stamps are refused without poisoning later synchronization")
        observation = self.captures[-1]
        memory = replace(
            Memory.create("robot", "front", observation.timestamp, observation.pose, observation.evidence, "fixture"),
            localization_checked=True,
        )
        self.goal = Destination("fixture", memory.pose, "memory", memory, target="fixture", object_id="object")
        for _ in range(3):
            self.frame(depth=False)
        assert self.captures[-1].depth is None and self.node._sensors.ready(camera=True)
        self.start()
        assert len(self.nav.sent) == 1 and self.commands.snapshot().status.state == "unavailable"
        self.passed("lost depth keeps scene observation but blocks object goal")
        self.frame(skew=-1)
        assert self.captures[-1].depth is None
        self.passed("out-of-skew depth never receives a fabricated position")
        self.frame()
        self.start()
        assert len(self.nav.sent) == 2
        for _ in range(3):
            self.frame(depth=False)
        self.until(lambda: self.nav.canceled)
        assert self.commands.busy
        self.finish("succeeded")
        assert self.commands.snapshot().status.state == "destination_unverified"
        self.passed("depth loss during motion retains ownership and late success stays unverified")

        self.goal = replace(self.goal, object_id="")
        self.frame()
        self.start()
        self.until(lambda: len(self.nav.canceled) == 2, timeout=3)
        assert self.commands.busy and not self.node._sensors.ready(camera=True)
        self.finish()
        self.passed("lost camera expires by monotonic age while ROS clock is paused")

        count = len(self.captures)
        self.frame(tf=False)
        assert len(self.captures) == count and not self.node._sensors.ready(camera=True)
        self.transforms()  # Delayed TF cannot retroactively authorize a discarded frame.
        self.spin()
        assert len(self.captures) == count
        self.frame()
        assert len(self.captures) == count + 1
        self.passed("missing and delayed TF discard the capture; later complete capture recovers")

        count = len(self.captures)
        self.frame(malformed=True)
        assert len(self.captures) == count and not self.node._sensors.ready(camera=True)
        self.passed("malformed RGB cannot refresh camera trust")
        self.frame()
        self.goal = replace(self.goal, object_id="object")
        self.start()
        self.nav.sent[-1][2](NavigationEvent("succeeded"))
        assert self.commands.needs_observation
        submit = self.node._worker.submit
        self.node._worker.has_capacity = lambda: False
        self.node._worker.submit = lambda obs: IngestWorker.submit(self.node._worker, obs)
        self.frame()
        assert self.commands.snapshot().status.state == "verifying_arrival" and len(self.tasks) == 1
        self.commands.cancel()
        self.tasks.pop()()  # Queued verification cannot revive the canceled request.
        assert not self.commands.busy
        self.node._worker.has_capacity = lambda: True
        self.node._worker.submit = submit
        self.passed("full ingestion queue still admits a fresh object arrival capture")

        self.goal = Destination("fixture", memory.pose, "named_place")
        self.start()
        self.set_time(self.seconds + 1)
        self.locate(invalid=True)
        self.set_time(self.seconds + 1)
        self.locate()
        self.nav.sent[-1][2](NavigationEvent("succeeded"))
        assert self.commands.snapshot().status.state == "canceled" and not self.tasks
        self.passed("invalid localization followed by recovery cannot turn old mission into success")

        self.start()
        before = len(self.nav.canceled)
        self.set_time(10)
        self.until(lambda: len(self.nav.canceled) > before)
        assert self.commands.busy and self.node._sensors.clock_changed.is_set()
        self.finish("succeeded")
        assert self.commands.snapshot().status.state == "canceled"
        self.locate()
        self.start()
        assert self.commands.snapshot().status.state == "unavailable"
        count = len(self.captures)
        self.frame(tf=False)
        assert len(self.captures) == count and not self.node._pending_images.depth
        self.passed("real backward /clock jump latches fault, cancels mission and blocks new captures/goals")

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.probe.destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    os.environ["PLACECELL_SENSOR_TEST_KEY"] = "offline-fixture"
    params = {
        "use_sim_time": True,
        "db_path": "",
        "map_id": "test-v1",
        "objects_enabled": True,
        "object_backend": "chat",
        "object_model": "scripted",
        "object_api_key_env": "PLACECELL_SENSOR_TEST_KEY",
        "navigation_enabled": True,
        "sensor_max_age_s": 2.0,
        "localization_max_age_s": 4.0,
        "rgbd_wait_s": 0.08,
        "tf_timeout_s": 0.02,
        "min_interval_s": 0.0,
        "max_interval_s": 0.1,
        "keyframe_dir": str(output / "images"),
        "corrections_path": str(output / "corrections.jsonl"),
        "command_journal_path": str(output / "commands.sqlite3"),
        "curator_interval_s": 0.0,
    }
    path = output / "parameters.yaml"
    path.write_text(yaml.safe_dump({"placecell": {"ros__parameters": params}}))
    rclpy.init(args=["--ros-args", "--params-file", str(path)])
    check, error = None, ""
    try:
        with patch(
            "placecell.providers._http.Endpoint.post", side_effect=AssertionError("unexpected provider call")
        ) as provider:
            check = Check(output)
            check.run()
            assert provider.call_count == 0
    except Exception:
        error = traceback.format_exc()
    finally:
        if check is not None:
            check.close()
        rclpy.try_shutdown()
    report = {
        "passed": not error,
        "checks": check.checks if check else [],
        "error": error,
        "transport": "real DDS RGB-D/localization/TF/clock and production node callbacks",
        "navigation": "scripted action transport and selected destinations",
        "paid_api_calls": 0,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))  # noqa: T201
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
