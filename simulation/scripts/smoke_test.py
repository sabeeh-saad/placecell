"""Exercise the live simulator through ROS; never substitutes ground truth for sensor data."""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PillowImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from placecell.ros2.depth import aligned_snapshot


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec / 1e9


def yaw(odometry):
    q = odometry.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Probe(Node):
    def __init__(self):
        super().__init__("simulation_smoke_test")
        self.rgb, self.depth, self.info = deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)
        self.scan, self.odom, self.sim_time = None, None, 0.0
        self.clocks = set()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        for topic, kind, callback in (
            ("/clock", Clock, self.clock),
            ("/odom", Odometry, lambda m: setattr(self, "odom", m)),
            ("/scan", LaserScan, lambda m: setattr(self, "scan", m)),
            ("/camera/color/image_raw", Image, lambda m: self.rgb.append(m)),
            ("/camera/aligned_depth_to_color/image_raw", Image, lambda m: self.depth.append(m)),
            ("/camera/color/camera_info", CameraInfo, lambda m: self.info.append(m)),
        ):
            self.create_subscription(kind, topic, callback, qos_profile_sensor_data)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 1)

    def clock(self, message):
        self.sim_time = message.clock.sec + message.clock.nanosec / 1e9
        if len(self.clocks) < 3:
            self.clocks.add(self.sim_time)

    def wait(self, predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return
        raise RuntimeError("Simulator did not satisfy the check before its wall-clock deadline")

    def pair(self):
        for rgb in reversed(self.rgb):
            depth = next((m for m in self.depth if stamp(m) == stamp(rgb)), None)
            info = next((m for m in self.info if stamp(m) == stamp(rgb)), None)
            if depth is not None and info is not None:
                try:
                    transform = self.tf.lookup_transform("odom", rgb.header.frame_id, Time.from_msg(rgb.header.stamp))
                except TransformException:
                    continue
                return rgb, depth, info, transform.transform
        return None

    def drive(self, linear, angular, duration):
        started = self.sim_time
        command = Twist()
        command.linear.x, command.angular.z = linear, angular

        def step():
            self.publisher.publish(command)
            return self.sim_time - started >= duration

        self.wait(step)
        self.publisher.publish(Twist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path.home() / "smoke"), help="Directory for camera PNG and JSON report")
    parser.add_argument("--sensors-only", action="store_true", help="Inspect sensors without issuing velocity commands")
    options = parser.parse_args()
    rclpy.init()
    probe = Probe()
    try:
        probe.wait(
            lambda: (
                probe.pair() is not None
                and probe.scan is not None
                and probe.odom is not None
                and len(probe.clocks) >= 3
            )
        )
        rgb, depth, info, transform = probe.pair()
        assert (rgb.width, rgb.height) == (depth.width, depth.height) == (320, 240)
        assert rgb.encoding == "rgb8" and rgb.header.frame_id == "camera_optical_frame"
        assert depth.encoding == "32FC1"
        snapshot = aligned_snapshot(depth, info, transform, rgb_stamp=stamp(rgb), rgb_frame=rgb.header.frame_id)
        valid = np.isfinite(snapshot.array()) & (snapshot.array() > 0)
        assert valid.mean() > 0.1, "Depth image has too few valid distances"
        rgb_pixels = np.ndarray(
            (rgb.height, rgb.width, 3), dtype=np.uint8, buffer=bytes(rgb.data), strides=(rgb.step, 3, 1)
        )
        assert rgb_pixels.std() > 5, "Camera image appears blank"
        ranges = np.asarray(probe.scan.ranges)
        assert probe.scan.header.frame_id == "laser_frame" and len(ranges) == 360
        assert np.isfinite(ranges).sum() > 100 and np.nanmin(ranges) > 0.08
        assert probe.odom.header.frame_id == "odom" and probe.odom.child_frame_id == "base_footprint"
        probe.tf.lookup_transform("base_footprint", "laser_frame", Time())
        # Optical +Z must point forward (+X in the base), with the camera above ground.
        optical = probe.tf.lookup_transform("base_footprint", "camera_optical_frame", Time()).transform
        q = optical.rotation
        assert 2 * (q.x * q.z + q.y * q.w) > 0.99 and optical.translation.z > 0.4
        output = Path(options.output)
        output.mkdir(parents=True, exist_ok=True)
        PillowImage.fromarray(rgb_pixels).save(output / "camera.png")
        report = {
            "rgb_size": [rgb.width, rgb.height],
            "camera_frame": rgb.header.frame_id,
            "depth_valid_fraction": float(valid.mean()),
            "lidar_samples": len(ranges),
            "simulation_clock_advances": True,
            "rgbd_accepted_by_placecell": True,
        }
        if not options.sensors_only:
            origin = probe.odom.pose.pose.position
            initial_yaw = yaw(probe.odom)
            probe.drive(0.15, 0.0, 2)
            displacement = math.hypot(
                probe.odom.pose.pose.position.x - origin.x, probe.odom.pose.pose.position.y - origin.y
            )
            assert 0.1 < displacement < 0.6, f"Unexpected forward displacement: {displacement}"
            probe.drive(0.0, 0.4, 2)
            turned = abs(math.atan2(math.sin(yaw(probe.odom) - initial_yaw), math.cos(yaw(probe.odom) - initial_yaw)))
            assert 0.3 < turned < 1.1, f"Unexpected rotation: {turned}"
            # Send a command, then become silent: the watchdog must bring the robot to rest.
            command = Twist()
            command.linear.x = 0.15
            for _ in range(5):
                probe.publisher.publish(command)
                rclpy.spin_once(probe, timeout_sec=0.05)
            started = probe.sim_time
            probe.wait(lambda: probe.sim_time - started >= 2)
            assert abs(probe.odom.twist.twist.linear.x) < 0.02
            assert abs(probe.odom.twist.twist.angular.z) < 0.02
            report.update(forward_distance_m=displacement, turn_rad=turned, silence_stops_robot=True)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        probe.get_logger().info(json.dumps(report))
    finally:
        probe.publisher.publish(Twist())
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
