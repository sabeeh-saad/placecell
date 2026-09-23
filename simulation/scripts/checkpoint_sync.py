"""Measure real Gazebo RGB-D admission without running any providers."""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor

from placecell.providers import HashingEmbedder
from placecell.providers.base import Capabilities
from placecell.ros2.depth import PendingImages
from placecell.ros2.node import create_node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/placecell.yaml").read_text())
    config["placecell"]["ros__parameters"].update(
        navigation_enabled=False,
        approach_enabled=False,
        caption_model="",
        db_path="",
        collection="sync",
        keyframe_dir=str(args.output / "images"),
        corrections_path=str(args.output / "corrections.jsonl"),
    )
    path = args.output / "parameters.yaml"
    path.write_text(yaml.safe_dump(config))
    embed = HashingEmbedder()
    embed._capabilities = Capabilities(text=True, image=True)
    rclpy.init(args=["--ros-args", "--params-file", str(path)])
    with (
        patch("placecell.ros2.node.build_embedder", return_value=embed),
        patch("placecell.providers.object_detection.ChatObjectDetector", return_value=object()),
    ):
        node = create_node()
    assert node._worker.stop()
    node._worker.has_capacity = lambda: False
    original = node._depth_at
    report = {"valid_depth": 0, "missing_depth": 0, "failures": []}

    def observe(message, dimensions):
        value = original(message, dimensions)
        report["valid_depth" if value is not None else "missing_depth"] += 1
        if value is None and len(report["failures"]) < 50:
            report["failures"].append(
                {
                    "rgb": PendingImages.stamp(message),
                    "now": node._memory_time(),
                    "depth": [PendingImages.stamp(d) for d in node._depth_frames],
                    "info": [PendingImages.stamp(i) for i in node._camera_infos],
                }
            )
        return value

    node._depth_at = observe
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        report["generation"] = node._sensors.generation(depth=True)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))  # noqa: T201 - diagnostic report


if __name__ == "__main__":
    main()
