"""Launch a fresh office/Nav2 instance for the no-cost recovery campaign."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import rclpy
from pipeline_test import PipelineProbe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--remap-action", action="store_true")
    parser.add_argument("--drop-first-result", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    with (args.output / "simulator.log").open("w") as log:
        simulator = subprocess.Popen(  # noqa: S603 - fixed isolated simulation launch
            ["ros2", "launch", str(root / "launch/office.launch.py"), "navigation:=true"],  # noqa: S607
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        probe = None
        try:
            rclpy.init()
            probe = PipelineProbe()
            probe.wait(probe.ready, 120)
            (args.output / "ready.json").write_text(json.dumps({"localization_ready": True, "pose": str(probe.pose())}))
            probe.destroy_node()
            probe = None
            rclpy.try_shutdown()
            with (args.output / "recovery.log").open("w") as recovery_log:
                result = subprocess.run(  # noqa: S603 - fixed local crash campaign
                    [
                        sys.executable,
                        str(root / "scripts/recovery_test.py"),
                        "--gazebo",
                        "--action",
                        "/navigate_to_pose",
                        "--repeat",
                        str(args.repeat),
                        "--output",
                        str(args.output / "results"),
                        *(["--remap-action"] if args.remap_action else []),
                        *(["--drop-first-result"] if args.drop_first_result else []),
                    ],
                    stdout=recovery_log,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                    check=False,
                )
            return result.returncode
        finally:
            if probe:
                probe.destroy_node()
            rclpy.try_shutdown()
            if simulator.poll() is None:
                os.killpg(simulator.pid, signal.SIGINT)
                try:
                    simulator.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(simulator.pid, signal.SIGKILL)
                    simulator.wait(10)


if __name__ == "__main__":
    raise SystemExit(main())
