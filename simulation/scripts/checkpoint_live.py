"""Bounded live-provider checkpoint through the production node and actual Gazebo/Nav2.

Run inside the simulator container. Credentials are read from a private mounted file;
provider accounting records metadata only, never headers or request bodies.
"""

import argparse
import json
import math
import os
import threading
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import rclpy
import yaml
from pipeline_test import PipelineProbe, snapshot
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from placecell.errors import ProviderError
from placecell.providers._http import UrllibTransport
from placecell.ros2.node import create_node
from placecell.tracing import read_trace


class RequestBudget:
    """A request/time bound plus a stop threshold on provider-reported spend.

    Unknown costs remain unknown. The USD threshold cannot bound an in-flight bill.
    """

    def __init__(self, output, calls=80, seconds=900, reported_usd=1.0):
        self.path = output / "provider-requests.jsonl"
        self.lock = threading.Lock()
        self.calls, self.cost = 0, 0.0
        self.limit, self.deadline, self.cost_limit = calls, time.monotonic() + seconds, reported_usd
        self.original = UrllibTransport.post_json

    def post(self, transport, url, headers, payload, timeout_s):
        with self.lock:
            if self.calls >= self.limit or time.monotonic() >= self.deadline or self.cost >= self.cost_limit:
                raise ProviderError("Checkpoint provider budget exhausted")
            self.calls += 1
            sequence = self.calls
        started = time.monotonic()
        record = {"sequence": sequence, "model": payload.get("model"), "started_monotonic": started}
        try:
            result = self.original(transport, url, headers, payload, min(timeout_s, 60))
            status, _, body = result
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            record.update(
                http_status=status,
                response_id=body.get("id") if isinstance(body, dict) else None,
                usage={key: usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")},
            )
            with self.lock:
                cost = usage.get("cost")
                if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0:
                    self.cost += cost
            return result
        except Exception as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            record["elapsed_s"] = time.monotonic() - started
            with self.lock, self.path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, help="Private credential file; otherwise use OPENROUTER_API_KEY")
    parser.add_argument("--capture-age", type=float, default=5.0)
    parser.add_argument("--single-goal", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.key_file is not None:
        os.environ["OPENROUTER_API_KEY"] = args.key_file.read_text().strip()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY or supply --key-file")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config/placecell.yaml").read_text())
    params = config["placecell"]["ros__parameters"]
    params.update(yaml.safe_load((root / "config/missions.yaml").read_text())["placecell"]["ros__parameters"])
    places = args.output / "places.json"
    places.write_text(json.dumps({"home": {"x": -1.0, "y": 0.3, "yaw": math.pi, "map_id": "office-v1"}}))
    params.update(
        db_path=str(args.output / "db"),
        collection="checkpoint",
        keyframe_dir=str(args.output / "keyframes"),
        corrections_path=str(args.output / "corrections.jsonl"),
        mission_context_path=str(args.output / "missions.sqlite3"),
        mission_trace_path=str(args.output / "traces.sqlite3"),
        command_journal_path=str(args.output / "commands.sqlite3"),
        places_file=str(places),
        recording_dir="",
        navigation_max_observation_age_s=args.capture_age,
    )
    config_path = args.output / "parameters.yaml"
    config_path.write_text(yaml.safe_dump(config))
    budget = RequestBudget(args.output)
    report = {
        "passed": False,
        "scope": "Live hosted providers, production PlaceCell node, Gazebo RGB-D/AMCL/Nav2",
        "capture_age_limit_s": args.capture_age,
        "request_limit": budget.limit,
        "wall_limit_s": 900,
        "reported_cost_stop_usd": budget.cost_limit,
        "checks": [],
    }
    node = executor = thread = None
    rclpy.init(args=["--ros-args", "--params-file", str(config_path)])
    probe = PipelineProbe()
    receipts = []
    publisher = probe.create_publisher(String, "/placecell/command_json", 10)
    probe.create_subscription(String, "/placecell/command_receipt", lambda m: receipts.append(json.loads(m.data)), 10)
    transport_patch = patch.object(UrllibTransport, "post_json", lambda transport, *a: budget.post(transport, *a))
    transport_patch.start()
    try:
        probe.wait(probe.ready, 120)
        report["observation_pose"] = probe.navigate(1.0, 0.0)
        node = create_node()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        database = args.output / "db/checkpoint.state.sqlite3"

        def learned():
            return any(
                "printer" in record["label"].lower()
                and record.get("position")
                and record["position"]["uncertainty_m"] <= 0.35
                and record["status"] == "present"
                for record in snapshot(database)["objects"]
            )

        probe.wait(learned, 180)
        report["before"] = snapshot(database)
        from PIL import Image

        picture = probe.rgb[-1]
        Image.frombytes("RGB", (picture.width, picture.height), bytes(picture.data)).save(
            args.output / "learned-camera.png"
        )
        report["checks"].append("live RGB-D ingestion learned a localized printer")
        report["departure_pose"] = probe.navigate(-1.0, 0.3, math.pi)
        probe.wait(lambda: publisher.get_subscription_count() > 0 and node._localization.ready(), 30)
        command = {
            "schema_version": 2,
            "command_id": "checkpoint-live-mission",
            "scope": {
                "robot_id": "office_robot",
                "map_id": "office-v1",
                "conversation_id": "office-reference-operator",
            },
            "issued_at_unix_s": time.time(),
            "command": "instruction",
            "text": "Go to the printer." if args.single_goal else "Go to the printer, then return to home.",
        }
        initial_distance = probe.distance
        publisher.publish(String(data=json.dumps(command)))
        probe.wait(lambda: receipts, 20)
        publisher.publish(String(data=json.dumps(command)))
        probe.wait(lambda: len(receipts) >= 2, 20)
        assert receipts[0]["disposition"] == "recorded" and receipts[1]["disposition"] == "duplicate", receipts
        report["checks"].append("identified duplicate refused during the actual mission")
        terminal = {
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
        probe.wait(lambda: probe.statuses and probe.statuses[-1]["state"] in terminal, 300)
        report["terminal"] = probe.statuses[-1]
        report["mission_distance_m"] = probe.distance - initial_distance
        report["after"] = snapshot(database)
        picture = probe.rgb[-1]
        Image.frombytes("RGB", (picture.width, picture.height), bytes(picture.data)).save(
            args.output / "terminal-camera.png"
        )
        assert report["terminal"]["state"] == "succeeded", report["terminal"]
        assert report["mission_distance_m"] > 1.0, "No actual semantic trip"
        report["checks"].append("live mission reached its goal(s) after fresh object verification")
        if not args.single_goal:
            assert any(s["state"] == "step_succeeded" and s.get("object_result") == "matched" for s in probe.statuses)
            assert math.hypot(probe.pose().x + 1, probe.pose().y - 0.3) < 0.35
            report["checks"].append("ordered mission returned to the configured home after confirming the printer")
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if node is not None:
            node.stop_navigation()
            deadline = time.monotonic() + 15
            while node.navigation_busy() and time.monotonic() < deadline:
                rclpy.spin_once(probe, timeout_sec=0.05)
            report["owned_at_shutdown"] = node.navigation_busy()
        if executor is not None:
            executor.shutdown(timeout_sec=15)
        if thread is not None:
            thread.join(timeout=5)
        if node is not None:
            if node._mission_traces is not None:
                report["trace_flushed"] = node._mission_traces.flush(timeout=30)
                report["trace_health"] = node._mission_traces.health()
            node.destroy_node()
        if report.get("trace_flushed"):
            trace = read_trace(args.output / "traces.sqlite3")
            (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        report.update(
            statuses=probe.statuses,
            receipts=receipts,
            provider_calls=budget.calls,
            reported_cost_usd=budget.cost,
            total_distance_m=probe.distance,
        )
        transport_patch.stop()
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(  # noqa: T201 - checkpoint summary
            json.dumps(
                {k: v for k, v in report.items() if k not in {"before", "after", "statuses", "receipts"}}, indent=2
            )
        )
        probe.destroy_node()
        rclpy.try_shutdown()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
