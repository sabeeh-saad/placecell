"""Live multi-product mission evaluation through real RGB-D, memory and Nav2.

Authored product positions are used only to score learning and outcomes. The model
receives the user instruction and normal memory evidence, never the scene manifest.
"""

import argparse
import base64
import json
import math
import os
import threading
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import rclpy
import yaml
from checkpoint_live import RequestBudget
from PIL import Image
from pipeline_test import PipelineProbe, snapshot
from product_scene import CASES, HOME, MAP_ID, PRODUCTS, score_mission
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from placecell import LocalizationGate, ObjectArrivalVerifier
from placecell.providers._http import UrllibTransport
from placecell.ros2.node import create_node, update_localization
from placecell.tracing import read_trace

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
    "clarification_required",
}
ROOT = Path(__file__).resolve().parents[1]


class Probe(PipelineProbe):
    def __init__(self, output):
        super().__init__()
        self.gate = LocalizationGate("map", MAP_ID, clock=lambda: self.sim_time)
        self.output, self.active_case, self.completions = output, "", []
        self.receipts = []
        self.json_command = self.create_publisher(String, "/placecell/command_json", 10)
        self.create_subscription(
            String, "/placecell/command_receipt", lambda m: self.receipts.append(json.loads(m.data)), 10
        )

    def localization(self, message):
        update_localization(self.gate, message, MAP_ID)

    def pose(self):
        pose = super().pose()
        return replace(pose, map_id=MAP_ID) if pose else None

    def picture(self, path):
        if not self.rgb:
            raise RuntimeError("No camera frame available for evidence")
        message = self.rgb[-1]
        Image.frombytes("RGB", (message.width, message.height), bytes(message.data)).save(path)

    def status(self, message):
        super().status(message)
        value = self.statuses[-1]
        if self.active_case and value["state"] in {"step_succeeded", "succeeded"}:
            path = self.output / f"{self.active_case}-step-{value['mission_step']}.png"
            self.picture(path)
            pose = self.pose()
            self.completions.append(
                {"status": value, "actual_pose": asdict(pose) if pose else None, "image": path.name}
            )


class Budget(RequestBudget):
    def __init__(self, output, calls, seconds, reported_usd):
        super().__init__(output, calls, seconds, reported_usd)
        self.fatal = None

    def check(self):
        if self.fatal is not None:
            raise RuntimeError(f"Provider authorization/credit check failed (HTTP {self.fatal}); test stopped")
        if self.calls >= self.limit or time.monotonic() >= self.deadline or self.cost >= self.cost_limit:
            raise RuntimeError("Product evaluation provider budget exhausted")

    def post(self, *args):
        self.check()
        result = super().post(*args)
        if result[0] in {401, 402, 403}:
            self.fatal = result[0]
        return result


def learned_identity(database, name):
    target = PRODUCTS[name]
    records = snapshot(database)["objects"]
    candidates = [
        record
        for record in records
        if any(alias in record["label"].casefold() for alias in target["aliases"])
        and record["status"] == "present"
        and record.get("position")
        and record["position"]["uncertainty_m"] <= 0.35
        and math.dist(tuple(record["position"][axis] for axis in ("x", "y", "z")), target["position"]) < 0.8
    ]
    return candidates[0] if len(candidates) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--preflight", action="store_true", help="Tour and render the products without model calls")
    parser.add_argument("--case", choices=[case["id"] for case in CASES], action="append")
    parser.add_argument("--explicit-home", action="store_true", help="Clarify the named home place in each instruction")
    parser.add_argument("--max-requests", type=int, default=400)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--reported-cost-stop", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "mode": "no-model scene preflight" if args.preflight else "live hosted multi-product missions",
        "cases": [],
        "training": [],
        "products": PRODUCTS,
        "map_id": MAP_ID,
        "capture_age_limit_s": 5.0,
        "explicit_home": args.explicit_home,
        "limits": {
            "requests": args.max_requests,
            "seconds": args.max_seconds,
            "reported_cost_stop_usd": args.reported_cost_stop,
        },
    }
    if args.key_file:
        os.environ["OPENROUTER_API_KEY"] = args.key_file.read_text().strip()
    if not args.preflight and not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Live evaluation needs --key-file or OPENROUTER_API_KEY")
    config = yaml.safe_load((ROOT / "config/placecell.yaml").read_text())
    params = config["placecell"]["ros__parameters"]
    params.update(yaml.safe_load((ROOT / "config/missions.yaml").read_text())["placecell"]["ros__parameters"])
    places = args.output / "places.json"
    places.write_text(json.dumps({"home": dict(zip(("x", "y", "yaw"), HOME, strict=True)) | {"map_id": MAP_ID}}))
    params.update(
        map_id=MAP_ID,
        db_path=str(args.output / "db"),
        collection="products",
        keyframe_dir=str(args.output / "keyframes"),
        corrections_path=str(args.output / "corrections.jsonl"),
        mission_context_path=str(args.output / "missions.sqlite3"),
        mission_trace_path=str(args.output / "traces.sqlite3"),
        command_journal_path=str(args.output / "commands.sqlite3"),
        places_file=str(places),
        recording_dir="",
    )
    parameters = args.output / "parameters.yaml"
    parameters.write_text(yaml.safe_dump(config))
    budget = Budget(args.output, args.max_requests, args.max_seconds, args.reported_cost_stop)
    node = executor = thread = None
    # Keep ROS available while the finally block cancels motion and writes evidence.
    rclpy.init(args=["--ros-args", "--params-file", str(parameters)], signal_handler_options=SignalHandlerOptions.NO)
    probe = Probe(args.output)
    transport_patch = patch.object(UrllibTransport, "post_json", lambda transport, *a: budget.post(transport, *a))
    original_verify = ObjectArrivalVerifier.verify_image
    arrival_count = 0

    def audit_arrival(verifier, reference, observation, image, *positional, **keywords):
        nonlocal arrival_count
        arrival_count += 1
        prefix = f"arrival-{arrival_count:03}"
        suffix = ".png" if image.startswith("data:image/png") else ".jpg"
        (args.output / (prefix + suffix)).write_bytes(base64.b64decode(image.split(",", 1)[1], validate=True))
        saved = []
        for index, view in enumerate(reference.views):
            crop = f"{prefix}-reference-{index}.png"
            (args.output / crop).write_bytes(view.crop_png)
            saved.append({"crop": crop, "box": asdict(view.box), "pose": asdict(view.memory.pose)})
        (args.output / (prefix + ".json")).write_text(
            json.dumps(
                {
                    "target": keywords.get("target"),
                    "reference": asdict(reference.record),
                    "views": saved,
                    "timestamp": observation.timestamp,
                    "pose": asdict(observation.pose),
                    "depth": asdict(observation.depth) if observation.depth else None,
                    "image": prefix + suffix,
                },
                indent=2,
            )
            + "\n"
        )
        return original_verify(verifier, reference, observation, image, *positional, **keywords)

    arrival_patch = patch.object(ObjectArrivalVerifier, "verify_image", audit_arrival)
    transport_patch.start()
    arrival_patch.start()
    try:
        probe.wait(probe.ready, 120)
        if not args.preflight:
            node = create_node()
            executor = MultiThreadedExecutor(num_threads=4)
            executor.add_node(node)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
        database = args.output / "db/products.state.sqlite3"
        identities = {}
        for name, product in PRODUCTS.items():
            reached = probe.navigate(*product["view"])
            arrived = probe.sim_time
            probe.wait(lambda arrived=arrived: probe.rgb and probe.sim_time > arrived + 1.5, 10)
            if node is not None:

                def learned(name=name):
                    budget.check()
                    return learned_identity(database, name) is not None

                probe.wait(learned, 120)
                identities[name] = learned_identity(database, name)["id"]
            image = args.output / f"learned-{name.replace(' ', '-')}.png"
            probe.picture(image)
            report["training"].append(
                {
                    "product": name,
                    "pose": reached,
                    "image": image.name,
                    "object_id": identities.get(name),
                    "simulation_time": probe.sim_time,
                }
            )
        report["identities"] = identities
        report["before"] = snapshot(database)
        probe.navigate(*HOME)
        if args.preflight:
            report["passed"] = True
            return 0
        cases = [case for case in CASES if not args.case or case["id"] in args.case]
        for case in cases:
            budget.check()
            instruction = case["instruction"]
            if args.explicit_home:
                instruction = 'The configured named place "home" is the starting place. ' + instruction
            probe.wait(lambda: probe.json_command.get_subscription_count() > 0 and node._localization.ready(), 20)
            start_status, start_receipt, start_distance = len(probe.statuses), len(probe.receipts), probe.distance
            probe.active_case, probe.completions = case["id"], []
            started = time.monotonic()
            command = {
                "schema_version": 2,
                "command_id": "products-" + case["id"],
                "scope": {"robot_id": "office_robot", "map_id": MAP_ID, "conversation_id": "office-reference-operator"},
                "issued_at_unix_s": time.time(),
                "command": "instruction",
                "text": instruction,
            }
            probe.json_command.publish(String(data=json.dumps(command)))
            probe.wait(lambda start_receipt=start_receipt: len(probe.receipts) > start_receipt, 20)
            assert probe.receipts[-1]["disposition"] == "recorded", probe.receipts[-1]

            def finished(start_status=start_status):
                budget.check()
                return len(probe.statuses) > start_status and probe.statuses[-1]["state"] in TERMINAL

            probe.wait(finished, 450)
            statuses = probe.statuses[start_status:]
            terminal_image = args.output / f"{case['id']}-terminal.png"
            probe.picture(terminal_image)
            terminal_pose = probe.pose()
            result = {
                "id": case["id"],
                "instruction": instruction,
                **score_mission(case, probe.completions, statuses, identities),
                "elapsed_s": time.monotonic() - started,
                "distance_m": probe.distance - start_distance,
                "statuses": statuses,
                "terminal_image": terminal_image.name,
                "terminal_pose": asdict(terminal_pose) if terminal_pose else None,
            }
            report["cases"].append(result)
            (args.output / f"{case['id']}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(  # noqa: T201 - test progress
                json.dumps({k: v for k, v in result.items() if k not in {"statuses", "completions", "terminal"}}),
                flush=True,
            )
            probe.active_case = ""
            if node.navigation_busy():
                # An ambiguity can retain pending choices. End this independent
                # trial explicitly and await transport release before the next setup.
                node.stop_navigation()
                probe.wait(lambda: not node.navigation_busy(), 20)
                report.setdefault("reset_after_terminal", []).append(case["id"])
            if case is not cases[-1]:
                probe.navigate(*HOME)  # Explicit setup between independently scored instructions.
        report["passed"] = len(report["cases"]) == len(cases) and all(case["passed"] for case in report["cases"])
        report["after"] = snapshot(database)
    except (Exception, KeyboardInterrupt):
        report["error"] = traceback.format_exc()
    finally:
        probe.active_case = ""
        if node is not None:
            try:
                node.stop_navigation()
                probe.wait(lambda: not node.navigation_busy(), 20)
            except Exception:
                report["cleanup_error"] = traceback.format_exc()
                report["passed"] = False
            report["owned_at_shutdown"] = node.navigation_busy()
        if executor:
            executor.shutdown(timeout_sec=20)
        if thread:
            thread.join(5)
        if node:
            report["trace_flushed"] = node._mission_traces.flush(timeout=30)
            node.destroy_node()
            trace = read_trace(args.output / "traces.sqlite3")
            (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
            report["trace_health"] = trace["health"]
        report.update(provider_calls=budget.calls, reported_cost_usd=budget.cost, total_distance_m=probe.distance)
        transport_patch.stop()
        arrival_patch.stop()
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in {"before", "after", "cases"}}), flush=True)  # noqa: T201 - test progress
        probe.destroy_node()
        rclpy.try_shutdown()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
