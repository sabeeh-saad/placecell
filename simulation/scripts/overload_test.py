"""Production camera/task saturation alongside real DDS cancellation; no paid models."""

import argparse
import json
import math
import resource
import threading
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import rclpy
from cancellation_test import Check, wait
from sensor_msgs.msg import Image
from std_msgs.msg import String

from placecell import Pose
from placecell.ros2.node import create_node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if not 100 <= args.samples <= 1000:
        parser.error("samples must be within 100..1000")
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / "production"
    params = {
        "db_path": str(root / "db"),
        "collection": "overload",
        "keyframe_dir": str(root / "images"),
        "corrections_path": str(root / "corrections.jsonl"),
        "command_journal_path": str(root / "commands.sqlite3"),
        "mission_context_path": str(root / "missions.sqlite3"),
        "navigation_ownership_path": str(root / "navigation.sqlite3"),
        "map_id": "overload-v1",
        "navigation_enabled": False,
        "caption_model": "scripted",
        "min_interval_s": 0.001,
        "max_interval_s": 0.001,
        "max_queue": 4,
        "batch_size": 1,
        "question_workers": 1,
        "question_queue": 2,
        "refine_interval_s": 0.0,
        "curator_interval_s": 0.0,
        "contradiction": False,
    }
    config = args.output / "parameters.yaml"
    config.write_text(json.dumps({"placecell": {"ros__parameters": params}}))
    rclpy.init(args=["--ros-args", "--params-file", str(config)])
    released, ingest_entered, ask_entered, maintenance_entered, producer_stop = (threading.Event() for _ in range(5))
    calls = []

    class Caption:
        def caption(self, items):
            calls.append(len(items))
            ingest_entered.set()
            assert released.wait(420), "scripted provider hold exceeded test limit"
            return ["printer" for _ in items]

    def blocked(event):
        event.set()
        assert released.wait(420)

    report = {"passed": False, "paid_api_calls": 0, "checks": [], "samples_requested": args.samples}
    node = check = producer = None
    start = time.monotonic()
    try:
        with patch("placecell.providers._http.Endpoint.post", side_effect=AssertionError("unexpected API call")) as api:
            with patch("placecell.providers.OpenAICompatibleCaptioner", return_value=Caption()):
                node = create_node()
            # Authored, trusted poses isolate overload from localization qualification.
            # Encoding, evidence ownership, ingestion, task queues and DDS remain real.
            captures = [0]

            def capture(message, **kwargs):
                captures[0] += 1
                stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
                return Pose(captures[0] * 10.0, 0, map_id="overload-v1"), stamp, None

            node._capture = capture
            node._answer = lambda _: blocked(ask_entered)
            node._run_refiner = lambda: blocked(maintenance_entered)
            check = Check(args.output, loaded_response_timeout_s=10.0)
            check.client_executor.add_node(node)
            rgb = check.probe.create_publisher(Image, "/camera/color/image_raw", 8)
            ask = check.probe.create_publisher(String, "/placecell/ask", 10)
            answers = {"busy": 0, "oversized": 0}

            def answer(message):
                error = json.loads(message.data).get("error", "")
                if error == "question queue full":
                    answers["busy"] += 1
                elif "2000" in error:
                    answers["oversized"] += 1

            check.probe.create_subscription(String, "/placecell/answer", answer, 100)
            wait(lambda: rgb.get_subscription_count() > 0 and ask.get_subscription_count() > 0, "input discovery")
            frames = [0]
            samples = []

            def produce():
                while not producer_stop.is_set():
                    message = Image()
                    message.header.stamp = check.probe.get_clock().now().to_msg()
                    message.header.frame_id = "camera"
                    message.width, message.height = 640, 480
                    message.encoding, message.step, message.data = "rgb8", 1920, b"\x80" * (640 * 480 * 3)
                    rgb.publish(message)
                    frames[0] += 1
                    if frames[0] % 20 == 0:
                        samples.append(
                            {
                                "elapsed_s": time.monotonic() - start,
                                "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            }
                        )
                    producer_stop.wait(0.05)

            producer = threading.Thread(target=produce, daemon=True)
            producer.start()
            wait(ingest_entered.is_set, "ingestion provider did not block")
            wait(
                lambda: node._worker.health()["queued"] == 4 and node._worker.dropped > 0, "camera queue not saturated"
            )
            ask.publish(String(data="where is the printer?"))
            wait(ask_entered.is_set, "question worker did not block")
            for _ in range(100):
                ask.publish(String(data="where is the printer?"))
                time.sleep(0.002)
            wait(lambda: answers["busy"] > 0 and node._questions.health()["queued"] == 2, "busy answer not delivered")
            ask.publish(String(data="x" * 2001))
            wait(lambda: answers["oversized"] == 1, "oversized question not rejected before queueing")
            node._refine()
            wait(maintenance_entered.is_set, "maintenance did not block")
            for _ in range(1000):
                node._refine()
            assert node._maintenance.health()["coalesced"] == 1000
            check.run(args.samples)
            report["saturated"] = {
                "ingestion": node._worker.health(),
                "questions": node._questions.health(),
                "maintenance": node._maintenance.health(),
                "answers": answers.copy(),
            }
            assert report["saturated"]["ingestion"]["queued"] == 4
            assert report["saturated"]["questions"]["active"] == 1
            assert report["saturated"]["questions"]["queued"] == 2
            assert report["saturated"]["maintenance"]["active"] == 1
            assert report["saturated"]["maintenance"]["queued"] == 0
            assert len(list((root / "images").glob("*.jpg"))) == 4
            producer_stop.set()
            producer.join(5)
            check.client_executor.remove_node(node)
            wait(lambda: node._questions.health()["rejected_full"] > 0, "question refusal counter missing")
            released.set()
            wait(lambda: node._worker.health()["queued"] == 0, "accepted ingestion work did not recover")
            assert node._store.count() == 4 and sum(calls) == 4
            wait(lambda: node._questions.health()["completed"] == 3, "question workers did not drain")
            report["recovered"] = {
                "ingestion": node._worker.health(),
                "questions": node._questions.health(),
                "maintenance": node._maintenance.health(),
                "memories": node._store.count(),
            }
            latencies = sorted(row["command_to_cancel_ms"] for row in check.rows if row["mode"] == "loaded")
            report["cancellation_ms"] = {
                "count": len(latencies),
                "p50": latencies[math.ceil(len(latencies) * 0.5) - 1],
                "p95": latencies[math.ceil(len(latencies) * 0.95) - 1],
                "p99": latencies[math.ceil(len(latencies) * 0.99) - 1],
                "max": max(latencies),
            }
            assert report["cancellation_ms"]["p99"] <= 500
            report["checks"] = [
                *check.checks,
                "four accepted camera jobs pin exactly four images under continuous overload",
                "question overflow emits busy responses; oversized inputs bypass no capacity check",
                "1000 maintenance timer repeats coalesce behind one active task",
                "release drains accepted work once with no duplicate sightings",
            ]
            report["frames_published"] = frames[0]
            report["camera_callbacks"] = captures[0]
            report["resource_samples"] = samples
            report["rss_peak_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            assert report["rss_peak_kib"] < 2 * 1024 * 1024
            assert api.call_count == 0
            report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        producer_stop.set()
        if producer is not None:
            producer.join(5)
        released.set()
        if check is not None:
            if node is not None:
                check.client_executor.remove_node(node)
            check.close()
            report["cancellation_trials"] = [
                {key: value for key, value in row.items() if not isinstance(value, threading.Event)}
                for row in check.rows
            ]
            report["trace_health"] = check.trace_health
            if not check.trace_closed or any(check.trace_health[key] for key in ("dropped_events", "write_errors")):
                report["passed"] = False
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        report["elapsed_wall_s"] = time.monotonic() - start
        report["workload"] = {
            "rgb": "640x480 RGB8 at up to 20 Hz",
            "ingestion_capacity": 4,
            "question_workers": 1,
            "question_waiting_capacity": 2,
            "cancellation_samples": args.samples,
            "provider": "scripted blocked captioner plus hashing embedder",
            "network": "disabled",
            "loaded_action_response_timeout_s": 10.0,
            "cancel_request_target_ms": 500,
        }
        report["limitations"] = [
            "Authored poses bypass TF/localization; no depth, object detection, Gazebo or physical stopping checks.",
            "Shared executor uses the real controller/Nav2 adapter against a controlled DDS action server.",
            "DDS may drop superseded samples; published frames and accepted callbacks are separate denominators.",
            "Bounded short stress trial, not the 24-hour endurance or full RGB-D/model resource gate.",
        ]
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
