"""Real DDS/action cancellation checks with a controlled server; no robot or paid models."""

import argparse
import json
import math
import platform
import threading
import time
import traceback
from pathlib import Path

import rclpy
from nav2_msgs.action import NavigateToPose
from operator_test import ScriptedModel
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from placecell import (
    CollectionInfo,
    DestinationResolver,
    InMemoryStore,
    MissionPlanner,
    PlanReviewAgent,
    Pose,
    Recall,
)
from placecell.mission_context import MissionContext
from placecell.navigation import NavigationCommands
from placecell.providers import HashingEmbedder
from placecell.ros2.navigation import create_navigation_timers, create_navigator
from placecell.ros2.node import BoundedTasks
from placecell.ros2.operator import OperatorInterface
from placecell.tracing import TraceStore, read_trace


def wait(predicate, message, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(message)
        time.sleep(0.005)


class BlockingModel(ScriptedModel):
    def __init__(self, reviewer=False):
        super().__init__(reviewer)
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()

    def complete(self, messages, tools):
        self.entered.set()
        if not self.release.wait(10):
            raise TimeoutError("scripted model remained blocked")
        return super().complete(messages, tools)


class Check:
    def __init__(self, output):
        self.rows, self.checks, self.events = [], [], []
        self.current = None
        self.handles = {}
        self.samples_by_goal = {}
        self.node, self.server_node, self.probe = Node("cancel_contract"), Node("controlled_nav2"), Node("cancel_probe")
        # No /clock publisher: production deadline timers must still run.
        self.node.set_parameters([Parameter("use_sim_time", value=True)])
        self.client_executor, self.server_executor = MultiThreadedExecutor(4), MultiThreadedExecutor(4)
        self.client_executor.add_node(self.node)
        self.client_executor.add_node(self.probe)
        self.server_executor.add_node(self.server_node)
        self.server = ActionServer(
            self.server_node,
            NavigateToPose,
            "/day8/navigate_to_pose",
            execute_callback=self.execute,
            goal_callback=self.goal,
            cancel_callback=self.cancel,
            handle_accepted_callback=self.accepted,
            callback_group=ReentrantCallbackGroup(),
        )
        embed = HashingEmbedder(64)
        self.store = InMemoryStore(CollectionInfo("cancel", embed.model_name, embed.dimension))
        self.context = MissionContext(output / "context.sqlite3")
        self.traces = TraceStore(output / "traces.sqlite3", queue_size=4096)
        self.tasks = BoundedTasks(1, 1, self.node.get_logger())
        self.model, self.reviewer = BlockingModel(), BlockingModel(True)
        self.navigator = create_navigator(self.node, "/day8/navigate_to_pose", 0.3, 30)
        self.commands = NavigationCommands(
            DestinationResolver(
                self.store,
                Recall(self.store, embed),
                robot_id="test",
                map_id="day8",
                places={"printer": Pose(1, 2, map_id="day8"), "cupboard": Pose(3, 2, map_id="day8")},
            ),
            self.navigator,
            self.tasks.submit,
            self.publish,
            mission_planner=MissionPlanner(self.model, PlanReviewAgent(self.reviewer)),
            mission_context=self.context,
            trace_store=self.traces,
            request_timeout_s=5,
        )
        self.bridge = OperatorInterface(self.node, self.commands)
        create_navigation_timers(self.node, self.navigator, self.commands)
        self.pub = self.probe.create_publisher(String, "/cancel_contract/command", 1)
        self.load_pub = self.probe.create_publisher(String, "/day8/observation_load", 1)
        self.load_on = False
        self.load_entered = threading.Event()
        self.load_count = 0
        self.load_sub = self.node.create_subscription(String, "/day8/observation_load", self.slow_observation, 1)
        self.load_timer = self.probe.create_timer(0.2, self.send_load, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.original_handle = self.commands.handle

        def received(text):
            row = self.current
            if text == "stop" and row is not None:
                row["received_s"] = time.monotonic()
            self.original_handle(text)
            if text == "stop" and row is not None:
                row["handled_s"] = time.monotonic()

        self.commands.handle = received
        self.original_cancel = ClientGoalHandle.cancel_goal_async

        def measured_cancel(handle):
            row = self.samples_by_goal.get(bytes(handle.goal_id.uuid).hex())
            before = time.monotonic()
            future = self.original_cancel(handle)
            if row is not None and "issued_s" not in row:
                row["call_started_s"], row["issued_s"] = before, time.monotonic()

                def acknowledgement(done):
                    row["ack_received_s"] = time.monotonic()
                    row["ack_accepted"] = bool(done.result().goals_canceling)

                future.add_done_callback(acknowledgement)
            return future

        ClientGoalHandle.cancel_goal_async = measured_cancel
        self.threads = [threading.Thread(target=e.spin) for e in (self.client_executor, self.server_executor)]
        for thread in self.threads:
            thread.start()
        wait(lambda: self.pub.get_subscription_count() > 0, "command discovery")
        wait(self.navigator._client.server_is_ready, "action discovery")

    def publish(self, update):
        self.events.append(update)
        self.bridge.publish(update)

    def send_load(self):
        if self.load_on:
            self.load_pub.publish(String(data="observation"))

    def slow_observation(self, _):
        self.load_count += 1
        self.load_entered.set()
        time.sleep(0.75)  # Synthetic slow image/TF callback on the node's default group.

    def goal(self, _):
        row = self.current
        row["goal_received_s"] = time.monotonic()
        if not row["accept_release"].wait(8):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def accepted(self, handle):
        row = self.current
        key = bytes(handle.goal_id.uuid).hex()
        row["goal_uuid"] = key
        row["server_accepted_s"] = time.monotonic()
        self.samples_by_goal[key] = row
        self.handles[key] = handle
        handle.execute()

    def cancel(self, handle):
        row = self.samples_by_goal[bytes(handle.goal_id.uuid).hex()]
        row["server_cancel_s"] = time.monotonic()
        assert row["ack_release"].wait(8), "cancel response deliberately withheld too long"
        return CancelResponse.REJECT if row["mode"] == "reject" else CancelResponse.ACCEPT

    def execute(self, handle):
        row = self.samples_by_goal[bytes(handle.goal_id.uuid).hex()]
        if not row["finish_release"].wait(10):
            handle.abort()
        elif handle.is_cancel_requested:
            handle.canceled()
        else:
            handle.succeed()  # Includes success after cancellation was rejected.
        return NavigateToPose.Result()

    def row(self, mode):
        row = {"id": len(self.rows) + 1, "mode": mode, "passed": False}
        for name in ("accept_release", "ack_release", "finish_release"):
            row[name] = threading.Event()
        if mode != "late_accept":
            row["accept_release"].set()
        if mode != "missing_ack":
            row["ack_release"].set()
        self.rows.append(row)
        self.current = row
        return row

    def instruction(self):
        self.pub.publish(String(data="Visit printer then cupboard"))

    def stop(self, row):
        row["published_s"] = time.monotonic()
        self.pub.publish(String(data="stop"))
        wait(lambda: "received_s" in row, "stop not received")

    def blocked_model(self, which):
        model = self.model if which == "planner" else self.reviewer
        model.entered.clear()
        model.release.clear()
        row = self.row(which)
        before = len(self.handles)
        self.instruction()
        wait(model.entered.is_set, "model not entered")
        self.stop(row)
        wait(lambda: "handled_s" in row and not self.commands.busy, "model blocked stop")
        assert self.events[-1].state == "canceled" and len(self.handles) == before
        assert (row["handled_s"] - row["received_s"]) * 1000 <= 500
        model.release.set()
        drained = threading.Event()
        wait(lambda: self.tasks.submit(drained.set), "command worker queue remained full")
        wait(drained.is_set, "late model work not drained")
        assert len(self.handles) == before, "late model reply dispatched a goal"
        row["passed"] = True
        self.checks.append(f"stop while {which} blocked; late reply discarded")

    def trip(self, mode):
        row = self.row(mode)
        before = len(self.handles)
        self.instruction()
        wait(lambda: "goal_received_s" in row, "goal not submitted")
        if mode != "late_accept":
            wait(lambda: self.commands.snapshot().status.state == "navigating", "goal not accepted")
        self.stop(row)
        if mode == "late_accept":
            wait(lambda: self.commands.snapshot().status.state == "uncertain", "paused-time response deadline failed")
            assert self.commands.busy and "issued_s" not in row
            row["accept_release"].set()
        wait(lambda: "issued_s" in row, "cancel request not issued")
        if mode == "missing_ack":
            wait(lambda: self.commands.snapshot().status.state == "uncertain", "cancel deadline did not run")
        elif mode == "reject":
            wait(lambda: self.commands.snapshot().status.state == "cancel_failed", "cancel rejection lost")
        else:
            wait(lambda: row.get("ack_accepted", False), "cancel acknowledgement missing")
        assert self.commands.busy, "acknowledgement released ownership before terminal result"
        self.instruction()
        wait(lambda: self.events[-1].state == "busy", "replacement trip was not refused")
        assert len(self.handles) == before + 1
        if mode == "missing_ack":
            row["ack_release"].set()
            wait(lambda: row.get("ack_accepted", False), "released acknowledgement missing")
        # Accepted cancellation changes server state after its callback returns.
        if mode != "reject":
            wait(lambda: self.handles[row["goal_uuid"]].is_cancel_requested, "server not canceling")
        self.handles[row["goal_uuid"]].publish_feedback(NavigateToPose.Feedback(distance_remaining=1.0))
        row["finish_release"].set()
        wait(lambda: not self.commands.busy, "terminal result did not release ownership")
        assert self.commands.snapshot().status.state == "canceled"
        assert len(self.handles) == before + 1, "second mission leg was dispatched"
        if mode == "late_accept":
            row["accept_to_cancel_ms"] = (row["issued_s"] - row["server_accepted_s"]) * 1000
            assert row["accept_to_cancel_ms"] <= 500
        else:
            row["command_to_cancel_ms"] = (row["issued_s"] - row["received_s"]) * 1000
            row["publish_to_cancel_ms"] = (row["issued_s"] - row["published_s"]) * 1000
            assert row["command_to_cancel_ms"] <= 500
        row["cancel_to_ack_ms"] = (row["ack_received_s"] - row["issued_s"]) * 1000
        row["passed"] = True

    def run(self, samples):
        self.blocked_model("planner")
        self.blocked_model("reviewer")
        for mode in ("late_accept", "reject", "missing_ack"):
            self.trip(mode)
            self.checks.append(mode + ": ownership retained, next goal refused, terminal cancels mission")
        self.load_on = True
        wait(self.load_entered.is_set, "background callback not exercised")
        for index in range(samples):
            self.trip("loaded")
            if (index + 1) % 10 == 0:
                # Artifact collection is outside every measured cancellation interval.
                # Avoid exiting with a large queue behind the bounded background writer.
                assert self.traces.flush(timeout=30), "trace capture did not drain between trial batches"
        self.checks.append("loaded cancellation with persistent context/traces and paused simulation clock")
        assert self.node.get_clock().now().nanoseconds == 0
        assert not any(e.state in {"succeeded", "step_succeeded"} for e in self.events)
        # A real two-goal success is a control against an adapter that always cancels.
        row = self.row("ordered_success_control")
        before = len(self.handles)
        self.instruction()
        wait(lambda: self.commands.snapshot().status.state == "navigating", "control goal not accepted")
        row["finish_release"].set()
        wait(lambda: len(self.handles) == before + 2 and not self.commands.busy, "control chain did not finish")
        assert self.commands.snapshot().status.state == "succeeded"
        assert "issued_s" not in row
        row["passed"] = True
        self.checks.append("positive control: two real action goals complete in order without cancellation")

    def close(self):
        self.load_on = False
        for row in self.rows:
            for value in row.values():
                if isinstance(value, threading.Event):
                    value.set()
        self.model.release.set()
        self.reviewer.release.set()
        self.commands.close()
        self.tasks.stop()
        for executor in (self.client_executor, self.server_executor):
            executor.shutdown(timeout_sec=5)
        for thread in self.threads:
            thread.join(5)
        ClientGoalHandle.cancel_goal_async = self.original_cancel
        self.trace_closed = self.traces.close(timeout=30)
        self.trace_health = self.traces.health()
        self.context.close()
        self.store.close()
        self.server.destroy()
        for node in (self.node, self.server_node, self.probe):
            node.destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if not 100 <= args.samples <= 1000:
        parser.error("samples must be within 100..1000")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        parser.error("report already exists")
    rclpy.init()
    check = Check(args.output)
    report = {"passed": False, "checks": [], "python": platform.python_version(), "samples_requested": args.samples}
    try:
        check.run(args.samples)
        loaded = sorted(r["command_to_cancel_ms"] for r in check.rows if r["mode"] == "loaded")
        report.update(
            passed=True,
            latency_ms={
                "count": len(loaded),
                "p50": loaded[math.ceil(len(loaded) * 0.5) - 1],
                "p95": loaded[math.ceil(len(loaded) * 0.95) - 1],
                "p99": loaded[math.ceil(len(loaded) * 0.99) - 1],
                "max": max(loaded),
                "method": "nearest-rank; monotonic command callback receipt to async cancel API return",
            },
        )
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        check.close()
        report["trace_closed"] = check.trace_closed
        report["trace_health"] = check.trace_health
        if not check.trace_closed or any(
            check.trace_health[key] for key in ("dropped_events", "write_errors", "trimmed_events", "truncated_events")
        ):
            report["passed"] = False
            report["capture_error"] = "Trace writer did not close cleanly without capture loss."
        if check.trace_closed:
            trace = read_trace(args.output / "traces.sqlite3")
            captured_states = [e["data"]["state"] for e in trace["events"] if e["stage"] == "status"]
            expected_states = [e.state for e in check.events if e.request_id]
            # close() emits an untraced idle event; only mission/request events are compared.
            report["trace_statuses_match"] = captured_states == expected_states
            if not report["trace_statuses_match"]:
                report["passed"] = False
                report["capture_error"] = "Trace status sequence does not match observed operator events."
        report.update(
            checks=check.checks,
            observation_callbacks=check.load_count,
            trials=[{k: v for k, v in row.items() if not isinstance(v, threading.Event)} for row in check.rows],
            workload={
                "command_executor_threads": 4,
                "controlled_server_threads": 4,
                "observation_input_hz": 5,
                "default_group_callback_delay_s": 0.75,
                "command_qos_depth": 1,
                "sim_time": "paused at zero",
                "persistent_context_and_traces": True,
            },
            limitations=[
                "Controlled NavigateToPose action server; no Gazebo, physical motion or live models",
                "Synthetic slow observation callback, not the full RGB-D ingestion workload",
                "No cross-process goal reconciliation or physical stop measurement",
                "Missing-handle delay is reported separately; no cancel can be sent before its handle exists",
            ],
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))  # noqa: T201
        rclpy.shutdown()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
