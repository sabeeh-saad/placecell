"""Offline ROS 2 operator contract checks; scripted navigation, no model or robot calls."""

import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from placecell import (
    ChatReply,
    CollectionInfo,
    DestinationResolver,
    InMemoryStore,
    MissionPlanner,
    NavigationCommands,
    NavigationEvent,
    PlanReviewAgent,
    Pose,
    Recall,
    ToolCall,
)
from placecell.command_identity import CommandJournal, CommandScope
from placecell.navigation import Resolution
from placecell.providers import HashingEmbedder
from placecell.providers.chat import OpenAICompatibleChat
from placecell.ros2.operator import OperatorInterface


class ScriptedModel:
    def __init__(self, reviewer=False):
        self.calls = 0
        self.reviewer = reviewer

    def complete(self, messages, tools):
        self.calls += 1
        if self.reviewer:
            return ChatReply(None, (ToolCall("r", "review_navigation_plan", {"decision": "approve", "message": "OK"}),))
        return ChatReply(
            None,
            (
                ToolCall(
                    "p",
                    "propose_navigation_plan",
                    {"decision": "ready", "destinations": ["printer", "cupboard"], "message": "Two visits."},
                ),
            ),
        )


class ScriptedNavigator:
    def __init__(self):
        self.sent, self.canceled = [], []

    def send(self, request_id, destination, callback):
        self.sent.append((request_id, destination, callback))

    def cancel(self, request_id):
        self.canceled.append(request_id)


class ContractCheck:
    def __init__(self):
        self.executor = SingleThreadedExecutor()
        self.probe = Node("operator_contract_probe")
        self.executor.add_node(self.probe)
        self.nodes, self.stores, self.checks = [], [], []
        self.retained = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
        )

    def until(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, "ROS operator check timed out"
            self.executor.spin_once(timeout_sec=0.05)

    def server(self, name, journal=None):
        node = Node(name)
        node.set_parameters([Parameter("use_sim_time", value=True)])
        self.nodes.append(node)
        self.executor.add_node(node)
        embedder = HashingEmbedder(64)
        store = InMemoryStore(CollectionInfo(name, embedder.model_name, embedder.dimension))
        self.stores.append(store)
        nav, tasks, model = ScriptedNavigator(), [], ScriptedModel()
        controller = NavigationCommands(
            DestinationResolver(
                store,
                Recall(store, embedder),
                robot_id="test",
                map_id="test-v1",
                places={"printer": Pose(1, 2, map_id="test-v1"), "cupboard": Pose(8, 2, map_id="test-v1")},
            ),
            nav,
            lambda f: tasks.append(f) is None,
            lambda update: bridge.publish(update),
            mission_planner=MissionPlanner(model, PlanReviewAgent(ScriptedModel(reviewer=True))),
        )
        bridge = OperatorInterface(node, controller, journal=journal)
        return bridge, controller, nav, tasks, model

    def snapshot(self, name="operator_contract"):
        client = self.probe.create_client(Trigger, f"/{name}/get_mission_snapshot")
        try:
            self.until(client.service_is_ready)
            future = client.call_async(Trigger.Request())
            self.until(future.done)
            assert future.result().success
            return json.loads(future.result().message)
        finally:
            self.probe.destroy_client(client)

    def late_snapshot(self, bridge, state, name="operator_contract"):
        # With the periodic writer stopped, receipt must come from DDS retained history.
        bridge._timer.cancel()
        bridge.publish_snapshot()
        messages = []
        sub = self.probe.create_subscription(
            String,
            f"/{name}/mission_snapshot",
            lambda msg: messages.append(json.loads(msg.data)),
            self.retained,
        )
        try:
            self.until(lambda: messages)
            assert messages[-1]["status"]["state"] == state, messages
            return messages[-1]
        finally:
            self.probe.destroy_subscription(sub)
            bridge._timer.reset()

    def run(self):
        bridge, _controller, nav, tasks, model = self.server("operator_contract")
        events, snapshots = [], []
        self.probe.create_subscription(
            String, "/operator_contract/navigation_status", lambda msg: events.append(json.loads(msg.data)), 10
        )
        self.probe.create_subscription(
            String,
            "/operator_contract/mission_snapshot",
            lambda msg: snapshots.append(json.loads(msg.data)),
            self.retained,
        )
        publisher = self.probe.create_publisher(String, "/operator_contract/command_json", 1)
        self.until(lambda: publisher.get_subscription_count() > 0 and len(snapshots) >= 2)
        assert snapshots[-1]["captured_at_unix_s"] > snapshots[0]["captured_at_unix_s"]
        assert self.nodes[0].get_clock().now().nanoseconds == 0
        self.checks.append("snapshot refresh continues while simulated time is paused")

        def send(value):
            publisher.publish(String(data=json.dumps(value)))

        send({"schema_version": 1, "command": "instruction", "text": "Visit printer then cupboard"})
        self.until(lambda: tasks)
        planning = self.late_snapshot(bridge, "planning")
        assert planning["busy"] and model.calls == 0 and not nav.sent
        assert self.snapshot()["status"] == planning["status"]
        self.checks.append("late subscriber recovers planning without replay or model work")
        tasks.pop(0)()
        nav.sent[0][2](NavigationEvent("navigating", distance_remaining=2.5))
        send({"schema_version": 99, "command": "stop"})
        self.until(lambda: any(event["state"] == "invalid" for event in events))
        moving = self.snapshot()
        assert moving["status"]["state"] == "navigating" and moving["status"]["distance_remaining"] == 2.5
        assert moving["sequence"] > moving["status"]["sequence"] and not nav.canceled
        self.checks.append("invalid command is visible without replacing active mission state")
        nav.sent[0][2](NavigationEvent("succeeded"))
        next_step = self.snapshot()
        assert next_step["status"]["mission_step"] == 2 and next_step["status"]["destination"] is None
        tasks.pop(0)()
        nav.sent[1][2](NavigationEvent("succeeded"))
        terminal = self.late_snapshot(bridge, "succeeded")
        assert not terminal["busy"] and terminal["status"]["mission_destinations"] == ["printer", "cupboard"]
        assert len(nav.sent) == 2 and model.calls == 1
        self.checks.append("late subscriber recovers complete multi-goal outcome")
        send({"schema_version": 1, "command": "instruction", "text": "Visit printer then cupboard"})
        self.until(lambda: tasks)
        tasks.pop(0)()
        send({"schema_version": 1, "command": "stop"})
        self.until(lambda: nav.canceled)
        canceling = self.snapshot()
        assert canceling["busy"] and canceling["status"]["state"] == "canceling"
        nav.sent[-1][2](NavigationEvent("succeeded"))
        assert self.snapshot()["status"]["state"] == "canceled" and not tasks
        self.checks.append("stop retains ownership and late success cannot advance the mission")

        old = self.probe.create_publisher(String, "/restarted_operator/command_json", self.retained)
        old.publish(String(data=json.dumps({"schema_version": 1, "command": "instruction", "text": "Visit printer"})))
        _, restarted, replay_nav, replay_tasks, replay_model = self.server("restarted_operator")
        fresh = self.snapshot("restarted_operator")
        assert fresh["instance_id"] != terminal["instance_id"] and fresh["status"]["state"] == "idle"
        deadline = time.monotonic() + 1
        self.until(lambda: time.monotonic() >= deadline)
        assert not replay_tasks and not replay_nav.sent and replay_model.calls == 0
        assert restarted.snapshot().sequence == 0
        self.checks.append("new controller has a new instance and does not receive old retained commands")

    def identity(self, output):
        scope = CommandScope("test", "test-v1", "default")
        path = output / "commands.sqlite3"
        journal = CommandJournal(path, scope)
        self.stores.append(journal)
        bridge, controller, nav, tasks, model = self.server("identity_operator", journal)
        receipts, events = [], []
        self.probe.create_subscription(
            String, "/identity_operator/command_receipt", lambda msg: receipts.append(json.loads(msg.data)), 10
        )
        self.probe.create_subscription(
            String, "/identity_operator/navigation_status", lambda msg: events.append(json.loads(msg.data)), 10
        )
        pub = self.probe.create_publisher(String, "/identity_operator/command_json", 1)
        self.until(lambda: pub.get_subscription_count() > 0 and bridge._receipts.get_subscription_count() > 0)
        base = {
            "schema_version": 2,
            "command_id": "mission-one",
            "scope": asdict(scope),
            "issued_at_unix_s": time.time(),
            "command": "instruction",
            "text": "Visit printer then cupboard",
        }

        def send(value):
            before = len(receipts)
            pub.publish(String(data=json.dumps(value)))
            self.until(lambda: len(receipts) > before)
            return receipts[-1]

        first = send(base)
        assert first["disposition"] == "recorded" and len(tasks) == 1
        assert send(base)["disposition"] == "duplicate" and len(tasks) == 1 and model.calls == 0
        tasks.pop(0)()
        assert send(base)["request_id"] == first["request_id"] and len(nav.sent) == 1
        assert send({**base, "text": "Visit cupboard"})["disposition"] == "conflict"
        refused = {**base, "command_id": "busy-command"}
        assert send(refused)["disposition"] == "recorded"
        self.until(lambda: any(e["state"] == "busy" for e in events))
        nav.sent[0][2](NavigationEvent("succeeded"))
        tasks.pop(0)()
        nav.sent[1][2](NavigationEvent("succeeded"))
        terminal = controller.snapshot()
        assert send(base)["disposition"] == send(refused)["disposition"] == "duplicate"
        assert controller.snapshot() == terminal and not tasks and len(nav.sent) == 2 and model.calls == 1
        self.checks.append("identified retries and conflicting reuse never duplicate planning or completed trips")
        self.checks.append("busy refusals remain consumed after the active mission finishes")

        assert send({**base, "command_id": "intentional-repeat"})["disposition"] == "recorded"
        tasks.pop(0)()
        assert len(nav.sent) == 3 and model.calls == 2
        target = controller.snapshot().status.request_id
        stop = {k: v for k, v in base.items() if k != "text"}
        stop.update(command="stop", command_id="stop-one", target_request_id=target)
        assert (
            send({**stop, "command_id": "stale-stop", "target_request_id": first["request_id"]})["disposition"]
            == "recorded"
        )
        self.until(lambda: any(e["state"] == "stale_command" for e in events))
        assert not nav.canceled
        assert send(stop)["disposition"] == "recorded"
        assert send(stop)["disposition"] == "duplicate" and len(nav.canceled) == 1
        nav.sent[-1][2](NavigationEvent("succeeded"))
        assert controller.snapshot().status.state == "canceled" and not tasks
        self.checks.append("new IDs permit deliberate repeat visits; targeted stop retries cancel once")
        self.checks.append("late first-delivery stop cannot affect a different request")

        assert (
            send({**base, "command_id": "expired", "issued_at_unix_s": time.time() - 86401})["disposition"] == "expired"
        )
        assert send({**base, "scope": {**asdict(scope), "map_id": "other"}})["disposition"] == "wrong_scope"
        assert not tasks
        self.checks.append("expired and cross-map envelopes never dispatch")

        # Replace the actual DDS node/controller and reopen the same disk journal.
        old_instance = controller.snapshot().status.instance_id
        old_node = self.nodes.pop()
        self.executor.remove_node(old_node)
        old_node.destroy_node()
        journal.close()
        self.stores.remove(journal)
        reopened = CommandJournal(path, scope)
        self.stores.append(reopened)
        _, controller, nav, tasks, model = self.server("identity_operator", reopened)
        fresh = self.snapshot("identity_operator")
        assert fresh["instance_id"] != old_instance and fresh["command_identity"]["durable"]
        self.until(lambda: pub.get_subscription_count() > 0)
        after = send(base)
        assert after["disposition"] == "duplicate" and after["request_id"] == first["request_id"]
        assert not tasks and not nav.sent and model.calls == 0 and controller.snapshot().sequence == 0
        self.checks.append("fresh controller and reopened durable journal suppress explicit retry across restart")

    def production_node(self, output):
        for enabled in (False, True):
            log_path = output / f"node-{enabled}.log"
            command = [
                sys.executable,
                "-m",
                "placecell.ros2.node",
                "--ros-args",
                "-p",
                f"navigation_enabled:={str(enabled).lower()}",
                "-p",
                "map_id:=test-v1",
                "-p",
                "db_path:=/tmp/operator-check/db",
                "-p",
                "keyframe_dir:=/tmp/operator-check/frames",
                "-p",
                "corrections_path:=/tmp/operator-check/corrections.jsonl",
                "-p",
                "command_journal_path:=/tmp/operator-check/commands.sqlite3",
            ]
            with log_path.open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603
                try:
                    snapshot = self.snapshot("placecell")
                    assert snapshot["navigation_enabled"] == enabled
                    assert snapshot["status"]["state"] == ("idle" if enabled else "disabled")
                    assert bool(snapshot["command_identity"]) == enabled
                    if enabled:
                        pub = self.probe.create_publisher(String, "/placecell/command_json", 1)
                        self.until(lambda pub=pub: pub.get_subscription_count() > 0)
                        pub.publish(String(data='{"schema_version":1,"command":"instruction","text":"go to 1, 2"}'))
                        self.until(lambda: self.snapshot("placecell")["status"]["state"] == "unavailable")
                        self.probe.destroy_publisher(pub)
                    self.checks.append(f"production node snapshot and service: navigation_enabled={enabled}")
                finally:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    assert process.returncode == 0, log_path.read_text()

    def provider_contracts(self):
        _, controller, nav, tasks, _ = self.server("model_contract_operator")
        pub = self.probe.create_publisher(String, "/model_contract_operator/command_json", 1)
        self.until(lambda: pub.get_subscription_count() > 0)

        class ReplyTransport:
            def __init__(self, arguments, refusal=None):
                self.arguments, self.refusal, self.calls = arguments, refusal, 0

            def post_json(self, *_):
                self.calls += 1
                return (
                    200,
                    {},
                    {
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": None,
                                    "refusal": self.refusal,
                                    "tool_calls": [
                                        {
                                            "id": "plan",
                                            "type": "function",
                                            "function": {
                                                "name": "propose_navigation_plan",
                                                "arguments": self.arguments,
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    },
                )

        valid = json.dumps({"decision": "ready", "destinations": ["printer", "cupboard"], "message": "Visit both"})
        cases = [
            ("duplicate decision", '{"decision":"reject",' + valid[1:], None),
            ("refused response", valid, "Provider refused"),
            ("unsupported action", valid[:-1] + ',"action":"drive"}', None),
            ("oversized response", "x" * 65537, None),
        ]
        for name, arguments, refusal in cases:
            transport = ReplyTransport(arguments, refusal)
            reviewer = ScriptedModel(reviewer=True)
            controller._mission_planner = MissionPlanner(
                OpenAICompatibleChat("fixture", transport=transport), PlanReviewAgent(reviewer)
            )
            pub.publish(
                String(data=json.dumps({"schema_version": 1, "command": "instruction", "text": "Visit printer"}))
            )
            self.until(lambda: tasks)
            tasks.pop(0)()
            state = self.snapshot("model_contract_operator")
            assert state["status"]["state"] == "rejected" and state["status"]["message"] and not state["busy"]
            assert transport.calls == 1 and reviewer.calls == 0 and not nav.sent
            self.checks.append(f"provider contract over ROS: {name} rejects without motion")
        transport = ReplyTransport(valid)
        controller._mission_planner = MissionPlanner(
            OpenAICompatibleChat("fixture", transport=transport), PlanReviewAgent(ScriptedModel(reviewer=True))
        )
        pub.publish(String(data=json.dumps({"schema_version": 1, "command": "instruction", "text": "Visit both"})))
        self.until(lambda: tasks)
        tasks.pop(0)()
        nav.sent[0][2](NavigationEvent("succeeded"))
        tasks.pop(0)()
        nav.sent[1][2](NavigationEvent("succeeded"))
        assert self.snapshot("model_contract_operator")["status"]["state"] == "succeeded" and len(nav.sent) == 2
        self.checks.append("provider contract over ROS: valid structured response completes ordered visits")

    def target_attribution(self):
        name = "target_contract_operator"
        bridge, controller, nav, tasks, _ = self.server(name)
        controller._mission_planner = None
        original = controller._resolver.resolve
        events = []
        self.probe.create_subscription(
            String, f"/{name}/navigation_status", lambda msg: events.append(json.loads(msg.data)), 10
        )
        pub = self.probe.create_publisher(String, f"/{name}/command_json", 1)
        self.until(lambda: pub.get_subscription_count() > 0 and bridge._status.get_subscription_count() > 0)
        for stage in ("retrieval", "identity", "geometry", "execution"):
            # Resolution is scripted here; core fixtures exercise real target checks.
            controller._resolver.resolve = (
                original
                if stage == "execution"
                else lambda command, stage=stage: Resolution(
                    "not_found", "Controlled target refusal", failure_stage=stage
                )
            )
            pub.publish(
                String(data=json.dumps({"schema_version": 1, "command": "instruction", "text": "go to printer"}))
            )
            self.until(lambda: tasks)
            tasks.pop(0)()
            if stage == "execution":
                assert len(nav.sent) == 1
                nav.sent[-1][2](NavigationEvent("failed", "Controlled navigation failure"))
            state = "failed" if stage == "execution" else "not_found"
            self.until(
                lambda state=state, stage=stage: (
                    events and events[-1]["state"] == state and events[-1]["failure_stage"] == stage
                )
            )
            snapshot = self.snapshot(name)
            retained = self.late_snapshot(bridge, state, name)
            assert snapshot["status"]["failure_stage"] == retained["status"]["failure_stage"] == stage
            assert not snapshot["busy"] and not tasks
            self.checks.append(f"target failure stage over ROS status, service and retained snapshot: {stage}")

    def close(self):
        self.executor.shutdown()
        for node in [*self.nodes, self.probe]:
            node.destroy_node()
        for store in self.stores:
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    check = ContractCheck()
    report = {"passed": False, "scope": "Real ROS topics, QoS, services and durable admission; scripted navigation"}
    try:
        check.run()
        check.identity(args.output)
        check.provider_contracts()
        check.target_attribution()
        check.production_node(args.output)
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        check.close()
        rclpy.shutdown()
        report["checks"] = check.checks
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))  # noqa: T201 - validation report
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
