"""Kill a controller while its NavigateToPose server survives, then reconcile exact ownership.

Default: controlled real DDS server for submission/acceptance/terminal fault windows.
--gazebo: real Nav2 in the office world; requires that world to be launched first.
"""

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile

from placecell import CollectionInfo, DestinationResolver, InMemoryStore, NavigationCommands, Pose, Recall
from placecell.command_identity import CommandJournal, CommandScope, IdentifiedCommand
from placecell.mission_context import MissionContext
from placecell.missions import MissionPlanner, PlanReviewAgent
from placecell.navigation_ownership import NavigationOwnership, NavigationScope
from placecell.providers import HashingEmbedder
from placecell.providers.chat import ChatReply, ToolCall
from placecell.ros2.navigation import create_navigation_timers, create_navigator
from placecell.ros2.node import BoundedTasks


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def wait(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Recovery checkpoint timed out"
        time.sleep(0.02)


def scope(action):
    return NavigationScope("recovery-robot", "office-v1", action)


def child(args):
    ros_args = ["--ros-args", "-r", "recovery_alias:=" + args.action] if args.remap_action else []
    rclpy.init(args=ros_args)
    node = Node("recovery_controller")
    node.set_parameters([Parameter("use_sim_time", value=args.gazebo)])
    owner = NavigationOwnership(args.output / "ownership.sqlite3", scope(args.action))
    nav = create_navigator(node, "recovery_alias" if args.remap_action else args.action, 1.0, 120.0, ownership=owner)
    if args.child == "recover":
        for kind, client in nav._recovery._clients.items():
            original_call = client.call_async
            remaining_drop = [args.drop_first_result and kind == "result"]

            def call(request, original_call=original_call, kind=kind, client=client, remaining_drop=remaining_drop):
                print(json.dumps({"recovery_service": kind, "event": "request", "time": time.monotonic()}), flush=True)  # noqa: T201
                future = original_call(request)
                if remaining_drop[0]:
                    remaining_drop[0] = False
                    client.remove_pending_request(future)

                def received(done):
                    response = done.result()
                    print(  # noqa: T201 - machine-readable service diagnostics
                        json.dumps(
                            {
                                "recovery_service": kind,
                                "event": "response",
                                "time": time.monotonic(),
                                "status": getattr(response, "status", None),
                                "return_code": getattr(response, "return_code", None),
                            }
                        ),
                        flush=True,
                    )

                future.add_done_callback(received)
                return future

            client.call_async = call
    embed = HashingEmbedder(64)
    store = InMemoryStore(CollectionInfo("recovery", embed.model_name, embed.dimension))
    tasks = BoundedTasks(1, 1, node.get_logger())
    history = MissionContext(args.output / "context.sqlite3")
    journal = CommandJournal(args.output / "commands.sqlite3", CommandScope("recovery-robot", "office-v1", "crash"))
    model_calls = []

    class Model:
        def __init__(self, reviewer=False):
            self.reviewer = reviewer

        def complete(self, messages, tools):
            model_calls.append("review" if self.reviewer else "plan")
            if args.child == ("reviewing" if self.reviewer else "planning"):
                checkpoint()
            if self.reviewer:
                return ChatReply(
                    None, (ToolCall("r", "review_navigation_plan", {"decision": "approve", "message": "OK"}),)
                )
            return ChatReply(
                None,
                (
                    ToolCall(
                        "p",
                        "propose_navigation_plan",
                        {
                            "decision": "ready",
                            "destinations": ["checkpoint", "later"],
                            "message": "Ordered visits",
                        },
                    ),
                ),
            )

    commands = NavigationCommands(
        DestinationResolver(
            store,
            Recall(store, embed),
            robot_id="recovery-robot",
            map_id="office-v1",
            places={
                "checkpoint": Pose(args.target_x, 0, map_id="office-v1"),
                "later": Pose(-args.target_x, 0, map_id="office-v1"),
            },
        ),
        nav,
        tasks.submit,
        lambda _event: None,
        startup_block_reason=lambda: nav.startup_block_reason,
        mission_context=history,
        mission_planner=MissionPlanner(Model(), PlanReviewAgent(Model(True))),
    )
    create_navigation_timers(node, nav, commands)
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    sends = []
    original_send = nav._client.send_goal_async

    def checkpoint():
        if args.child in {"submitted", "accepted", "terminal"}:
            assert len(sends) == 1, "Crash must interrupt the first mission leg"
        write(
            args.output / "checkpoint.json",
            {
                "point": args.child,
                "goal_submissions": len(sends),
                "target_x": args.target_x,
                **owner.snapshot(),
            },
        )
        while True:
            time.sleep(1)

    def send(*a, **kw):
        sends.append(bytes(kw["goal_uuid"].uuid).hex())
        result = original_send(*a, **kw)
        if args.child == "submitted":
            checkpoint()
        return result

    nav._client.send_goal_async = send
    if args.child == "reserved":
        original = owner.reserve

        def reserve(request):
            result = original(request)
            checkpoint()
            return result

        owner.reserve = reserve
    if args.child == "terminal":
        original = owner.terminal

        def terminal(*a):
            checkpoint()
            return original(*a)

        owner.terminal = terminal
    try:
        if args.child == "recover":
            initial = commands.snapshot()
            if owner.snapshot()["state"] == "pending":
                assert initial.busy and initial.status.state == "uncertain"
                commands.handle("go to checkpoint then later")
                commands.handle("stop")
            else:
                assert not initial.busy and initial.status.state == "idle"
            command = IdentifiedCommand(
                **{
                    **json.loads((args.output / "instruction.json").read_text()),
                    "scope": journal.scope,
                }
            )
            assert journal.claim(command).disposition == "duplicate"
            assert not sends and not tasks._queue.qsize()
            write(args.output / "startup.json", asdict(commands.snapshot()))
        else:
            deadline = time.monotonic() + 30
            while not nav._client.server_is_ready():
                assert time.monotonic() < deadline
                executor.spin_once(timeout_sec=0.05)
            # Allow independent DDS response endpoints to finish discovery.
            until = time.monotonic() + 0.5
            while time.monotonic() < until:
                executor.spin_once(timeout_sec=0.05)
            command = IdentifiedCommand(
                "crash-mission", journal.scope, time.time(), "instruction", "Go to checkpoint then later"
            )
            write(args.output / "instruction.json", asdict(command))
            receipt = journal.claim(command)
            assert receipt.disposition == "recorded"
            if args.child == "command_reserved":
                checkpoint()
            commands.handle(command.text, request_id=receipt.request_id)
        started, ready_at = time.monotonic(), None
        while time.monotonic() - started < 60:
            executor.spin_once(timeout_sec=0.05)
            snapshot = commands.snapshot()
            if args.child == "accepted" and snapshot.status.state == "navigating":
                checkpoint()
            if args.child == "recover":
                assert not sends and not model_calls
                if snapshot.busy and time.monotonic() - started >= 3:
                    write(args.output / "blocked.json", asdict(snapshot))
                if not snapshot.busy and snapshot.status.state == "idle":
                    ready_at = ready_at or time.monotonic()
                    if time.monotonic() - ready_at >= 1:
                        write(
                            args.output / "recovered.json",
                            {
                                "snapshot": asdict(snapshot),
                                "new_goal_sends": len(sends),
                                "new_model_calls": len(model_calls),
                                "retained_context_events": len(history.recent()),
                                "ownership": owner.snapshot(),
                            },
                        )
                        return
        raise TimeoutError("Controller checkpoint timed out")
    finally:
        commands.close()
        tasks.stop()
        executor.shutdown()
        nav.close()
        journal.close()
        history.close()
        node.destroy_node()
        store.close()
        rclpy.try_shutdown()


class Check:
    def __init__(self, args):
        self.args = args
        self.node = Node("recovery_probe")
        self.executor = MultiThreadedExecutor(8)
        self.executor.add_node(self.node)
        self.handles, self.released, self.cancel_ids, self.statuses = {}, set(), [], {}
        self.mode = ""
        self.case_index, self.target_x = 0, 2.0
        self.velocities = deque(maxlen=1024)
        self.odometry = deque(maxlen=1024)
        if args.gazebo:
            self.node.create_subscription(
                Twist, "/sim/cmd_vel", lambda m: self.velocities.append((time.monotonic(), m.linear.x, m.angular.z)), 10
            )
            self.node.create_subscription(
                Odometry,
                "/odom",
                lambda m: self.odometry.append((time.monotonic(), m.pose.pose.position.x, m.pose.pose.position.y)),
                10,
            )
        self.server = None
        if not args.gazebo:
            self.server = ActionServer(
                self.node,
                NavigateToPose,
                args.action,
                self.execute,
                cancel_callback=self.cancel,
                callback_group=ReentrantCallbackGroup(),
                result_timeout=120,
            )
        self.node.create_subscription(
            GoalStatusArray,
            args.action + "/_action/status",
            self.status,
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self.processes = []
        self.logs = []

    def status(self, message):
        for status in message.status_list:
            self.statuses[bytes(status.goal_info.goal_id.uuid).hex()] = status.status

    def execute(self, handle):
        identity = bytes(handle.goal_id.uuid).hex()
        self.handles[identity] = handle
        if self.mode == "terminal" and handle.request.pose.pose.position.y != 9:
            handle.succeed()
        else:
            wait(lambda: identity in self.released, 600)
            if handle.is_cancel_requested:
                handle.canceled()
            else:
                handle.succeed()
        return NavigateToPose.Result()

    def cancel(self, handle):
        self.cancel_ids.append(bytes(handle.goal_id.uuid).hex())
        return CancelResponse.ACCEPT

    def start(self, root, mode):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(root),
            "--child",
            mode,
            "--action",
            self.args.action,
        ]
        if self.args.gazebo:
            command.append("--gazebo")
        if self.args.remap_action:
            command.append("--remap-action")
        if self.args.drop_first_result:
            command.append("--drop-first-result")
        command.extend(["--target-x", str(self.target_x)])
        log = (root / f"{mode}.log").open("w")
        self.logs.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603 - local fixture
        self.processes.append(process)
        return process

    @staticmethod
    def kill(process):
        process.kill()
        process.wait(10)
        assert process.returncode == -signal.SIGKILL

    def run_case(self, root, mode):
        root.mkdir()
        self.mode = mode
        # Repeated fixed destinations can reach Nav2's goal tolerance before SIGKILL.
        # Alternating sides keeps each tested first leg genuinely in flight.
        self.target_x = 2.0 if self.case_index % 2 == 0 else -2.0
        self.case_index += 1
        owner = NavigationOwnership(root / "ownership.sqlite3", scope(self.args.action))
        owner.attest_clean("Isolated test server; previous case terminal, no other controller active")
        owner.close()
        driver = self.start(root, mode)
        wait(lambda: (root / "checkpoint.json").exists())
        identity = json.loads((root / "checkpoint.json").read_text())["goal_id"]
        clean_cases = {"planning", "reviewing", "command_reserved"}
        if mode not in {"reserved", *clean_cases}:
            wait(lambda: identity in self.statuses)
            if mode != "terminal":
                assert self.statuses[identity] in (1, 2, 3), self.statuses
        self.kill(driver)
        surviving_status = self.statuses.get(identity)
        recovered = self.start(root, "recover")
        wait(lambda: (root / "startup.json").exists())
        if mode == "reserved":
            wait(lambda: (root / "blocked.json").exists())
            assert not (root / "recovered.json").exists()
            self.kill(recovered)
            owner = NavigationOwnership(root / "ownership.sqlite3", scope(self.args.action))
            assert owner.snapshot()["state"] == "pending"
            owner.close()
            return {
                "surviving_status": surviving_status,
                "outcome": "unknown goal remains blocked",
                "goal_id": identity,
            }
        if not self.args.gazebo and mode not in {"terminal", *clean_cases}:
            wait(lambda: identity in self.cancel_ids and (root / "blocked.json").exists())
            assert not (root / "recovered.json").exists(), "Cancel acknowledgement released ownership"
            self.released.add(identity)
        wait(lambda: (root / "recovered.json").exists(), 30)
        recovered.wait(15)
        assert recovered.returncode == 0, (root / "recover.log").read_text()
        result = json.loads((root / "recovered.json").read_text())
        assert result["ownership"]["state"] == "clean" and result["new_goal_sends"] == 0
        assert result["ownership"]["goal_id"] == identity
        if mode in clean_cases:
            assert result["retained_context_events"] > 0 or mode == "command_reserved"
            return {"surviving_status": surviving_status, **result}
        wait(lambda: self.statuses.get(identity) in (4, 5, 6))
        if self.args.gazebo:
            after = time.monotonic()
            wait(
                lambda: (
                    self.velocities
                    and self.odometry
                    and self.velocities[-1][0] > after + 1
                    and self.odometry[-1][0] > after + 1
                )
            )
            velocities = [v for v in list(self.velocities) if v[0] >= after]
            positions = [v for v in list(self.odometry) if v[0] >= after]
            assert velocities and max(abs(v) for row in velocities for v in row[1:]) < 0.01
            drift = ((positions[-1][1] - positions[0][1]) ** 2 + (positions[-1][2] - positions[0][2]) ** 2) ** 0.5
            assert drift < 0.02, drift
            result["post_recovery_velocity_samples"] = len(velocities)
            result["post_recovery_drift_m"] = drift
        return {"surviving_status": surviving_status, "terminal_status": self.statuses[identity], **result}

    def run(self):
        sentinel = None
        if not self.args.gazebo:
            client = ActionClient(self.node, NavigateToPose, self.args.action)
            wait(client.server_is_ready)
            goal = NavigateToPose.Goal()
            goal.pose.pose.position.y = 9.0
            future = client.send_goal_async(goal)
            wait(future.done)
            sentinel = bytes(future.result().goal_id.uuid).hex()
            wait(lambda: sentinel in self.handles)
        cases = (
            ("submitted", "accepted")
            if self.args.gazebo
            else (
                "command_reserved",
                "planning",
                "reviewing",
                "reserved",
                "submitted",
                "accepted",
                "terminal",
            )
        )
        rows = []
        for repeat in range(self.args.repeat):
            for mode in cases:
                row = {"repeat": repeat, "checkpoint": mode, "passed": False}
                try:
                    row.update(self.run_case(self.args.output / f"{repeat}-{mode}", mode))
                    if sentinel:
                        assert sentinel not in self.cancel_ids and self.handles[sentinel].is_active
                    row["passed"] = True
                except Exception:
                    row["error"] = traceback.format_exc()
                rows.append(row)
                write(
                    self.args.output / "report.json",
                    {
                        "passed": len(rows) == len(cases) * self.args.repeat and all(r["passed"] for r in rows),
                        "expected_cases": len(cases) * self.args.repeat,
                        "completed": len(rows) == len(cases) * self.args.repeat,
                        "cases": rows,
                        "paid_api_calls": 0,
                        "action_remapped": self.args.remap_action,
                        "first_result_reply_dropped": self.args.drop_first_result,
                        "server": "Nav2 in Gazebo" if self.args.gazebo else "controlled DDS NavigateToPose",
                        "unrelated_goal_preserved": sentinel not in self.cancel_ids if sentinel else None,
                    },
                )
                if not row["passed"]:
                    raise AssertionError(row["error"])

    def close(self):
        for process in self.processes:
            if process.poll() is None:
                self.kill(process)
        for log in self.logs:
            log.close()
        self.released.update(self.handles)
        self.executor.shutdown(timeout_sec=5)
        self.node.destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gazebo", action="store_true")
    parser.add_argument("--remap-action", action="store_true")
    parser.add_argument("--drop-first-result", action="store_true")
    parser.add_argument("--target-x", type=float, default=2.0)
    parser.add_argument("--action", default="/recovery/navigate_to_pose")
    parser.add_argument(
        "--child",
        choices=(
            "command_reserved",
            "planning",
            "reviewing",
            "reserved",
            "submitted",
            "accepted",
            "terminal",
            "recover",
        ),
    )
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    check = Check(args)
    try:
        check.run()
    finally:
        check.close()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
