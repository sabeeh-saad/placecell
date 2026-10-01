"""The rclpy node. Subscribes to a camera, looks up the robot pose in TF, ingests in the
background, and answers questions published on `~/ask` with JSON on `~/answer`.

Run with `placecell-ros2` inside a sourced ROS 2 environment, or `ros2 run` once packaged.
Configuration is plain ROS parameters; API keys come from the environment variables named
by `api_key_env` and the per-endpoint `*_api_key_env` parameters, never from a parameter, so
they do not end up in launch files or logs. The shared `api_key_env` key goes only to the
scheme, host and port of `chat_base_url`; see `endpoint`.

This module imports without ROS: `create_node` loads the rclpy class from `placecell_node`.
The builders, workers and answer helpers that used to live here are re-exported below.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from placecell.operator import navigation_payload as navigation_payload
from placecell.ros2.answers import ANSWER_SCHEMA_VERSION as ANSWER_SCHEMA_VERSION
from placecell.ros2.answers import NO_CONFIDENT_ANSWER as NO_CONFIDENT_ANSWER
from placecell.ros2.answers import answer_error as answer_error
from placecell.ros2.answers import answer_payload as answer_payload
from placecell.ros2.answers import answer_question as answer_question
from placecell.ros2.bridge import update_localization as update_localization
from placecell.ros2.components import ENDPOINTS as ENDPOINTS
from placecell.ros2.components import build_embedder as build_embedder
from placecell.ros2.components import build_mission_planner as build_mission_planner
from placecell.ros2.components import build_store as build_store
from placecell.ros2.components import build_trace_store as build_trace_store
from placecell.ros2.components import chat_options as chat_options
from placecell.ros2.components import embedding_api_key as embedding_api_key
from placecell.ros2.components import endpoint as endpoint
from placecell.ros2.components import shared_api_key as shared_api_key
from placecell.ros2.workers import BoundedTasks as BoundedTasks
from placecell.ros2.workers import IngestWorker as IngestWorker


def create_node() -> Any:
    """The placecell node. Needs a sourced ROS 2 environment and an initialised rclpy context."""
    from placecell.ros2.placecell_node import PlacecellNode

    return PlacecellNode()


def main(args: list[str] | None = None) -> None:  # pragma: no cover - needs a ROS 2 environment
    import signal

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    # Own the signals: rclpy's handler tears the context down from inside the signal handler,
    # which races with the executor's wait set. A flag lets the loop finish its iteration instead.
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = create_node()
    # Leave room for commands, action results and steady deadlines while camera/TF work
    # is active. Commands and deadline timers have separate callback groups.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
    finally:
        node.stop_navigation()
        deadline = time.monotonic() + 2.0
        while node.navigation_busy() and rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if node.navigation_busy():
            node.get_logger().warning("Shutdown could not confirm navigation cancellation; check Nav2 status.")
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
