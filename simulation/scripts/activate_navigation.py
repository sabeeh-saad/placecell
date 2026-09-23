"""Activate the simulation's discovered Nav2 lifecycle managers within a wall-time bound."""

import time

import rclpy
from nav2_msgs.srv import ManageLifecycleNodes
from rclpy.node import Node


def main():
    rclpy.init()
    node = Node("simulation_navigation_startup")
    try:
        deadline = time.monotonic() + 90
        clients = [
            node.create_client(ManageLifecycleNodes, f"/{name}/manage_nodes")
            for name in ("lifecycle_manager_localization", "lifecycle_manager_navigation")
        ]
        for client in clients:
            while not client.wait_for_service(timeout_sec=0.5):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Navigation lifecycle service discovery timed out")
        # A service request endpoint can appear before its independent reply endpoint.
        settle_until = time.monotonic() + 1
        while time.monotonic() < settle_until:
            rclpy.spin_once(node, timeout_sec=0.1)
        futures = [
            client.call_async(ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.STARTUP))
            for client in clients
        ]
        while not all(future.done() for future in futures):
            if time.monotonic() >= deadline:
                raise TimeoutError("Navigation lifecycle activation timed out")
            rclpy.spin_once(node, timeout_sec=0.1)
        if any(future.exception() is not None or not future.result().success for future in futures):
            raise RuntimeError("Navigation lifecycle manager rejected startup")
        node.get_logger().info("Localization and navigation lifecycle managers activated")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
