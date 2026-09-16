"""Bound manual velocity commands and stop after their publisher becomes silent."""

import rclpy
from control import VelocityWatchdog
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CommandGuard(Node):
    def __init__(self):
        super().__init__("simulation_command_guard")
        self.publisher = self.create_publisher(Twist, "/sim/cmd_vel", 1)
        self.watchdog = VelocityWatchdog()
        self.create_subscription(Twist, "/cmd_vel", self.receive, 1)
        # Wall time keeps the watchdog active while the simulation is paused.
        self.create_timer(0.05, self.tick)

    def receive(self, message):
        self.watchdog.update(message.linear.x, message.angular.z)

    def tick(self):
        command = Twist()
        command.linear.x, command.angular.z = self.watchdog.command()
        self.publisher.publish(command)


def main():
    rclpy.init()
    node = CommandGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
