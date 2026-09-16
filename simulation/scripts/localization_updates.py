"""Ask AMCL to incorporate fresh laser scans even while stationary during vision calls."""

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_srvs.srv import Empty


class Updates(Node):
    def __init__(self):
        super().__init__("stationary_localization_updates")
        self.set_parameters([Parameter("use_sim_time", value=True)])
        self.initial_pose = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
        self.state = self.create_client(GetState, "/amcl/get_state")
        self.state_pending = None
        self.initialized = False
        self.client = self.create_client(Empty, "/request_nomotion_update")
        self.pending = None
        self.create_timer(2.0, self.update)

    def update(self):
        if not self.initialized:
            if self.state_pending is not None and self.state_pending.done():
                if self.state_pending.result().current_state.id == 3:
                    pose = PoseWithCovarianceStamped()
                    pose.header.frame_id = "map"
                    pose.header.stamp = self.get_clock().now().to_msg()
                    pose.pose.pose.orientation.w = 1.0
                    # An initial estimate at the spawn point, with real uncertainty.
                    # Subsequent poses and covariance come only from AMCL laser updates.
                    pose.pose.covariance[0] = pose.pose.covariance[7] = 0.04
                    pose.pose.covariance[35] = 0.04
                    self.initial_pose.publish(pose)
                    self.initialized = True
                self.state_pending = None
            if not self.initialized and self.state.service_is_ready() and self.state_pending is None:
                self.state_pending = self.state.call_async(GetState.Request())
            return
        if self.client.service_is_ready() and (self.pending is None or self.pending.done()):
            self.pending = self.client.call_async(Empty.Request())


if __name__ == "__main__":
    rclpy.init()
    node = Updates()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
