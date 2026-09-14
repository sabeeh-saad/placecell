from concurrent.futures import Future
from types import SimpleNamespace as Obj

import numpy as np
import pytest

from placecell import Pose
from placecell.errors import ValidationError
from placecell.ros2.approach import Nav2PlanningEnvironment, costmap_from_message, footprint_radius
from tests.test_nav2 import Handle


def message_pose(x=0, y=0):
    return Obj(position=Obj(x=x, y=y, z=0), orientation=Obj(x=0, y=0, z=0, w=1))


def header(frame="map"):
    return Obj(frame_id=frame, stamp=Obj(sec=1000, nanosec=0))


def costmap_message():
    return Obj(
        header=header(),
        metadata=Obj(origin=message_pose(-5, -5), size_x=100, size_y=100, resolution=0.1),
        data=np.zeros(10_000, dtype=np.uint8),
    )


def footprint_message():
    return Obj(
        header=header(),
        polygon=Obj(points=[Obj(x=x, y=y) for x, y in [(2.3, 3.2), (1.7, 3.2), (1.7, 2.8), (2.3, 2.8)]]),
    )


class Client:
    def __init__(self):
        self.future = Future()
        self.ready = True
        self.goals = []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return self.future


def environment(client=None, clock=lambda: 0):
    return Nav2PlanningEnvironment(
        client or Client(),
        lambda start, goal: (start, goal),
        lambda: Pose(0, 0, map_id="office"),
        "map",
        "office",
        monotonic=clock,
    )


def test_published_footprint_is_relative_to_robot_origin_not_world_origin():
    footprint = footprint_message()
    assert footprint_radius(footprint, 2, 3) == pytest.approx(np.hypot(0.3, 0.2))
    with pytest.raises(ValidationError):
        footprint_radius(footprint, 0, 0)
    env = environment()
    env.update_costmap(costmap_message())
    env.update_footprint(footprint, 2, 3)
    snapshot = env.snapshot(10, lambda: False)
    assert snapshot.footprint_radius_m == pytest.approx(np.hypot(0.3, 0.2))
    assert snapshot.costmap.timestamp == 1000
    env.invalidate_footprint()
    with pytest.raises(ValidationError):
        env.snapshot(10, lambda: False)


def test_bad_costmap_revokes_previous_snapshot():
    env = environment()
    env.update_costmap(costmap_message())
    message = costmap_message()
    message.header.frame_id = "odom"
    with pytest.raises(ValidationError):
        env.update_costmap(message)
    env.update_footprint(footprint_message(), 2, 3)
    with pytest.raises(ValidationError):
        env.snapshot(10, lambda: False)
    message = costmap_message()
    message.data = [-1] * 10_000
    with pytest.raises(ValidationError):
        costmap_from_message(message, "map", "office")


def test_bad_footprint_revokes_previous_snapshot_without_factory_wrapper():
    env = environment()
    env.update_costmap(costmap_message())
    env.update_footprint(footprint_message(), 2, 3)
    with pytest.raises(AttributeError):
        env.update_footprint(Obj(), 2, 3)
    with pytest.raises(ValidationError):
        env.snapshot(10, lambda: False)


def complete(client, status=4, path_frame="map", error=0):
    handle = Handle()
    handle.result.set_result(
        Obj(
            status=status,
            result=Obj(
                error_code=error,
                path=Obj(
                    header=header(path_frame),
                    poses=[Obj(header=header(), pose=message_pose()), Obj(header=header(), pose=message_pose(1))],
                ),
            ),
        )
    )
    client.future.set_result(handle)
    return handle


def test_planner_results_are_read_without_sending_navigation_actions():
    client = Client()
    handle = complete(client)
    env = environment(client)
    path = env.path(Pose(0, 0, map_id="office"), Pose(1, 0, map_id="office"), 10, lambda: False)
    assert path == (Pose(0, 0, map_id="office"), Pose(1, 0, map_id="office"))
    assert handle.cancel_calls == 0 and len(client.goals) == 1


@pytest.mark.parametrize("status,error", [(6, 0), (4, 1)])
def test_failed_planner_returns_no_path(status, error):
    client = Client()
    complete(client, status=status, error=error)
    assert environment(client).path(Pose(0, 0), Pose(1, 0), 10, lambda: False) is None


def test_foreign_path_is_rejected():
    client = Client()
    complete(client, path_frame="odom")
    with pytest.raises(ValidationError):
        environment(client).path(Pose(0, 0), Pose(1, 0), 10, lambda: False)


def test_timeout_before_acceptance_cancels_late_planning_goal():
    clock = iter([0, 0, 100])
    client = Client()
    env = environment(client, lambda: next(clock))
    with pytest.raises(ValidationError, match="timed out"):
        env.path(Pose(0, 0), Pose(1, 0), 10, lambda: False)
    handle = Handle()
    client.future.set_result(handle)
    assert handle.cancel_calls == 1


def test_cancel_after_acceptance_cancels_planning_goal():
    clock = iter([0, 0, 0, 100])
    client = Client()
    handle = Handle()
    client.future.set_result(handle)
    with pytest.raises(ValidationError, match="timed out"):
        environment(client, lambda: next(clock)).path(Pose(0, 0), Pose(1, 0), 10, lambda: False)
    assert handle.cancel_calls == 1


def test_canceled_and_unavailable_requests_never_reach_planner():
    client = Client()
    env = environment(client)
    with pytest.raises(ValidationError):
        env.path(Pose(0, 0), Pose(1, 0), 10, lambda: True)
    client.ready = False
    with pytest.raises(ValidationError):
        env.path(Pose(0, 0), Pose(1, 0), 10, lambda: False)
    assert not client.goals


def test_ros_factory_wires_native_planner_and_invalidates_bad_sensor_inputs(monkeypatch):
    import sys

    from placecell.ros2.approach import create_planning_environment

    client, subscriptions, warnings, transforms = Client(), {}, [], []
    stamp = Obj(sec=1000, nanosec=0)
    modules = {
        "geometry_msgs.msg": Obj(PolygonStamped=object, PoseStamped=lambda: Obj(header=header(), pose=message_pose())),
        "nav2_msgs.action": Obj(ComputePathToPose=Obj(Goal=Obj)),
        "nav2_msgs.msg": Obj(Costmap=object),
        "rclpy.action": Obj(ActionClient=lambda *a: client),
        "rclpy.duration": Obj(Duration=Obj),
        "rclpy.qos": Obj(qos_profile_sensor_data="sensor-qos"),
        "rclpy.time": Obj(Time=Obj),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    node = Obj(
        get_clock=lambda: Obj(now=lambda: Obj(to_msg=lambda: stamp)),
        get_logger=lambda: Obj(warning=lambda message, **k: warnings.append(message)),
        create_subscription=lambda cls, topic, callback, qos: subscriptions.__setitem__(topic, callback),
    )

    def transform(*args):
        transforms.append(args)
        return Obj(transform=Obj(translation=Obj(x=2, y=3)))

    env = create_planning_environment(
        node,
        Obj(lookup_transform=transform),
        lambda: Pose(0, 0, map_id="office"),
        frame_id="map",
        map_id="office",
        base_frame="base_link",
        costmap_topic="costmap",
        footprint_topic="footprint",
        action_name="compute_path_to_pose",
        planner_id="GridBased",
        timeout_s=2,
    )
    goal = env.make_goal(Pose(0, 0), Pose(1, 2, 0.5))
    assert goal.use_start and goal.planner_id == "GridBased"
    assert goal.goal.pose.position.x == 1 and goal.goal.header.stamp == stamp
    assert goal.goal.pose.orientation.z == pytest.approx(np.sin(0.25))
    subscriptions["costmap"](costmap_message())
    subscriptions["footprint"](footprint_message())
    assert env.snapshot(float("inf"), lambda: False).footprint_timestamp == 1000
    assert transforms[0][:2] == ("map", "base_link")
    assert transforms[0][2].seconds == 1000
    subscriptions["footprint"](Obj())
    with pytest.raises(ValidationError):
        env.snapshot(float("inf"), lambda: False)
    subscriptions["footprint"](footprint_message())
    subscriptions["costmap"](Obj())
    with pytest.raises(ValidationError):
        env.snapshot(float("inf"), lambda: False)
    assert len(warnings) == 2
