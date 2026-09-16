"""Launch the bundled office, robot and ROS sensor bridge."""

from pathlib import Path

import xacro
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    description = xacro.process_file(str(root / "models/robot/robot.urdf.xacro")).toxml()
    clock = {"use_sim_time": True}
    server = ExecuteProcess(
        cmd=["xvfb-run", "-a", "gz", "sim", "-s", "-r", "-v", "3", str(root / "worlds/office.sdf")],
        output="screen",
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[clock, {"config_file": str(root / "config/bridge.yaml")}],
        output="screen",
    )
    state = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[clock, {"robot_description": description}],
        output="screen",
    )
    guard = ExecuteProcess(cmd=["python3", str(root / "scripts/command_guard.py")], output="screen")
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-world", "office", "-topic", "robot_description", "-name", "placecell_robot", "-z", "0.02"],
        parameters=[clock],
        output="screen",
    )

    def spawn_finished(event, _context):
        if event.returncode:
            return [EmitEvent(event=Shutdown(reason="Robot spawning failed"))]
        return []

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="false", description="Open the Gazebo desktop GUI"),
            DeclareLaunchArgument("navigation", default_value="false", description="Start AMCL and Nav2"),
            RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=spawn_finished)),
            *[
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=process,
                        on_exit=[EmitEvent(event=Shutdown(reason="A required simulation process exited"))],
                    )
                )
                for process in (server, bridge, state, guard)
            ],
            server,
            bridge,
            state,
            guard,
            spawn,
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(root / "launch/navigation.launch.py")),
                condition=IfCondition(LaunchConfiguration("navigation")),
            ),
            ExecuteProcess(cmd=["gz", "sim", "-g"], condition=IfCondition(LaunchConfiguration("gui")), output="screen"),
        ]
    )
