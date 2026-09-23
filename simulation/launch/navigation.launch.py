"""Real map-server/AMCL and Nav2, isolated inside the office container."""

import os
import runpy
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def merge(destination, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            merge(destination[key], value)
        else:
            destination[key] = value


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    bringup = Path(get_package_share_directory("nav2_bringup"))
    generate = runpy.run_path(str(root / "scripts/make_map.py"))["write_map"]
    world = Path(os.environ.get("PLACECELL_SIM_WORLD", root / "worlds/office.sdf"))
    map_path = generate(world, Path.home() / "maps")
    parameters = yaml.safe_load((bringup / "params/nav2_params.yaml").read_text())
    merge(parameters, yaml.safe_load((root / "config/nav2.yaml").read_text()))
    params_path = Path.home() / "office-nav2.yaml"
    params_path.write_text(yaml.safe_dump(parameters))
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(bringup / "launch/bringup_launch.py")),
                launch_arguments={
                    "map": str(map_path),
                    "params_file": str(params_path),
                    "use_sim_time": "true",
                    "autostart": "false",
                    "use_composition": "False",
                    "slam": "False",
                }.items(),
            ),
            # Allow Fast DDS request AND reply endpoints to discover each other
            # before the first lifecycle transition (rmw_fastrtps issue #842).
            TimerAction(
                period=3.0,
                actions=[
                    ExecuteProcess(cmd=["python3", str(root / "scripts/activate_navigation.py")], output="screen")
                ],
            ),
            ExecuteProcess(cmd=["python3", str(root / "scripts/localization_updates.py")], output="screen"),
        ]
    )
