"""Real map-server/AMCL and Nav2, isolated inside the office container."""

import runpy
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
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
    map_path = generate(root / "worlds/office.sdf", Path.home() / "maps")
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
                    "autostart": "true",
                    "use_composition": "False",
                    "slam": "False",
                }.items(),
            ),
            ExecuteProcess(cmd=["python3", str(root / "scripts/localization_updates.py")], output="screen"),
        ]
    )
