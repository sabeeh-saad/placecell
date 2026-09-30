"""The node's ROS parameter surface: names, order, defaults and the types rclpy infers.

A changed default type breaks every launch file that sets it, e.g. 3600.0 must stay a double.
After a deliberate change, regenerate the golden file and review its diff:

    PLACECELL_UPDATE_GOLDEN=1 pytest tests/test_ros2_node_parameters.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.conftest import golden
from tests.ros_fakes import hermetic_parameters, ros_type
from tests.ros_fakes import make_node as make_node
from tests.ros_fakes import ros as ros

GOLDEN = Path(__file__).parent / "data" / "ros2_parameters.json"
SIMULATION = Path(__file__).resolve().parents[1] / "simulation" / "config"
BUILTIN = {"use_sim_time"}  # rclpy declares it for every node


def profile(*names: str) -> dict:
    parameters: dict = {}
    for name in names:
        parameters.update(yaml.safe_load((SIMULATION / name).read_text())["placecell"]["ros__parameters"])
    return parameters


def test_declared_parameters_match_the_golden_file(make_node):
    node = make_node()
    actual = [{"name": d.name, "default": d.default, "type": d.type} for d in node.declared]
    assert actual == golden(GOLDEN, actual)
    assert len(actual) == 160
    # JSON compares 3600 == 3600.0, so the inferred type carries the int/float distinction.
    assert all(ros_type(row["default"]) == row["type"] for row in golden(GOLDEN, actual))


def test_declared_values_are_the_defaults_unless_overridden(make_node):
    node = make_node(robot_id="rover", max_queue=7)
    values = {d.name: d.value for d in node.declared}
    defaults = {d.name: d.default for d in node.declared}
    assert values["robot_id"] == "rover" and values["max_queue"] == 7
    overridden = {"robot_id", "max_queue", *hermetic_parameters(Path())}
    assert all(values[k] == defaults[k] for k in defaults if k not in overridden)


def test_overrides_with_another_type_are_rejected_like_rclpy(make_node):
    with pytest.raises(TypeError, match="curator_interval_s"):
        make_node(curator_interval_s=60)


@pytest.mark.parametrize("names", [("placecell.yaml",), ("placecell.yaml", "missions.yaml")])
def test_simulation_profiles_set_declared_parameters_with_their_types(make_node, names):
    node = make_node()
    types = {d.name: d.type for d in node.declared}
    parameters = profile(*names)
    assert parameters.keys() >= BUILTIN
    undeclared = sorted(k for k in parameters.keys() - BUILTIN if k not in types)
    assert not undeclared
    mismatched = sorted(k for k, v in parameters.items() if k not in BUILTIN and ros_type(v) != types[k])
    assert not mismatched


@pytest.mark.parametrize("names", [("placecell.yaml",), ("placecell.yaml", "missions.yaml")])
def test_simulation_profiles_build_a_node(make_node, monkeypatch, tmp_path, names):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    monkeypatch.setattr(
        "placecell.providers._http.Endpoint.post", lambda *a: pytest.fail("construction must not call a provider")
    )
    parameters = {k: v for k, v in profile(*names).items() if k not in BUILTIN}
    # Keep the profile's settings but move its storage into the test directory.
    parameters.update({k: v for k, v in hermetic_parameters(tmp_path).items() if k != "map_id"})
    if parameters.get("mission_trace_path"):
        parameters["mission_trace_path"] = str(tmp_path / "traces.sqlite3")
    node = make_node(**parameters)
    assert node._commands is not None and node._navigator is not None
    assert (node._mission_context is not None) is (len(names) == 2)
