"""Typed node settings map one-to-one onto the declared ROS parameters."""

from __future__ import annotations

from dataclasses import fields

from placecell.ros2.config import DEFAULTS, PARAMETER_ORDER, NodeConfig


def ros_names(config: NodeConfig) -> list[str]:
    return [f.metadata["ros"] for group in fields(config) for f in fields(getattr(config, group.name))]


def test_every_parameter_backs_exactly_one_setting():
    names = ros_names(NodeConfig())
    assert len(names) == len(set(names)) == len(PARAMETER_ORDER) == 160
    assert set(names) == set(PARAMETER_ORDER)


def test_defaults_give_the_flat_parameters_in_declaration_order():
    parameters = NodeConfig().parameters()
    assert parameters == DEFAULTS and list(parameters) == list(PARAMETER_ORDER)


def test_values_round_trip_through_the_settings():
    values = {name: f"{name}-set" if isinstance(value, str) else value for name, value in DEFAULTS.items()}
    config = NodeConfig.from_parameters(values)
    assert config.parameters() == values
    assert (config.chat.shared_api_key_env, config.camera.info_topic) == ("api_key_env-set", "camera_info_topic-set")
