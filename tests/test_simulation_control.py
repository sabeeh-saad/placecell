import pytest
from simulation.scripts.control import VelocityWatchdog


def test_silent_publisher_stops_at_deadline_and_fresh_commands_resume():
    now = [10.0]
    guard = VelocityWatchdog(lambda: now[0])
    assert guard.command() == (0, 0)
    guard.update(0.2, -0.4)
    now[0] += 0.49
    assert guard.command() == (0.2, -0.4)
    now[0] = 10.5
    assert guard.command() == (0, 0)
    guard.update(0.1, 0.0)
    assert guard.command() == (0.1, 0)
    now[0] = 9
    assert guard.command() == (0, 0)


@pytest.mark.parametrize("linear, angular", [(100, -100), (-100, 100)])
def test_excessive_velocities_are_limited_in_both_directions(linear, angular):
    guard = VelocityWatchdog()
    guard.update(linear, angular)
    assert guard.command() == (0.35 if linear > 0 else -0.35, 0.8 if angular > 0 else -0.8)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_commands_stop_both_axes(invalid):
    guard = VelocityWatchdog()
    guard.update(0.2, 0.2)
    guard.update(invalid, 0.2)
    assert guard.command() == (0, 0)
    guard.update(0.2, invalid)
    assert guard.command() == (0, 0)
