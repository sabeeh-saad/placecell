"""Mission progress bookkeeping, without a controller or planner."""

from __future__ import annotations

from placecell.missions import MissionPlan
from placecell.navigation_mission import MissionSequencer


def test_a_mission_walks_its_steps_and_reports_them_one_based():
    mission = MissionSequencer()
    assert not mission.active and not mission.has_next() and mission.status_fields() == ("", 0, ())
    mission.id = "m1"
    assert mission.status_fields() == ("m1", 0, ())  # planning: no plan yet
    mission.start(MissionPlan("ready", ("printer", "cupboard"), "Two stops."), "Visit the printer", [])
    assert mission.active and mission.status_fields() == ("m1", 1, ("printer", "cupboard"))
    assert mission.has_next() and mission.advance() == "cupboard"
    assert not mission.has_next() and mission.status_fields() == ("m1", 2, ("printer", "cupboard"))
    mission.clear()
    assert not mission.active and mission.status_fields() == ("", 0, ()) and mission.grounding == ()


def test_grounding_is_the_request_and_earlier_instructions_only():
    mission = MissionSequencer()
    context = [
        {"kind": "instruction", "data": {"text": "Go to the printer"}},
        {"kind": "status", "data": {"state": "succeeded"}},
        {"kind": "instruction", "data": {"text": None}},
        {"kind": "instruction"},
    ]
    mission.start(MissionPlan("ready", ("printer",), "One stop."), "Then the kitchen", context)
    assert mission.grounding == ("Then the kitchen", "Go to the printer")
