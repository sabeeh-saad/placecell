"""The controller's phase names and groups, and the phase changes it is expected to make."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from placecell import fault_injection as faults
from placecell import navigation_state as states
from placecell.navigation_state import TRANSITIONS, NavState
from placecell.tracing import read_trace, trace_scope
from tests.conftest import DATA

NAVIGATION = "src/placecell/navigation.py"

# Each group as navigation.py spelled it before it was named.
LITERALS = {
    "EXECUTION_FAILURES": {"failed", "rejected", "unavailable", "uncertain", "cancel_failed", "canceled", "canceling"},
    "ARRIVAL_PHASES": {"awaiting_observation", "verifying_arrival"},
    "CANCEL_INTENT": {"canceling", "cancel_failed", "uncertain"},
    "TRIP_OUTCOMES": {"succeeded", "canceled", "failed", "rejected", "unavailable"},
    "LOOKUP_PHASES": {"planning", "resolving"},
    "SEARCH_DEADLINE_PHASES": {"planning_search", "awaiting_observation", "verifying_arrival"},
    "NO_GOAL_PHASES": {"planning", "resolving", "planning_search", "awaiting_observation", "verifying_arrival"},
    "SEARCHABLE_VERDICTS": {"missing", "unobserved"},
    "FINAL_VERDICTS": {"ambiguous", "missing", "unobserved"},
}


@pytest.mark.parametrize("name", sorted(LITERALS))
def test_named_group_equals_the_literal_it_replaced(name):
    group = getattr(states, name)
    assert isinstance(group, frozenset)
    assert group == LITERALS[name]
    assert all(type(value) is str for value in group)  # plain strings, never enum members


def test_phase_values_are_the_published_strings():
    assert len({state.value for state in NavState}) == len(NavState)
    for state in NavState:
        assert type(state.value) is str and state.value == state.name.lower()
        assert state == state.value and state.value in {state}


def test_transitions_name_only_phases():
    phases = {state.value for state in NavState}
    assert set(TRANSITIONS) == phases
    assert set().union(*TRANSITIONS.values()) <= phases


def test_transitions_allow_every_phase_change_the_controller_makes():
    census = json.loads((DATA / "navigation_transitions.json").read_text())
    # `from: null` is the constructor's first assignment, not a phase change. Rows that
    # only tests wrote are the deliberate stray changes below.
    assert [row["to"] for row in census if row["from"] is None] == ["idle"]
    made = [(row["from"], row["to"]) for row in census if row["from"] is not None and NAVIGATION in row["writers"]]
    assert made and [pair for pair in made if pair[1] not in TRANSITIONS[pair[0]]] == []


def test_unexpected_phase_change_is_traced_not_refused(tmp_path, unexpected_phases):
    rig = faults._Rig(tmp_path)
    try:
        with trace_scope(rig.traces.context("mission", "request")):
            rig.commands._set_phase(NavState.NAVIGATING)
        assert rig.commands._state == "navigating" and type(rig.commands._state) is str
        assert unexpected_phases == [("idle", "navigating")]
        unexpected_phases.clear()
        assert rig.traces.flush()
        events = read_trace(rig.traces.path)["events"]
        assert [e["data"] for e in events if e["stage"] == "phase.unexpected"] == [
            {"from_phase": "idle", "to_phase": "navigating"}
        ]
    finally:
        rig.close()


def test_fault_contract_fails_on_an_unexpected_phase_change(monkeypatch, unexpected_phases):
    def stray(rig, case):
        rig.start()
        rig.commands._set_phase(NavState.PLANNING)  # submitting -> planning is never made

    monkeypatch.setattr(faults, "CASES", (replace(faults.CASES[0], exercise=stray),))
    result = faults.run_faults()["results"][0]
    unexpected_phases.clear()
    assert not result["passed"]
    failed = [check for check in result["checks"] if not check["passed"]]
    assert [check["name"] for check in failed] == ["phase changes listed in TRANSITIONS"]
    assert failed[0]["actual"] == [{"from_phase": "submitting", "to_phase": "planning"}]
