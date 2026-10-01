"""The controller's named phase groups, pinned to the set literals they replaced."""

from __future__ import annotations

import pytest

from placecell import navigation_state as states
from placecell.navigation_state import NavState

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
