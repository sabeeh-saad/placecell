"""Names for the navigation controller's phases and for the groups of them its checks test.

`NavigationCommands` keeps its phase as a plain string and publishes the same strings in
status updates. Naming them once makes a misspelt phase an attribute error and gives each
membership test a name that says what the group means.
"""

from __future__ import annotations

from enum import Enum


class NavState(str, Enum):
    """Every phase the controller enters. Use `.value`: f-strings format members differently on 3.10 and 3.11+."""

    IDLE = "idle"
    PLANNING = "planning"
    RESOLVING = "resolving"
    SUBMITTING = "submitting"
    NAVIGATING = "navigating"
    CANCELING = "canceling"
    CANCEL_FAILED = "cancel_failed"
    UNCERTAIN = "uncertain"
    AWAITING_OBSERVATION = "awaiting_observation"
    VERIFYING_ARRIVAL = "verifying_arrival"
    PLANNING_SEARCH = "planning_search"
    SUCCEEDED = "succeeded"
    DESTINATION_UNVERIFIED = "destination_unverified"
    DESTINATION_AMBIGUOUS = "destination_ambiguous"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    CLARIFICATION_REQUIRED = "clarification_required"


def _values(*states: NavState) -> frozenset[str]:
    return frozenset(state.value for state in states)


EXECUTION_FAILURES = _values(
    NavState.FAILED,
    NavState.REJECTED,
    NavState.UNAVAILABLE,
    NavState.UNCERTAIN,
    NavState.CANCEL_FAILED,
    NavState.CANCELED,
    NavState.CANCELING,
)
"""Statuses attributed to execution when they name no other failure stage."""

ARRIVAL_PHASES = _values(NavState.AWAITING_OBSERVATION, NavState.VERIFYING_ARRIVAL)
"""Waiting for or checking a fresh view after arrival; visual verification owns completion."""

CANCEL_INTENT = _values(NavState.CANCELING, NavState.CANCEL_FAILED, NavState.UNCERTAIN)
"""Transport events that start cancellation, so a later success cannot advance the mission."""

TRIP_OUTCOMES = _values(NavState.SUCCEEDED, NavState.CANCELED, NavState.FAILED, NavState.REJECTED, NavState.UNAVAILABLE)
"""Transport events that end the trip."""

LOOKUP_PHASES = _values(NavState.PLANNING, NavState.RESOLVING)
"""Waiting for the planner or the destination lookup, bounded by the command timeout."""

SEARCH_DEADLINE_PHASES = _values(NavState.PLANNING_SEARCH, NavState.AWAITING_OBSERVATION, NavState.VERIFYING_ARRIVAL)
"""After arrival: the local search deadline ends verification here instead of canceling motion."""

NO_GOAL_PHASES = LOOKUP_PHASES | SEARCH_DEADLINE_PHASES
"""No Nav2 goal is outstanding, so a stop completes without a transport cancel."""

SEARCHABLE_VERDICTS = frozenset({"missing", "unobserved"})
"""Object arrival results that send local search to another viewpoint."""

FINAL_VERDICTS = frozenset({"ambiguous", "missing", "unobserved"})
"""Object arrival results that a retried capture cannot change."""
