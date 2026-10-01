"""Names for the navigation controller's phases, the groups of them its checks test, and the changes it expects.

`NavigationCommands` keeps its phase as a plain string and publishes the same strings in
status updates. Naming them once makes a misspelt phase an attribute error and gives each
membership test a name that says what the group means. `TRANSITIONS` lists the phase
changes the controller makes; one outside it is traced, never refused. `TripState` holds
the current trip's details beside the phase.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

from placecell.navigation_search import LocalSearch

if TYPE_CHECKING:
    from placecell.navigation import Destination


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

_GOAL_EVENTS = CANCEL_INTENT | TRIP_OUTCOMES | _values(NavState.NAVIGATING)
# While a goal may be outstanding any transport event can arrive. A memory goal's success
# waits for a fresh view, or ends unverified once cancellation or lost provenance prevents one.
_AFTER_GOAL = _GOAL_EVENTS | _values(NavState.AWAITING_OBSERVATION, NavState.DESTINATION_UNVERIFIED)


def _settled(outcome: NavState, *also: NavState) -> frozenset[str]:
    return _values(outcome, NavState.RESOLVING, *also)


TRANSITIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        NavState.IDLE.value: _values(NavState.RESOLVING),
        NavState.PLANNING.value: _values(
            NavState.RESOLVING,
            NavState.REJECTED,
            NavState.CLARIFICATION_REQUIRED,
            NavState.NOT_FOUND,
            NavState.CANCELED,
        ),
        NavState.RESOLVING.value: _values(
            NavState.RESOLVING,
            NavState.PLANNING,
            NavState.SUBMITTING,
            NavState.AMBIGUOUS,
            NavState.NOT_FOUND,
            NavState.UNAVAILABLE,
            NavState.CLARIFICATION_REQUIRED,
            NavState.CANCELED,
        ),
        NavState.SUBMITTING.value: _AFTER_GOAL | _values(NavState.NOT_FOUND),
        NavState.NAVIGATING.value: _AFTER_GOAL,
        NavState.CANCELING.value: _AFTER_GOAL,
        NavState.CANCEL_FAILED.value: _AFTER_GOAL,
        NavState.UNCERTAIN.value: _AFTER_GOAL,
        NavState.AWAITING_OBSERVATION.value: _values(
            NavState.VERIFYING_ARRIVAL, NavState.DESTINATION_UNVERIFIED, NavState.CANCELED
        ),
        NavState.VERIFYING_ARRIVAL.value: _values(
            NavState.AWAITING_OBSERVATION,
            NavState.PLANNING_SEARCH,
            NavState.SUCCEEDED,
            NavState.DESTINATION_UNVERIFIED,
            NavState.CANCELED,
        ),
        NavState.PLANNING_SEARCH.value: _values(
            NavState.SUBMITTING, NavState.DESTINATION_UNVERIFIED, NavState.CANCELED
        ),
        NavState.SUCCEEDED.value: _settled(NavState.SUCCEEDED, NavState.CANCELED),
        NavState.DESTINATION_UNVERIFIED.value: _settled(
            NavState.DESTINATION_UNVERIFIED, NavState.DESTINATION_AMBIGUOUS
        ),
        NavState.DESTINATION_AMBIGUOUS.value: _settled(NavState.DESTINATION_AMBIGUOUS),
        NavState.CANCELED.value: _settled(NavState.CANCELED),
        NavState.FAILED.value: _settled(NavState.FAILED),
        NavState.REJECTED.value: _settled(NavState.REJECTED),
        NavState.UNAVAILABLE.value: _settled(NavState.UNAVAILABLE),
        NavState.NOT_FOUND.value: _values(NavState.RESOLVING),
        NavState.CLARIFICATION_REQUIRED.value: _values(NavState.RESOLVING),
        NavState.AMBIGUOUS.value: _values(
            NavState.RESOLVING, NavState.NOT_FOUND, NavState.UNAVAILABLE, NavState.CANCELED
        ),
    }
)
"""Each phase's allowed next phases, read from every phase change the controller makes.

The suite's census in tests/data/navigation_transitions.json must stay inside it. An outcome
may follow itself because `_complete` records the outcome that `_event` or `_finish_arrival`
has just set, and arrival records an unverified outcome before naming it ambiguous. Any
settled phase starts the next command or mission step at `resolving`.
"""


@dataclass(slots=True)
class TripState:
    """The current trip's details; `NavigationCommands` changes them only under its lock.

    The controller keeps the owning request and the phase. A trip's transport ID outlives it
    so that a later stop or choice expiry can name it.
    """

    destination: Destination | None = None
    requested_at: float = 0.0
    canceling: bool = False
    interruption_reason: str = ""
    accepted_sensors: object = None
    transport_id: str = ""
    leg: int = 0
    arrival_attempts: int = 0
    arrival_stamp: float | None = None
    arrival_after: float = 0.0
    arrival_deadline: float = 0.0
    image_deadline: float = 0.0
    search: LocalSearch = field(default_factory=LocalSearch)

    def begin(self, request_id: str, now: float) -> None:
        """Start a request or mission step. Arrival fields are set again when its goal is reached."""
        self.destination = None
        self.requested_at = now
        self.canceling = False
        self.interruption_reason = ""
        self.accepted_sensors = None
        self.arrival_stamp = None
        self.search = LocalSearch()
        self.leg = 0
        self.transport_id = request_id
