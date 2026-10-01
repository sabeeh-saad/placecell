"""Rules for confirming a destination from a fresh view after Nav2 reaches it.

`NavigationCommands` owns the arrival phases and applies these rules under its lock, with
its current clocks, capture age limit and provenance probe. They read and stamp the trip's
arrival fields but never lock, publish or call a model, so each rule can be tested alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from placecell.errors import FailureStage

if TYPE_CHECKING:
    from placecell.navigation import Destination
    from placecell.navigation_state import TripState
    from placecell.pipeline import Observation


@dataclass(frozen=True, slots=True)
class ArrivalCheck:
    """Arrival rules over the controller's clocks, capture age limit and provenance probe."""

    clock: Callable[[], float]
    observation_clock: Callable[[], float]
    max_observation_age: float
    provenance_ready: Callable[[], bool]

    def arm(self, trip: TripState, timeout_s: float) -> None:
        """Wait for a later capture within the arrival limit and any running local search deadline."""
        trip.arrival_attempts = 0
        trip.arrival_after = self.observation_clock()
        trip.arrival_deadline = self.clock() + timeout_s
        if trip.search_deadline is not None:
            trip.arrival_deadline = min(trip.arrival_deadline, trip.search_deadline)

    def accepts(self, trip: TripState, destination: Destination, observation: Observation) -> bool:
        """A localized capture from the remembered view's camera, taken after arrival, fresh and at the goal pose."""
        memory = destination.memory
        return not (
            memory is None
            or observation.robot_id != memory.robot_id
            or observation.camera_id != memory.camera_id
            or not observation.localization_checked
            or (bool(destination.object_id) and observation.depth is None)
            or not self.provenance_ready()
            or not trip.arrival_after < observation.timestamp <= self.observation_clock()
            or not 0 <= self.observation_clock() - observation.timestamp <= self.max_observation_age
            or not observation.pose.same_frame(destination.pose)
            or observation.pose.distance_to(destination.pose) > 0.35
            or observation.pose.heading_difference(destination.pose) > 0.35
        )

    def take(self, trip: TripState, observation: Observation) -> None:
        """Start an attempt on an accepted capture; its image expires when the capture reaches the age limit."""
        trip.arrival_stamp = observation.timestamp
        trip.arrival_attempts += 1
        trip.image_deadline = (
            self.clock() + self.max_observation_age - (self.observation_clock() - observation.timestamp)
        )

    def fresh(self, trip: TripState) -> bool:
        return (
            trip.arrival_stamp is not None
            and 0 <= self.observation_clock() - trip.arrival_stamp <= self.max_observation_age
            and self.clock() <= trip.image_deadline
        )

    def can_retry(self, trip: TripState, max_attempts: int) -> bool:
        """Only an object target may try another capture, within its attempts and the original deadline."""
        return (
            trip.destination is not None
            and bool(trip.destination.object_id)
            and trip.arrival_attempts < max_attempts
            and self.clock() < trip.arrival_deadline
            and self.provenance_ready()
        )

    def retry(self, trip: TripState) -> None:
        """Drop the expired capture and wait for one taken from now on."""
        trip.arrival_stamp = None
        trip.arrival_after = self.observation_clock()

    def too_late(self, trip: TripState) -> bool:
        """Provenance was lost or the arrival deadline passed."""
        return not self.provenance_ready() or self.clock() >= trip.arrival_deadline


@dataclass(frozen=True, slots=True)
class ArrivalOutcome:
    """A verification result as it is settled into the trip's final arrival verdict."""

    matched: bool
    reason: str
    object_result: str = ""
    failure_stage: FailureStage = ""
    error_type: str = ""

    def expired(self) -> ArrivalOutcome:
        """The checked image aged out during verification and no retry is left."""
        return replace(
            self,
            matched=False,
            reason="The arrival image expired during verification.",
            failure_stage="geometry",
            error_type="",
            object_result="unavailable" if self.object_result else "",
        )

    def late(self) -> ArrivalOutcome:
        """A match that came after the deadline or after provenance was lost."""
        return replace(
            self,
            matched=False,
            reason="Arrival verification expired or localization became unavailable.",
            failure_stage="geometry",
            object_result="unavailable" if self.object_result else "",
        )

    def checked(self, available: bool, object_target: bool) -> ArrivalOutcome:
        """A match stands only while the selected target reference is still available."""
        if available:
            return replace(self, matched=available)
        return replace(
            self,
            matched=False,
            object_result="unavailable" if object_target else "",
            reason="The selected target reference became unavailable.",
            failure_stage="retrieval",
        )

    def settled(self) -> ArrivalOutcome:
        """A match has no failure stage; a failure without one is an identity failure."""
        return replace(self, failure_stage="" if self.matched else (self.failure_stage or "identity"))

    @property
    def message(self) -> str:
        if self.matched:
            return "Destination visible at the reached viewpoint. " + self.reason
        return "Reached the pose; destination unverified. " + self.reason
