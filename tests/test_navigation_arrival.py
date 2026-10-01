"""Arrival rules applied to a trip's fields, without a controller, real clocks or providers."""

from __future__ import annotations

from dataclasses import replace

import pytest

from placecell.memory import Evidence, EvidenceKind, Memory, Pose
from placecell.navigation import Destination
from placecell.navigation_arrival import ArrivalCheck, ArrivalOutcome
from placecell.navigation_state import TripState
from placecell.pipeline import Observation

POSE = Pose(1, 2, map_id="office")
EVIDENCE = Evidence(EvidenceKind.FRAME, "frames/a.jpg")


def check(now: float = 100.0, observed: float = 1000.0, ready: bool = True) -> ArrivalCheck:
    return ArrivalCheck(lambda: now, lambda: observed, 5.0, lambda: ready)


def destination(object_id: str = "") -> Destination:
    memory = Memory.create("r1", "front", 900.0, POSE, EVIDENCE, "printer")
    return Destination("printer", POSE, "memory", memory, "printer", object_id)


def observation(**changes) -> Observation:
    return replace(Observation("r1", "front", 999.0, POSE, EVIDENCE, localization_checked=True), **changes)


def test_arming_waits_for_later_captures_and_never_outlives_the_search():
    trip = TripState(arrival_attempts=2)
    check().arm(trip, 30.0)
    assert (trip.arrival_attempts, trip.arrival_after, trip.arrival_deadline) == (0, 1000.0, 130.0)
    trip.search.deadline = 110.0
    check().arm(trip, 30.0)
    assert trip.arrival_deadline == 110.0


@pytest.mark.parametrize(
    "change",
    [
        {"robot_id": "r2"},
        {"camera_id": "rear"},
        {"localization_checked": False},
        {"timestamp": 990.0},  # not after arrival
        {"timestamp": 994.0},  # older than the age limit
        {"timestamp": 1001.0},  # from the future
        {"pose": Pose(1.5, 2, map_id="office")},
        {"pose": Pose(1, 2, 0.5, map_id="office")},
        {"pose": Pose(1, 2, map_id="warehouse")},
    ],
)
def test_only_a_fresh_localized_capture_at_the_goal_is_accepted(change):
    trip = TripState(arrival_after=990.0)
    assert check().accepts(trip, destination(), observation())
    assert not check().accepts(trip, destination(), observation(**change))


def test_object_targets_need_aligned_depth_and_every_target_needs_provenance():
    trip = TripState(arrival_after=990.0)
    assert not check().accepts(trip, destination("cup-1"), observation())
    assert not check(ready=False).accepts(trip, destination(), observation())
    assert not check().accepts(trip, replace(destination(), memory=None), observation())


def test_an_attempt_expires_with_its_capture_and_retries_stay_inside_the_deadline():
    trip = TripState(destination=destination("cup-1"), arrival_deadline=130.0)
    check().take(trip, observation(timestamp=998.0))
    assert (trip.arrival_stamp, trip.arrival_attempts, trip.image_deadline) == (998.0, 1, 103.0)
    assert check().fresh(trip)
    assert not check(now=103.5).fresh(trip) and not check(observed=1003.5).fresh(trip)
    assert check().can_retry(trip, 2) and not check().can_retry(trip, 1)
    assert not check(now=130.0).can_retry(trip, 2) and not check(ready=False).can_retry(trip, 2)
    assert not check().can_retry(replace(trip, destination=destination()), 2)  # scene targets never retry
    check(observed=1001.0).retry(trip)
    assert (trip.arrival_stamp, trip.arrival_after) == (None, 1001.0) and not check().fresh(trip)


def test_a_match_is_late_after_the_deadline_or_without_provenance():
    trip = TripState(arrival_deadline=130.0)
    assert not check(now=129.0).too_late(trip)
    assert check(now=130.0).too_late(trip) and check(ready=False).too_late(trip)


def test_outcomes_settle_into_the_reported_verdict():
    matched = ArrivalOutcome(True, "Printer visible.", "matched")
    assert matched.settled().failure_stage == ""
    assert matched.message == "Destination visible at the reached viewpoint. Printer visible."
    assert matched.checked(True, True) == matched
    late = matched.late().settled()
    assert (late.matched, late.object_result, late.failure_stage) == (False, "unavailable", "geometry")
    expired = ArrivalOutcome(False, "Timed out.", "", "identity", "TimeoutError").expired()
    assert (expired.reason, expired.object_result, expired.failure_stage, expired.error_type) == (
        "The arrival image expired during verification.",
        "",
        "geometry",
        "",
    )
    gone = matched.checked(False, True)
    assert (gone.matched, gone.object_result, gone.failure_stage) == (False, "unavailable", "retrieval")
    assert ArrivalOutcome(True, "Visible.").checked(False, False).object_result == ""
    unverified = ArrivalOutcome(False, "Not visible.").settled()
    assert unverified.failure_stage == "identity"
    assert unverified.message == "Reached the pose; destination unverified. Not visible."
