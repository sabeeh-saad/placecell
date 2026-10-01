"""Status numbering, attribution and context rows, without a controller."""

from __future__ import annotations

import pytest

from placecell.navigation import NavigationUpdate
from placecell.navigation_state import NavState
from placecell.navigation_status import StatusPublisher


class Context:
    def __init__(self, fail: bool = False) -> None:
        self.rows: list[tuple[str, str, dict]] = []
        self.fail = fail

    def record(self, request_id, kind, payload):
        if self.fail:
            raise OSError("disk full")
        self.rows.append((request_id, kind, payload))


def publisher() -> StatusPublisher:
    return StatusPublisher("instance", NavigationUpdate("", "idle", "No navigation request is active."))


def test_numbering_stamps_instance_and_sequence_and_only_state_updates_replace_the_snapshot():
    status = publisher()
    first = status.number(NavigationUpdate("r1", "invalid", "Bad"), state_update=False)
    second = status.number(NavigationUpdate("r2", "resolving", "Looking"), state_update=True)
    assert (first.instance_id, first.sequence, second.sequence) == ("instance", 1, 2)
    assert status.snapshot == second and status.sequence == 2


def test_an_enum_state_is_refused_before_it_is_published():
    with pytest.raises(AssertionError, match="plain str"):
        publisher().number(NavigationUpdate("r", NavState.CANCELED, "Stopped"), state_update=False)


def test_attribution_defaults_execution_failures_and_names_the_mission():
    update = StatusPublisher.attribute(NavigationUpdate("r", "canceled", "Stopped"), "m", 2, ("a", "b"))
    assert (update.failure_stage, update.mission_id, update.mission_step) == ("execution", "m", 2)
    assert update.mission_destinations == ("a", "b")
    kept = StatusPublisher.attribute(NavigationUpdate("r", "failed", "", failure_stage="geometry"), "", 1, ("a",))
    assert (kept.failure_stage, kept.mission_id, kept.mission_step) == ("geometry", "", 0)


def test_each_outcome_is_recorded_once_and_a_failed_write_is_reported():
    status, context = publisher(), Context()
    update = NavigationUpdate("r", "navigating", "Moving")
    assert status.record(update, context) is update
    status.record(update, context)
    assert [row[2]["state"] for row in context.rows] == ["navigating"]
    failed = status.record(NavigationUpdate("r", "succeeded", "Done"), Context(fail=True))
    assert failed.message == "Done Context persistence failed: disk full" and not status.context_ok
    assert status.record_instruction(Context(), "r2", "go to the printer") is None and status.context_ok
    assert isinstance(status.record_instruction(Context(fail=True), "r3", "stop"), OSError)
    assert not status.context_ok
