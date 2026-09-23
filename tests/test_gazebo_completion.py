"""Remaining integration failures: evidence versions, fixture noise and hosted latency."""

from threading import Event

import numpy as np
import pytest

from placecell.depth import Box
from placecell.verification import SceneVerdict
from tests.test_depth import snapshot
from tests.test_object_navigation import Harness


def test_scan_schedule_does_not_invalidate_checked_object_evidence(tmp_path):
    h = Harness(tmp_path, search=False)
    h.start()
    h.arrive()
    journal = h.store.objects
    before = journal.generation
    evidence = journal.evidence_generation

    def scanned(*_):
        journal.record_scan("another-scope", h.now)
        return SceneVerdict("matched", "Current scene matches the request")

    h.resolver._verifier.verify = scanned
    h.observe()
    assert journal.generation > before and journal.evidence_generation == evidence
    assert h.events[-1].state == "succeeded"


def test_scripted_detector_rejects_small_color_artifacts_and_keeps_two_displays():
    pytest.importorskip("cv2")
    from simulation.scripts.checkpoint_fixtures import displays

    pixels = np.zeros((100, 200, 3), dtype=np.int16)
    pixels[80:83, 180:185] = [10, 35, 70]  # The 5x3 artifact measured in the failed JPEG.
    assert displays(pixels) == []
    pixels[20:30, 20:40] = pixels[20:30, 120:140] = [10, 35, 70]
    assert displays(pixels) == [(20, 20, 20, 10), (120, 20, 20, 10)]


def test_absence_uses_observed_surface_without_requiring_supporting_table_to_disappear():
    box = Box(0.3, 0.3, 0.7, 0.7)
    initial = np.full((100, 100), 5.0)
    initial[30:70, 30:70] = 2.0
    initial[70:] = 1.9  # A table below the visible object, inside its enclosing sphere.
    position = snapshot(initial).locate(box)
    removed = initial.copy()
    removed[30:70, 30:70] = 5.0
    assert snapshot(removed).clear_region(position) is not None
    assert snapshot(initial).clear_region(position) is None
    for depth in (0, 1.5, 2.0):
        hidden = removed.copy()
        hidden[50, 50] = depth
        assert snapshot(hidden).clear_region(position) is None


def test_expired_attempt_retries_new_capture_without_extending_arrival_deadline(tmp_path):
    h = Harness(tmp_path, search=False)
    h.commands._arrival_max_attempts = 3
    h.start()
    h.arrive()
    deadline = h.commands._arrival_deadline

    def slow():
        h.now += 6
        h.mono += 6
        h.commands.poll()  # Keep ownership while the stale provider call finishes.
        assert h.commands.snapshot().status.state == "verifying_arrival"

    h.comparator.after = slow
    h.observe()
    assert h.commands.needs_observation
    assert h.commands._arrival_deadline == deadline
    h.comparator.after = lambda: None
    h.observe()
    assert h.events[-1].state == "succeeded" and len(h.comparator.calls) == 2


def test_retries_exhaust_budget_and_late_verdict_cannot_restart_stopped_work(tmp_path):
    h = Harness(tmp_path, search=False)
    h.commands._arrival_max_attempts = 2
    h.start()
    h.arrive()

    def slow():
        h.now += 6
        h.mono += 6

    h.comparator.after = slow
    h.observe()
    h.observe()
    assert not h.commands.busy and h.events[-1].state == "destination_unverified"
    assert len(h.comparator.calls) == 2

    (tmp_path / "stopped").mkdir()
    stopped = Harness(tmp_path / "stopped", search=False)
    stopped.commands._arrival_max_attempts = 3
    stopped.start()
    stopped.arrive()
    stopped.comparator.after = lambda: stopped.commands.handle("stop")
    stopped.observe()
    assert not stopped.commands.needs_observation and stopped.events[-1].state == "canceled"


def test_request_check_overlaps_identity_check_without_caption_embedding(tmp_path):
    h = Harness(tmp_path, search=False)
    h.start()
    h.arrive()
    comparing, checked = Event(), Event()

    def request(*_):
        assert comparing.wait(2), "Request check must run alongside the identity check"
        checked.set()
        return SceneVerdict("matched", "The requested printer is visible")

    def compare():
        comparing.set()
        assert checked.wait(2)

    def unused_caption(_):
        raise AssertionError("Arrival must use image vectors without another caption embedding request")

    h.resolver._verifier.verify = request
    h.arrival.tracker.embedder.embed_text = unused_caption
    h.comparator.after = compare
    h.observe()
    assert h.events[-1].state == "succeeded"


@pytest.mark.parametrize("fault", ["deadline", "sensor_loss"])
def test_retry_cannot_extend_deadline_or_overrule_sensor_loss(tmp_path, fault):
    h = Harness(tmp_path, search=False)
    h.commands._arrival_max_attempts = 3
    h.start()
    h.arrive()

    def slow():
        h.now += 6
        h.mono += 31 if fault == "deadline" else 6
        if fault == "sensor_loss":
            h.ready = False

    h.comparator.after = slow
    h.observe()
    assert not h.commands.busy and not h.commands.needs_observation
    assert h.events[-1].state != "succeeded" and len(h.comparator.calls) == 1
