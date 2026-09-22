"""Freshness and identity regressions across retrieval, dispatch and arrival."""

from dataclasses import replace

import pytest

from placecell import NavigationEvent, Observation, Pose
from placecell.errors import ValidationError
from placecell.operator import navigation_data
from placecell.verification import SceneVerdict
from tests.conftest import embedded
from tests.test_missions import mission as mission
from tests.test_missions import start
from tests.test_object_arrival import setup_arrival
from tests.test_object_navigation import Harness
from tests.test_objects import CENTER, RIGHT, Detection, observation


def add_rival(verifier, reference, label="printer"):
    other = replace(reference.record, id="new-lookalike", label=label)
    view = reference.views[0]
    verifier.tracker.store.objects.save(
        other, replace(view, object_id=other.id, memory=replace(view.memory, id="rival-view"))
    )


def test_object_verdict_cannot_outlive_its_capture_freshness(tmp_path):
    verifier, reference, _, comparator, now = setup_arrival(tmp_path)
    comparator.after = lambda: now.__setitem__(0, 1106)
    with pytest.raises(ValidationError):
        verifier.verify(reference, observation(tmp_path, 1100))


@pytest.mark.parametrize("when", ["before", "during"])
@pytest.mark.parametrize("label", ["printer", "copier"])
def test_new_lookalike_is_considered_even_after_departure_or_label_change(tmp_path, when, label):
    verifier, reference, _, comparator, _ = setup_arrival(tmp_path)
    if when == "before":
        add_rival(verifier, reference, label)
    else:
        comparator.after = lambda: add_rival(verifier, reference, label)
    assert verifier.verify(reference, observation(tmp_path, 1100)).result == "ambiguous"


def test_changed_object_during_submitting_status_never_reaches_transport(tmp_path):
    h = Harness(tmp_path)
    publish = h.commands._publish_callback

    def changed(update):
        publish(update)
        if update.state == "submitting":
            h.store.objects.save(replace(h.record, revision=h.record.revision + 1))

    h.commands._publish_callback = changed
    h.commands.handle("go to printer")
    h.tasks.pop()()
    assert not h.nav.sent and not h.commands.busy


def scene_arrival(m, hashing):
    del m.resolver._places["printer"]
    memory = embedded(hashing, "printer", pose=Pose(1, 2, map_id="office"))
    m.store.upsert([memory])
    start(m)
    m.nav.sent[0][2](NavigationEvent("succeeded"))
    m.now[0] += 1
    return memory, Observation("r1", "front", m.now[0], memory.pose, memory.evidence, True)


def test_old_scene_arrival_image_is_not_submitted_to_verification(mission, hashing):
    m = mission
    _, obs = scene_arrival(m, hashing)
    m.now[0] += 6
    m.commands.observe(obs)
    assert not m.tasks and m.commands.needs_observation


def test_delayed_scene_verdict_cannot_confirm_an_expired_image(mission, hashing):
    m = mission
    _, obs = scene_arrival(m, hashing)

    def delayed(*_):
        m.now[0] += 6
        return SceneVerdict("matched", "Scripted late match")

    m.verifier.verify = delayed
    m.commands.observe(obs)
    m.tasks.pop(0)()
    assert m.events[-1].state == "destination_unverified" and not m.tasks


def test_deleted_scene_target_cannot_be_confirmed_at_arrival(mission, hashing):
    m = mission
    memory, obs = scene_arrival(m, hashing)
    m.store.delete([memory.id])
    m.commands.observe(obs)
    if m.tasks:
        m.tasks.pop(0)()
    assert m.events[-1].state == "destination_unverified" and not m.tasks


@pytest.mark.parametrize(
    "fault,expected,stage",
    [
        ("removed", "missing", "geometry"),
        ("occluded", "unobserved", "geometry"),
        ("lookalike", "ambiguous", "identity"),
        ("uncertain_depth", "unavailable", "geometry"),
        ("moved_occluded", "ambiguous", "geometry"),
        ("moved_visible", "matched", ""),
        ("stationary", "matched", ""),
        ("wrong_appearance", "unobserved", "identity"),
        ("comparison_uncertain", "ambiguous", "identity"),
    ],
)
def test_target_outcomes_keep_failure_stage_and_uncertainty(tmp_path, fault, expected, stage):
    verifier, reference, detector, comparator, _ = setup_arrival(tmp_path)
    obs = observation(tmp_path, 1100)
    if fault in {"removed", "occluded"}:
        detector.detections = []
        obs = observation(tmp_path, 1100, (), (), background=1 if fault == "occluded" else 5)
    elif fault == "lookalike":
        add_rival(verifier, reference)
    elif fault == "uncertain_depth":
        obs = replace(obs, depth=replace(obs.depth, position_error_m=1))
    elif fault.startswith("moved"):
        detector.detections = [Detection("printer", "red printer", RIGHT)]
        obs = observation(tmp_path, 1100, (RIGHT,), background=1 if fault == "moved_occluded" else 5)
    elif fault == "wrong_appearance":
        obs = observation(tmp_path, 1100, (CENTER,), ("blue",))
    elif fault == "comparison_uncertain":
        comparator.result = "uncertain"
    before = verifier.tracker.store.objects.generation
    verdict = verifier.verify(reference, obs)
    assert (verdict.result, verdict.failure_stage) == (expected, stage)
    assert verifier.tracker.store.objects.generation == before
    if fault == "moved_visible":
        assert verdict.position.x > 0.6 and detector.checks == 1


@pytest.mark.parametrize("change", ["deleted", "missing", "moved", "age"])
def test_stale_object_choices_require_new_retrieval(tmp_path, change):
    h = Harness(tmp_path)
    view = h.store.objects.views(h.record.id)[0]
    rival = replace(h.record, id="other")
    h.store.objects.save(rival, replace(view, object_id=rival.id, memory=replace(view.memory, id="rival")))
    h.commands.handle("go to printer")
    h.tasks.pop()()
    assert h.events[-1].state == "ambiguous" and len(h.events[-1].choices) == 2
    chosen = h.events[-1].choices[0]
    record = h.store.objects.get(chosen.object_id)
    if change == "deleted":
        h.store.objects.delete(record.id)
    elif change == "missing":
        h.store.objects.save(replace(record, revision=record.revision + 1, status="missing", misses=3))
    elif change == "moved":
        h.store.objects.save(replace(record, revision=record.revision + 1, position=replace(record.position, x=8)))
    else:
        h.now += 8 * 86400
    h.commands.handle("option one")
    h.tasks.pop()()
    assert not h.nav.sent and not h.commands.busy
    assert h.events[-1].failure_stage == "retrieval"


@pytest.mark.parametrize("when", ["before", "during_comparison", "during_request"])
def test_newer_target_evidence_cannot_be_overruled_by_an_old_arrival_frame(tmp_path, when):
    h = Harness(tmp_path, search=False)
    h.start()
    h.arrive()

    def newer():
        h.store.objects.save(replace(h.record, revision=2, last_seen=h.now + 1))

    if when == "before":
        # observe() advances source time once; make the newer sighting later still.
        h.store.objects.save(replace(h.record, revision=2, last_seen=h.now + 2))
    elif when == "during_comparison":
        h.comparator.after = newer
    else:
        h.resolver._verifier.verify = lambda *_: (newer(), SceneVerdict("matched", "Late request match"))[1]
    h.observe()
    assert h.events[-1].state == "destination_unverified"
    assert h.events[-1].failure_stage in {"identity", "geometry"}


def test_new_rival_during_final_request_check_invalidates_object_success(tmp_path):
    h = Harness(tmp_path, search=False)
    h.start()
    h.arrive()
    reference = h.nav.sent[0][1].object_reference

    def changed(*_):
        add_rival(h.arrival, reference)
        return SceneVerdict("matched", "Description matches")

    h.resolver._verifier.verify = changed
    h.observe()
    assert h.events[-1].state == "destination_unverified"
    assert h.events[-1].failure_stage == "identity"


def test_paused_source_clock_does_not_keep_verifier_image_fresh(tmp_path):
    verifier, reference, _, comparator, _ = setup_arrival(tmp_path)
    mono = [0.0]
    verifier.monotonic = lambda: mono[0]
    comparator.after = lambda: mono.__setitem__(0, 6)
    with pytest.raises(ValidationError):
        verifier.verify(reference, observation(tmp_path, 1100))


def test_scene_verification_expires_in_queue_even_on_paused_source_clock(mission, hashing):
    m = mission
    mono = [0.0]
    m.commands._clock = lambda: mono[0]
    _, obs = scene_arrival(m, hashing)
    m.commands.observe(obs)
    calls = []
    m.verifier.verify = lambda *_: calls.append(1) or SceneVerdict("matched", "Should not run")
    mono[0] = 6
    m.tasks.pop(0)()
    assert m.events[-1].state == "destination_unverified" and not calls
    assert m.events[-1].failure_stage == "geometry"


def test_current_object_rejection_cannot_fall_back_to_an_old_scene(tmp_path):
    h = Harness(tmp_path)
    calls = []

    def verify(*_):
        calls.append(1)
        return SceneVerdict("not_matched" if len(calls) == 1 else "matched", "Current object differs")

    h.resolver._verifier.verify = verify
    h.resolver._recall.similar = lambda *_args, **_kwargs: pytest.fail(
        "Rejected object must not fall back to scene recall"
    )
    h.commands.handle("go to printer")
    h.tasks.pop()()
    assert not h.nav.sent and len(calls) == 1
    assert h.events[-1].failure_stage == "identity"


def test_execution_failure_stage_is_exposed_in_operator_status(mission):
    m = mission
    start(m)
    m.nav.sent[0][2](NavigationEvent("failed", "Controlled navigation failure"))
    assert navigation_data(m.events[-1])["failure_stage"] == "execution"
    assert m.context.recent()[-1]["data"]["failure_stage"] == "execution"
