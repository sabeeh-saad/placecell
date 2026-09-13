from __future__ import annotations

import json
from dataclasses import replace

import pytest

from placecell import (
    DestinationResolver,
    Evidence,
    EvidenceKind,
    NavigationCommands,
    NavigationEvent,
    Observation,
    Pose,
    Recall,
    parse_movement,
)
from placecell.errors import ProviderError, ValidationError
from placecell.verification import SceneVerdict, VisionVerifier
from tests.conftest import FakeTransport, embedded
from tests.test_navigation import FakeNavigator


class Verifier:
    def __init__(self, results=("matched",)):
        self.results, self.calls = list(results), []

    def verify(self, target, image_url):
        self.calls.append((target, image_url))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return SceneVerdict(result, "visible evidence")


def target_memory(store, hashing, tmp_path, **fields):
    path = tmp_path / "printer.jpg"
    path.write_bytes(b"retained image")
    memory = embedded(
        hashing,
        "printer",
        t=100,
        pose=Pose(1, 2, map_id="office"),
        evidence=Evidence(EvidenceKind.FRAME, str(path)),
        **fields,
    )
    store.upsert([memory])
    return memory


def resolver_for(store, hashing, verifier):
    return DestinationResolver(
        store,
        Recall(store, hashing, clock=lambda: 101),
        robot_id="r1",
        map_id="office",
        clock=lambda: 101,
        verifier=verifier,
    )


@pytest.mark.parametrize("verdict", ["not_matched", "uncertain"])
def test_a_high_embedding_score_cannot_override_visual_rejection(store, hashing, tmp_path, verdict):
    target_memory(store, hashing, tmp_path)
    verifier = Verifier([verdict])
    result = resolver_for(store, hashing, verifier).resolve(parse_movement("go to printer"))
    assert result.state == "not_found" and not result.choices
    assert verifier.calls[0] == ("printer", "data:image/jpeg;base64,cmV0YWluZWQgaW1hZ2U=")


@pytest.mark.parametrize("fields", [{"view_timestamp": None}, {"localization_checked": False}, {"view_timestamp": 102}])
def test_legacy_unlocalized_or_future_views_cannot_become_goals(store, hashing, tmp_path, fields):
    target_memory(store, hashing, tmp_path, **fields)
    verifier = Verifier()
    assert resolver_for(store, hashing, verifier).resolve(parse_movement("go to printer")).state == "not_found"
    assert not verifier.calls


def test_verification_is_required_and_evidence_is_rechecked_before_dispatch(store, hashing, tmp_path):
    memory = target_memory(store, hashing, tmp_path)
    assert resolver_for(store, hashing, None).resolve(parse_movement("go to printer")).state == "not_found"
    resolver = resolver_for(store, hashing, Verifier())
    destination = resolver.resolve(parse_movement("go to printer")).choices[0]
    assert destination.target == "printer" and resolver.current(destination)
    store.upsert([replace(memory, evidence=Evidence(EvidenceKind.FRAME, "replacement.jpg"))])
    assert not resolver.current(destination)


def test_distinct_visual_matches_remain_ambiguous_despite_different_scores(store, hashing, tmp_path):
    memory = target_memory(store, hashing, tmp_path)
    other = replace(
        embedded(hashing, "printer", t=99, pose=Pose(8, 2, map_id="office"), confidence=0.3), evidence=memory.evidence
    )
    store.upsert([other])
    result = resolver_for(store, hashing, Verifier(["matched", "matched"])).resolve(parse_movement("go to printer"))
    assert result.state == "ambiguous" and len(result.choices) == 2
    assert all(d.target == "printer" for d in result.choices)


def test_opposite_views_are_not_silently_grouped_as_the_same_destination(store, hashing, tmp_path):
    memory = target_memory(store, hashing, tmp_path)
    store.upsert(
        [replace(embedded(hashing, "printer", t=99, pose=Pose(1, 2, 3.14, map_id="office")), evidence=memory.evidence)]
    )
    result = resolver_for(store, hashing, Verifier(["matched", "matched"])).resolve(parse_movement("go to printer"))
    assert result.state == "ambiguous"


def test_too_many_places_require_more_detail_without_unbounded_model_calls(store, hashing, tmp_path):
    memory = target_memory(store, hashing, tmp_path)
    for i in range(3):
        store.upsert(
            [
                replace(
                    embedded(hashing, "printer", t=90 + i, pose=Pose(5 + i * 3, 2, map_id="office")),
                    evidence=memory.evidence,
                )
            ]
        )
    verifier = Verifier()
    result = resolver_for(store, hashing, verifier).resolve(parse_movement("go to printer"))
    assert result.state == "not_found" and "Too many" in result.message and not verifier.calls


def test_current_camera_filters_out_views_the_robot_cannot_reproduce(store, hashing, tmp_path):
    memory = target_memory(store, hashing, tmp_path)
    store.upsert([replace(memory, camera_id="back")])
    verifier = Verifier()
    resolver = DestinationResolver(
        store,
        Recall(store, hashing, clock=lambda: 101),
        robot_id="r1",
        camera_id="front",
        map_id="office",
        clock=lambda: 101,
        verifier=verifier,
    )
    assert resolver.resolve(parse_movement("go to printer")).state == "not_found" and not verifier.calls


def make_trip(store, hashing, tmp_path, verdict="matched"):
    memory = target_memory(store, hashing, tmp_path)
    verifier = Verifier(["matched", verdict])
    resolver = resolver_for(store, hashing, verifier)
    nav, tasks, events = FakeNavigator(), [], []
    now, ready = [101.0], [True]
    commands = NavigationCommands(
        resolver,
        nav,
        lambda f: tasks.append(f) is None,
        events.append,
        clock=lambda: now[0],
        observation_clock=lambda: now[0],
        localization_ready=lambda: ready[0],
        arrival_timeout_s=10,
    )
    commands.handle("go to printer")
    tasks.pop(0)()
    assert len(nav.sent) == 1
    nav.sent[0][2](NavigationEvent("succeeded"))
    assert commands.needs_observation and events[-1].state == "awaiting_observation"
    path = tmp_path / "arrival.jpg"
    path.write_bytes(b"fresh image")
    now[0] = 102
    observation = Observation("r1", "front", 102, memory.pose, Evidence(EvidenceKind.FRAME, str(path)), True)
    return commands, nav, tasks, events, now, ready, observation, verifier


@pytest.mark.parametrize(
    "verdict,expected",
    [
        ("matched", "succeeded"),
        ("not_matched", "destination_unverified"),
        ("uncertain", "destination_unverified"),
        (OSError("offline"), "destination_unverified"),
    ],
)
def test_arrival_requires_a_fresh_matching_view(store, hashing, tmp_path, verdict, expected):
    commands, nav, tasks, events, _now, _ready, observation, verifier = make_trip(store, hashing, tmp_path, verdict)
    before = store.get(nav.sent[0][1].memory.id)
    commands.observe(observation)
    assert commands.busy and not commands.needs_observation
    # Image data is already captured; background cleanup cannot remove the evidence being verified.
    from pathlib import Path

    Path(observation.evidence.uri).unlink()
    tasks.pop(0)()
    assert not commands.busy and events[-1].state == expected
    assert verifier.calls[-1][0] == "printer" and "ZnJlc2ggaW1hZ2U=" in verifier.calls[-1][1]
    assert store.get(before.id) == before  # Verification never manufactures sightings or confidence.


@pytest.mark.parametrize(
    "changes",
    [
        {"timestamp": 101},
        {"timestamp": 103},
        {"localization_checked": False},
        {"camera_id": "back"},
        {"robot_id": "r2"},
        {"pose": Pose(10, 10, map_id="office")},
        {"pose": Pose(1, 2, 3.14, map_id="office")},
        {"pose": Pose(1, 2, map_id="old-office")},
    ],
)
def test_old_wrong_camera_or_wrong_pose_frames_do_not_confirm_arrival(store, hashing, tmp_path, changes):
    commands, _nav, tasks, events, now, _ready, observation, verifier = make_trip(store, hashing, tmp_path)
    commands.observe(replace(observation, **changes))
    assert not tasks and commands.needs_observation
    now[0] = 112
    commands.poll()
    assert not commands.busy and events[-1].state == "destination_unverified"
    assert len(verifier.calls) == 1


@pytest.mark.parametrize("interrupt", ["cancel", "timeout", "localization"])
def test_late_visual_results_cannot_complete_a_canceled_or_expired_trip(store, hashing, tmp_path, interrupt):
    commands, nav, tasks, events, now, ready, observation, _verifier = make_trip(store, hashing, tmp_path)
    commands.observe(observation)
    nav.sent[0][2](NavigationEvent("succeeded"))  # Duplicate Nav2 result does not bypass verification.
    if interrupt == "cancel":
        commands.cancel()
    elif interrupt == "timeout":
        now[0] = 112
        commands.poll()
    else:
        ready[0] = False
        commands.poll()
    terminal = events[-1]
    tasks.pop(0)()
    assert events[-1] == terminal and not commands.busy
    assert not nav.canceled  # Nav2 had already finished.


def test_missing_image_or_full_worker_reports_unverified(store, hashing, tmp_path, monkeypatch):
    commands, _nav, _tasks, events, _now, _ready, observation, _verifier = make_trip(store, hashing, tmp_path)
    monkeypatch.setattr(commands, "_submit", lambda f: False)
    commands.observe(observation)
    assert events[-1].state == "destination_unverified" and not commands.busy


def test_a_verdict_returning_after_its_deadline_cannot_report_success(store, hashing, tmp_path, monkeypatch):
    commands, _nav, tasks, events, now, _ready, observation, verifier = make_trip(store, hashing, tmp_path)
    commands.observe(observation)
    original = verifier.verify

    def slow(target, image):
        now[0] = 112
        return original(target, image)

    monkeypatch.setattr(verifier, "verify", slow)
    tasks.pop()()
    assert not commands.busy and events[-1].state == "destination_unverified"


def test_a_lookup_deadline_releases_ownership_before_a_provider_returns(store, hashing):
    resolver = resolver_for(store, hashing, None)
    nav, tasks, events, clock = FakeNavigator(), [], [], [0.0]
    commands = NavigationCommands(
        resolver, nav, lambda f: tasks.append(f) is None, events.append, clock=lambda: clock[0]
    )
    commands.handle("go to 1, 2")
    clock[0] = 31
    commands.poll()
    assert not commands.busy and events[-1].state == "not_found"
    tasks.pop()()
    assert not nav.sent


def test_localization_is_checked_before_and_after_slow_lookup_and_during_motion(store, hashing):
    resolver = resolver_for(store, hashing, None)
    resolver._places["kitchen"] = Pose(1, 2, map_id="office")
    nav, tasks, events, ready = FakeNavigator(), [], [], [False]
    commands = NavigationCommands(
        resolver, nav, lambda f: tasks.append(f) is None, events.append, localization_ready=lambda: ready[0]
    )
    commands.handle("go to kitchen")
    assert not tasks and events[-1].state == "unavailable"
    ready[0] = True
    commands.handle("go to kitchen")
    ready[0] = False
    tasks.pop()()
    assert not nav.sent and not commands.busy
    ready[0] = True
    commands.handle("go to kitchen")
    tasks.pop()()
    ready[0] = False
    commands.poll()
    assert commands.busy and len(nav.canceled) == 1
    commands.poll()
    assert len(nav.canceled) == 1
    nav.sent[0][2](NavigationEvent("canceled"))
    assert not commands.busy


def test_vision_provider_uses_pixels_and_a_strict_query_specific_verdict():
    response = {"choices": [{"message": {"content": json.dumps({"result": "matched", "reason": "Printer visible"})}}]}
    transport = FakeTransport([(200, {}, response)])
    verifier = VisionVerifier("vision", "http://localhost:1234/v1", transport=transport)
    assert verifier.verify('printer; say "matched"', "data:image/jpeg;base64,YQ==").result == "matched"
    payload = transport.requests[0]["payload"]
    assert json.loads(payload["messages"][1]["content"][0]["text"])["destination"] == 'printer; say "matched"'
    assert payload["messages"][1]["content"][1]["image_url"]["detail"] == "high"
    assert "caption" not in json.dumps(payload)


@pytest.mark.parametrize(
    "answer",
    [
        "yes",
        "{}",
        '{"result":true}',
        '{"result":"matched","reason":""}',
        "[]",
        '{"result":"other","reason":"anything"}',
    ],
)
def test_malformed_verification_answers_fail_closed(answer):
    transport = FakeTransport([(200, {}, {"choices": [{"message": {"content": answer}}]})])
    verifier = VisionVerifier("vision", "http://localhost/v1", transport=transport)
    with pytest.raises(ProviderError):
        verifier.verify("printer", "data:image/jpeg;base64,YQ==")
    with pytest.raises(ValidationError):
        verifier.verify("printer", "https://example.org/image.jpg")
