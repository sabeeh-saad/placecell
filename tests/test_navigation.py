from __future__ import annotations

import json
from dataclasses import replace

import pytest

from placecell import (
    Correction,
    Destination,
    DestinationResolver,
    InMemoryCorrectionLog,
    NavigationCommands,
    NavigationEvent,
    NavigationPolicy,
    NavigationUpdate,
    Pose,
    Recall,
    load_named_places,
    parse_movement,
)
from placecell.errors import ValidationError
from placecell.ros2.node import navigation_payload
from tests.conftest import embedded


class FakeNavigator:
    def __init__(self):
        self.sent = []
        self.canceled = []

    def send(self, request_id, destination, callback):
        self.sent.append((request_id, destination, callback))

    def cancel(self, request_id):
        self.canceled.append(request_id)


def make_commands(store, hashing, **kwargs):
    resolver = DestinationResolver(
        store, Recall(store, hashing, clock=lambda: 3000), robot_id="r1", clock=lambda: 3000, **kwargs
    )
    navigator, tasks, events = FakeNavigator(), [], []
    commands = NavigationCommands(resolver, navigator, lambda f: tasks.append(f) is None, events.append)
    return commands, navigator, tasks, events


@pytest.mark.parametrize(
    "text",
    ["go to kitchen", "Robot, please go to the kitchen!", "can you take me to kitchen?", "navigate to kitchen please"],
)
def test_direct_commands(text):
    command = parse_movement(text)
    assert command.kind == "go" and command.destination == "kitchen"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "where is the kitchen?",
        "don't go to kitchen",
        "go to kitchen then turn left",
        "go to kitchen and stop",
        "x" * 501,
    ],
)
def test_questions_negation_and_compound_commands_do_not_move(text):
    with pytest.raises(ValidationError):
        parse_movement(text)


def test_stop_choices_and_explicit_coordinates():
    assert parse_movement("robot stop").kind == "cancel"
    assert parse_movement("please cancel navigation").kind == "cancel"
    assert parse_movement("option two").choice == 2
    assert parse_movement("go to 2.5, -1, 1.57").coordinates == (2.5, -1, 1.57)
    assert parse_movement("go to .5, -1").coordinates == (0.5, -1, 0)
    with pytest.raises(ValidationError):
        parse_movement("go to " + "9" * 320 + ", 1")


def test_named_places_and_coordinates_are_scoped_to_the_active_map(store, hashing, tmp_path):
    path = tmp_path / "places.json"
    path.write_text(json.dumps({"The Kitchen": {"x": 1, "y": 2, "map_id": "office"}}))
    places = load_named_places(path)
    commands, navigator, tasks, events = make_commands(store, hashing, map_id="office", places=places)
    commands.handle("go to kitchen")
    tasks.pop()()
    assert navigator.sent[0][1].pose == Pose(1, 2, map_id="office")
    assert navigator.sent[0][1].source == "named_place"
    navigator.sent[0][2](NavigationEvent("succeeded"))
    commands.handle("go to 3, 4")
    tasks.pop()()
    assert navigator.sent[-1][1].pose == Pose(3, 4, map_id="office")
    other, nav, tasks, events = make_commands(store, hashing, map_id="warehouse", places=places)
    other.handle("go to kitchen")
    tasks.pop()()
    assert not nav.sent and events[-1].state == "not_found"


def test_memory_resolution_excludes_summaries_other_maps_and_other_robots(store, hashing):
    target = embedded(hashing, "printer", pose=Pose(2, 3, 1.2, map_id="office"))
    rows = [
        target,
        embedded(hashing, "printer", t=2000, role="summary", x=8),
        embedded(hashing, "printer", t=2001, pose=Pose(9, 9, map_id="warehouse")),
        embedded(hashing, "printer", t=2002, robot="other", pose=Pose(5, 5, map_id="office")),
    ]
    store.upsert(rows)
    commands, navigator, tasks, events = make_commands(store, hashing, map_id="office")
    commands.handle("go to printer")
    tasks.pop()()
    assert navigator.sent[0][1].pose == target.pose
    assert navigator.sent[0][1].memory.id == target.id
    assert commands.busy
    navigator.sent[0][2](NavigationEvent("navigating", distance_remaining=2.0))
    assert events[-1].distance_remaining == 2
    navigator.sent[0][2](NavigationEvent("succeeded"))
    assert not commands.busy


@pytest.mark.parametrize(
    "fields", [{"caption": "unrelated wall"}, {"confidence": 0.01}, {"superseded": True}, {"last_seen": 900000}]
)
def test_unreliable_memories_do_not_become_goals(store, hashing, fields):
    row = embedded(hashing, "printer", **{k: v for k, v in fields.items() if k != "caption"})
    if "caption" in fields:
        row = embedded(hashing, fields["caption"])
    store.upsert([row])
    commands, navigator, tasks, events = make_commands(store, hashing)
    commands.handle("go to printer")
    tasks.pop()()
    assert not navigator.sent and events[-1].state == "not_found"
    assert not commands.busy


def test_age_limit_and_feedback_apply_at_resolution_and_dispatch(store, hashing):
    row = embedded(hashing, "printer")
    store.upsert([row])
    log = InMemoryCorrectionLog()
    recall = Recall(store, hashing, clock=lambda: 3000, corrections=log)
    resolver = DestinationResolver(store, recall, robot_id="r1", clock=lambda: 3000)
    destination = resolver.resolve(parse_movement("go to printer")).choices[0]
    for _ in range(3):
        log.record(Correction(row.id, "wrong"))
    assert not resolver.current(destination)
    assert resolver.resolve(parse_movement("go to printer")).state == "not_found"
    old = DestinationResolver(store, Recall(store, hashing, clock=lambda: 900000), robot_id="r1", clock=lambda: 900000)
    assert old.resolve(parse_movement("go to printer")).state == "not_found"


def test_ambiguous_places_require_an_explicit_selection(store, hashing):
    a, b = embedded(hashing, "printer", x=1), embedded(hashing, "printer", t=1001, x=8)
    store.upsert([a, b])
    commands, navigator, tasks, events = make_commands(store, hashing)
    commands.handle("go to printer")
    tasks.pop()()
    assert not navigator.sent and events[-1].state == "ambiguous"
    options = events[-1].choices
    assert len(options) == 2
    commands.handle("option two")
    tasks.pop()()
    assert navigator.sent[0][1] == options[1]
    assert navigator.sent[0][1].pose == options[1].pose


def test_nearby_sightings_do_not_create_duplicate_destination_options(store, hashing):
    store.upsert([embedded(hashing, "printer", t=i, x=0.2 * i) for i in range(3)])
    commands, navigator, tasks, _events = make_commands(store, hashing)
    commands.handle("go to printer")
    tasks.pop()()
    assert len(navigator.sent) == 1


@pytest.mark.parametrize("change", ["caption", "vector", "pose", "delete"])
def test_changed_destination_options_are_rejected(store, hashing, change):
    rows = [embedded(hashing, "printer", t=t, x=x) for t, x in ((1000, 1), (1001, 8))]
    store.upsert(rows)
    commands, navigator, tasks, events = make_commands(store, hashing)
    commands.handle("go to printer")
    tasks.pop()()
    selected = events[-1].choices[0].memory
    if change == "delete":
        store.delete([selected.id])
    else:
        fields = {
            "caption": {"caption": "wall"},
            "vector": {"embedding": hashing.embed_text(["wall"])[0]},
            "pose": {"pose": Pose(9, 9)},
        }[change]
        store.upsert([replace(selected, **fields)])
    commands.handle("option one")
    tasks.pop()()
    assert not navigator.sent and events[-1].state == "not_found"


def test_cancel_during_resolution_prevents_a_late_goal(store, hashing):
    commands, navigator, tasks, events = make_commands(store, hashing, places={"kitchen": Pose(1, 2)})
    commands.handle("go to kitchen")
    commands.handle("stop")
    tasks.pop()()
    assert not commands.busy and not navigator.sent and events[-1].state == "canceled"
    commands.handle("option one")
    assert events[-1].state == "invalid"


def test_cancel_while_provider_is_running_and_lookup_failure(store, hashing, monkeypatch):
    commands, navigator, tasks, events = make_commands(store, hashing)

    def resolve(command):
        commands.cancel()
        raise OSError("provider unavailable")

    monkeypatch.setattr(commands._resolver, "resolve", resolve)
    commands.handle("go to kitchen")
    tasks.pop()()
    assert not navigator.sent and events[-1].state == "canceled"
    monkeypatch.setattr(commands._resolver, "resolve", lambda command: (_ for _ in ()).throw(OSError("unavailable")))
    commands.handle("go to kitchen")
    tasks.pop()()
    assert events[-1].state == "not_found" and not commands.busy


def test_a_trip_stays_owned_until_nav2_confirms_a_terminal_result(store, hashing):
    commands, navigator, tasks, events = make_commands(store, hashing, places={"kitchen": Pose(1, 2)})
    commands.handle("where is kitchen?")
    assert events[-1].state == "invalid"
    commands.handle("go to kitchen")
    tasks.pop()()
    request_id, _, callback = navigator.sent[0]
    commands.handle("go to kitchen")
    assert events[-1].state == "busy" and len(navigator.sent) == 1
    commands.handle("stop")
    assert navigator.canceled == [request_id] and commands.busy
    callback(NavigationEvent("navigating"))
    assert events[-1].state == "canceling"
    callback(NavigationEvent("cancel_failed"))
    assert commands.busy
    callback(NavigationEvent("uncertain"))
    assert commands.busy
    callback(NavigationEvent("canceled"))
    assert not commands.busy
    callback(NavigationEvent("navigating"))
    assert events[-1].state == "canceled"
    commands.close()
    commands.handle("go to kitchen")
    assert not tasks and events[-1].state == "busy"


def test_worker_and_transport_failures_do_not_queue_an_unowned_trip(store, hashing, monkeypatch):
    commands, navigator, tasks, events = make_commands(store, hashing, places={"kitchen": Pose(1, 2)})
    monkeypatch.setattr(commands, "_submit", lambda f: False)
    commands.handle("go to kitchen")
    assert not commands.busy and events[-1].state == "busy"
    monkeypatch.setattr(commands, "_submit", lambda f: tasks.append(f) is None)
    monkeypatch.setattr(navigator, "send", lambda *args: (_ for _ in ()).throw(OSError("transport")))
    commands.handle("go to kitchen")
    tasks.pop()()
    assert commands.busy and events[-1].state == "uncertain"
    monkeypatch.setattr(navigator, "cancel", lambda *args: (_ for _ in ()).throw(OSError("transport")))
    commands.cancel()
    assert commands.busy and events[-1].state == "uncertain"


def test_configuration_and_status_payload(store, hashing, tmp_path):
    for kwargs in ({"min_similarity": 0}, {"min_confidence": float("nan")}, {"max_age_s": 0}, {"candidates": 1}):
        with pytest.raises(ValidationError):
            NavigationPolicy(**kwargs)
    with pytest.raises(ValidationError):
        DestinationResolver(store, Recall(store, hashing), robot_id="")
    resolver = DestinationResolver(store, Recall(store, hashing), robot_id="r1")
    with pytest.raises(ValidationError):
        resolver.resolve(parse_movement("stop"))
    assert not resolver.current(Destination("bad map", Pose(0, 0, map_id="other"), "named_place"))
    path = tmp_path / "places.json"
    for data in (
        [],
        {"": {}},
        {"kitchen": []},
        {"kitchen": {"x": "no", "y": 1}},
        {"Kitchen": {"x": 0, "y": 0}, "kitchen": {"x": 1, "y": 1}},
    ):
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            load_named_places(path)
    destination = Destination("kitchen", Pose(1, 2, 0.5, map_id="office"), "named_place")
    payload = json.loads(navigation_payload(NavigationUpdate("r", "ambiguous", "choose", choices=(destination,))))
    assert payload["choices"][0]["option"] == 1 and payload["choices"][0]["map_id"] == "office"
    payload = json.loads(
        navigation_payload(NavigationUpdate("r", "navigating", "", destination, distance_remaining=1.5))
    )
    assert payload["destination"]["yaw"] == 0.5 and payload["distance_remaining"] == 1.5


def test_delayed_requests_and_choices_expire(store, hashing, monkeypatch):
    store.upsert([embedded(hashing, "printer", t=t, x=x) for t, x in ((1000, 1), (1001, 8))])
    commands, navigator, tasks, events = make_commands(store, hashing)
    clock = [0.0]
    monkeypatch.setattr(commands, "_clock", lambda: clock[0])
    commands.handle("go to printer")
    clock[0] = 31
    tasks.pop()()
    assert not commands.busy and not navigator.sent and "expired" in events[-1].message
    commands.handle("go to printer")
    tasks.pop()()
    assert events[-1].state == "ambiguous"
    clock[0] += 31
    commands.handle("option one")
    assert events[-1].state == "invalid" and not tasks
    original = commands._resolver.resolve

    def delayed(command):
        result = original(command)
        clock[0] += 31
        return result

    monkeypatch.setattr(commands._resolver, "resolve", delayed)
    commands.handle("go to printer")
    tasks.pop()()
    assert not navigator.sent and "timed out" in events[-1].message
    with pytest.raises(ValidationError):
        NavigationCommands(commands._resolver, navigator, lambda f: True, events.append, request_timeout_s=0)
