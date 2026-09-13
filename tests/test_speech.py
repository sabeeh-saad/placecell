from __future__ import annotations

import json
import sys
from threading import Event
from types import SimpleNamespace as Obj

import pytest

from placecell import DestinationResolver, Ingester, NavigationCommands, Observation, Pose, Recall
from placecell.errors import ProviderError, ValidationError
from placecell.speech import SpeechGate, SpeechWorker, Transcript, VoskRecognizer, load_vosk
from tests.conftest import FakeCaptioner, embedded, frame
from tests.test_navigation import FakeNavigator, MatchingVerifier
from tests.test_ros2_bridge import _Log


class Recognizer:
    def __init__(self, results=()):
        self.results = list(results)
        self.resets = 0

    def accept(self, pcm):
        return self.results.pop(0) if self.results else None

    def reset(self):
        self.resets += 1


def test_speech_gate_requires_complete_direct_commands_and_a_wake_phrase():
    gate = SpeechGate()
    assert gate.command(Transcript("robot go to kitchen", 0.9)) == "go to kitchen"
    assert gate.command(Transcript("robot stop", 0.9)) == "stop"
    assert gate.command(Transcript("robot option two", 0.9)) == "option two"
    for text, confidence in (
        ("go to kitchen", 1),
        ("robotics go to kitchen", 1),
        ("robot where is kitchen", 1),
        ("robot don't go to kitchen", 1),
        ("robot go to kitchen if the door is open", 1),
        ("robot go to kitchen", 0.5),
        ("robot stop", float("nan")),
        ("robot stop", 2),
    ):
        assert gate.command(Transcript(text, confidence)) is None
    assert SpeechGate("hello robot").command(Transcript("hello robot go to kitchen", 1)) == "go to kitchen"
    for kwargs in ({"wake_word": " "}, {"min_confidence": 0}, {"min_confidence": float("nan")}):
        with pytest.raises(ValidationError):
            SpeechGate(**kwargs)


def test_vosk_adapter_uses_only_final_results_with_complete_word_scores():
    class Vosk:
        def SetWords(self, enabled):  # noqa: N802 - provider API
            self.words_enabled = enabled

        def AcceptWaveform(self, pcm):  # noqa: N802 - provider API
            return pcm == b"final"

        def Result(self):  # noqa: N802 - provider API
            return self.result

        def Reset(self):  # noqa: N802 - provider API
            self.reset = True

    backend = Vosk()
    backend.result = json.dumps(
        {"text": "robot stop", "result": [{"word": "robot", "conf": 0.95}, {"word": "stop", "conf": 0.9}]}
    )
    adapter = VoskRecognizer(backend)
    assert backend.words_enabled and adapter.accept(b"partial") is None
    assert adapter.accept(b"final") == Transcript("robot stop", 0.9)
    adapter.reset()
    assert backend.reset
    backend.result = '{"text": ""}'
    assert adapter.accept(b"final") == Transcript("", 0)
    for data in (
        [],
        {"text": []},
        {"text": "robot stop"},
        {"text": "robot stop", "result": [{"word": "stop", "conf": 1}]},
        {"text": "stop", "result": [{"word": "stop", "conf": float("nan")}]},
        {"text": "stop", "result": [{"word": "stop", "conf": -1}]},
    ):
        backend.result = json.dumps(data)
        with pytest.raises(ProviderError):
            adapter.accept(b"final")


def test_local_model_factory_never_chooses_a_download(monkeypatch, tmp_path):
    calls = []
    backend = Obj(SetWords=lambda enabled: calls.append(enabled))
    monkeypatch.setitem(
        sys.modules,
        "vosk",
        Obj(Model=lambda path: calls.append(path) or "model", KaldiRecognizer=lambda model, rate: backend),
    )
    load_vosk(tmp_path, 16000)
    assert calls == [str(tmp_path), True]
    with pytest.raises(ValidationError):
        load_vosk(tmp_path / "missing", 16000)


def test_stale_or_overlong_audio_discards_the_entire_utterance():
    recognizer = Recognizer([None, Transcript("robot go to kitchen", 1), Transcript("robot stop", 1)])
    out = []
    worker = SpeechWorker(recognizer, SpeechGate(), out.append, _Log(), max_utterance_s=0.1, clock=lambda: 10)
    worker._process(8, b"\0\0")
    assert recognizer.resets == 1
    worker._process(10, b"\0" * 4000)
    assert recognizer.resets == 2
    worker._process(10, b"\0\0")
    assert not out
    worker._process(10, b"\0\0")
    assert out == ["stop"]


def test_audio_queue_overflow_drops_fragments_and_recovers_at_a_new_utterance():
    ready, published = Event(), Event()
    out = []

    class NotifyingRecognizer(Recognizer):
        def reset(self):
            super().reset()
            ready.set()

    recognizer = NotifyingRecognizer([Transcript("robot go to kitchen", 1), Transcript("robot stop", 1)])
    worker = SpeechWorker(
        recognizer, SpeechGate(), lambda command: (out.append(command), published.set()), _Log(), capacity=1
    )
    assert not worker.running
    assert worker.feed(b"\0\0")
    assert not worker.feed(b"\0\0")
    worker.start()
    try:
        assert ready.wait(2)
        assert worker.feed(b"\0\0")  # finish and discard the interrupted utterance
        for _ in range(100):
            if not recognizer.results or len(recognizer.results) == 1:
                break
            Event().wait(0.01)
        assert not out
        assert worker.feed(b"\0\0")
        assert published.wait(2) and out == ["stop"]
    finally:
        assert worker.stop()
    assert not worker.running and not worker.feed(b"\0\0")
    assert worker.dropped == 1


def test_recognizer_errors_discard_audio_and_shutdown_does_not_flush_a_partial_command():
    failed, reset = Event(), Event()

    class FailingRecognizer(Recognizer):
        def accept(self, pcm):
            raise ProviderError("bad audio")

        def reset(self):
            reset.set()

    class Log(_Log):
        def error(self, message):
            super().error(message)
            failed.set()

    out = []
    worker = SpeechWorker(FailingRecognizer(), SpeechGate(), out.append, Log())
    worker.feed(b"\0\0")
    worker.start()
    try:
        assert failed.wait(2) and reset.wait(2)
    finally:
        worker.stop()
    assert out == []
    worker = SpeechWorker(Recognizer([Transcript("robot go to kitchen", 1)]), SpeechGate(), out.append, _Log())
    assert not worker.feed(b"odd") and not worker.feed(b"\0\0", overflow=True)
    worker.feed(b"\0\0")
    assert worker.stop() and out == []
    for kwargs in ({"capacity": 0}, {"sample_rate": 4000}, {"max_audio_age_s": 0}):
        with pytest.raises(ValidationError):
            SpeechWorker(Recognizer(), SpeechGate(), out.append, _Log(), **kwargs)


def test_spoken_destination_reaches_navigation_while_observations_keep_updating(store, hashing, monkeypatch):
    monkeypatch.setattr("placecell.navigation.data_url", lambda uri: "data:image/jpeg;base64,YQ==")
    target = embedded(hashing, "printer")
    store.upsert([target])
    resolver = DestinationResolver(
        store,
        Recall(store, hashing, clock=lambda: 3000),
        robot_id="r1",
        clock=lambda: 3000,
        verifier=MatchingVerifier(),
    )
    navigator, tasks, statuses = FakeNavigator(), [], []
    commands = NavigationCommands(resolver, navigator, lambda f: tasks.append(f) is None, statuses.append)
    worker = SpeechWorker(
        Recognizer([Transcript("robot go to printer", 1)]), SpeechGate(), commands.handle, _Log(), clock=lambda: 3000
    )
    worker._process(3000, b"\0\0")
    tasks.pop()()
    assert commands.busy and navigator.sent[0][1].pose == target.pose
    before = store.get(target.id).observations
    report = Ingester(hashing, store, FakeCaptioner("printer")).ingest(
        [
            Observation("r1", "front", 3000, Pose(0, 0), frame("new-view.jpg")),
        ]
    )
    assert report.merged == 1 and commands.busy
    assert store.get(target.id).observations == before + 1
    assert store.get(target.id).last_seen == 3000
    assert store.refinements.pending()[0]["memory_id"] == target.id
