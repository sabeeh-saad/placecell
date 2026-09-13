"""Optional offline speech recognition with bounded audio buffering and explicit commands."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from placecell.errors import ProviderError, ValidationError
from placecell.navigation import parse_movement


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    confidence: float


class StreamingRecognizer(Protocol):
    def accept(self, pcm: bytes) -> Transcript | None:
        """Return None until an utterance ends; an empty transcript still marks an endpoint."""
        ...

    def reset(self) -> None: ...


class VoskRecognizer:
    def __init__(self, recognizer: Any) -> None:
        self._recognizer = recognizer
        recognizer.SetWords(True)

    def accept(self, pcm: bytes) -> Transcript | None:
        if not self._recognizer.AcceptWaveform(pcm):
            return None
        try:
            data = json.loads(self._recognizer.Result())
            text, words = data.get("text", ""), data.get("result", [])
            if not isinstance(text, str) or not isinstance(words, list):
                raise ValueError("expected transcript text and word scores")
            if not text.strip():
                return Transcript("", 0)
            scores = [float(w["conf"]) for w in words]
            if not scores or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
                raise ValueError("missing or invalid word confidence")
            if " ".join(w["word"] for w in words) != text:
                raise ValueError("word scores do not cover the complete transcript")
            return Transcript(text, min(scores))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ProviderError(f"invalid speech recognition result: {e}") from e

    def reset(self) -> None:
        self._recognizer.Reset()


def load_vosk(model_path: str | Path, sample_rate: int) -> VoskRecognizer:
    """Use an explicitly installed local model; never download one during robot startup."""
    from vosk import KaldiRecognizer, Model

    path = Path(model_path).expanduser()
    if not path.is_dir() or sample_rate < 8000:
        raise ValidationError("speech needs a local model directory and sample rate of at least 8000 Hz")
    return VoskRecognizer(KaldiRecognizer(Model(str(path)), sample_rate))


class SpeechGate:
    def __init__(self, wake_word: str = "robot", min_confidence: float = 0.8) -> None:
        if not wake_word.strip() or not math.isfinite(min_confidence) or not 0 < min_confidence <= 1:
            raise ValidationError("speech requires a wake phrase and confidence threshold within (0, 1]")
        self._prefix = " ".join(wake_word.casefold().split()) + " "
        self._min_confidence = min_confidence

    def command(self, transcript: Transcript) -> str | None:
        text = " ".join(transcript.text.casefold().split())
        if not math.isfinite(transcript.confidence) or not self._min_confidence <= transcript.confidence <= 1:
            return None
        if not text.startswith(self._prefix):
            return None
        command = text[len(self._prefix) :]
        try:
            parse_movement(command)
        except ValidationError:
            return None
        return command


class SpeechWorker:
    """Audio callbacks only enqueue PCM. Overflows discard the interrupted utterance."""

    def __init__(
        self,
        recognizer: StreamingRecognizer,
        gate: SpeechGate,
        publish: Callable[[str], None],
        log: Any,
        *,
        sample_rate: int = 16000,
        capacity: int = 8,
        max_audio_age_s: float = 1.0,
        max_utterance_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            sample_rate < 8000
            or capacity < 1
            or any(not math.isfinite(v) or v <= 0 for v in (max_audio_age_s, max_utterance_s))
        ):
            raise ValidationError("invalid speech buffer limits")
        self._recognizer, self._gate, self._publish, self._log = recognizer, gate, publish, log
        self._sample_rate, self._max_age, self._max_utterance = sample_rate, max_audio_age_s, max_utterance_s
        self._clock = clock
        self._queue: queue.Queue[tuple[float, bytes]] = queue.Queue(maxsize=capacity)
        self._stop, self._gap = threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, name="placecell-speech", daemon=True)
        self.dropped = 0
        self._discard_utterance = False
        self._audio_bytes = 0

    def start(self) -> None:
        self._thread.start()

    @property
    def running(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set()

    def feed(self, pcm: bytes, overflow: bool = False) -> bool:
        if self._stop.is_set():
            return False
        if overflow or not pcm or len(pcm) % 2 or len(pcm) > self._sample_rate * 2:
            self._gap.set()
            self.dropped += 1
            return False
        try:
            self._queue.put_nowait((self._clock(), pcm))
            return True
        except queue.Full:
            self._gap.set()
            self.dropped += 1
            return False

    def _reset(self) -> None:
        self._recognizer.reset()
        self._discard_utterance = True
        self._audio_bytes = 0

    def _process(self, timestamp: float, pcm: bytes) -> None:
        if self._clock() - timestamp > self._max_age:
            self._reset()
            self.dropped += 1
            return
        self._audio_bytes += len(pcm)
        if self._audio_bytes > self._sample_rate * 2 * self._max_utterance:
            self._reset()
        transcript = self._recognizer.accept(pcm)
        if transcript is not None:
            if not self._discard_utterance and not self._gap.is_set() and not self._stop.is_set():
                command = self._gate.command(transcript)
                if command is not None:
                    self._publish(command)
            self._discard_utterance = False
            self._audio_bytes = 0

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._gap.is_set():
                    self._gap.clear()
                    # Drop queued fragments too; a missing 'don't' must not become a go command.
                    for _ in range(self._queue.maxsize):
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            break
                    self._reset()
                    self._log.warning("audio interrupted; discarding the utterance, please repeat the command")
                try:
                    timestamp, pcm = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._process(timestamp, pcm)
                except Exception as e:
                    self._gap.set()
                    self._log.error(f"speech recognition failed; utterance discarded: {e}")
        except Exception as e:
            self._log.error(f"speech worker stopped: {e}")
        finally:
            self._stop.set()

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout)
        return not self._thread.is_alive()
