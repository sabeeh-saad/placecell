"""Resolve explicit movement commands to map-scoped destinations and manage one trip."""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from placecell.errors import ValidationError
from placecell.memory import Memory, Pose
from placecell.objects import ObjectRecall
from placecell.pipeline import Observation
from placecell.providers.captioning import data_url
from placecell.retrieval import RankedMemory, Recall
from placecell.store.base import Filter, VectorStore
from placecell.verification import SceneVerdict, SceneVerifier


@dataclass(frozen=True, slots=True)
class MovementCommand:
    kind: Literal["go", "cancel", "choose"]
    destination: str = ""
    coordinates: tuple[float, float, float] | None = None
    choice: int = 0


def parse_movement(text: str) -> MovementCommand:
    """Accept direct movement requests; questions, negation and compound commands do not move the robot."""
    if not text.strip() or len(text) > 500:
        raise ValidationError("Say 'go to <place>', 'option one', or 'stop'.")
    text = " ".join(text.casefold().split()).rstrip(".!?")
    text = re.sub(r"^robot[ ,]+", "", text)
    text = re.sub(r"^please\s+|\s+please$", "", text)
    if text in {"stop", "stop moving", "cancel", "cancel navigation"}:
        return MovementCommand("cancel")
    choices = {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}
    choice = re.fullmatch(r"(?:option|choice) (one|two|three|[123])", text)
    if choice:
        return MovementCommand("choose", choice=choices[choice[1]])
    match = re.fullmatch(r"(?:(?:can|could) you )?(?:go to|navigate to|take me to) (.+)", text)
    if not match:
        raise ValidationError("Say 'go to <place>', 'option one', or 'stop'. Questions use the ask topic.")
    destination = match[1]
    if re.search(
        r"\b(?:not|don't|do not|then|instead|except|if|unless|before|after|but|or)\b|;"
        r"|\band\s+(?:go|stop|turn|take|navigate)\b",
        destination,
    ):
        raise ValidationError("Give one destination at a time.")
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    coordinates = re.fullmatch(rf"({number})\s*,\s*({number})(?:\s*,\s*({number}))?", destination)
    if coordinates:
        x, y, yaw = float(coordinates[1]), float(coordinates[2]), float(coordinates[3] or 0)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValidationError("Coordinates must be finite metres and yaw in radians.")
        return MovementCommand("go", destination, (x, y, yaw))
    return MovementCommand("go", destination.removeprefix("the "))


@dataclass(frozen=True, slots=True)
class NavigationPolicy:
    min_similarity: float = 0.5
    min_confidence: float = 0.2
    max_age_s: float = 7 * 24 * 3600.0
    same_place_radius_m: float = 1.0
    candidates: int = 12
    verification_candidates: int = 3

    def __post_init__(self) -> None:
        for value in (self.min_similarity, self.min_confidence):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValidationError("navigation score thresholds must be finite and within (0, 1]")
        if any(not math.isfinite(v) or v <= 0 for v in (self.max_age_s, self.same_place_radius_m)):
            raise ValidationError("navigation age and distance limits must be finite and positive")
        if not isinstance(self.candidates, int) or not 2 <= self.candidates <= 50:
            raise ValidationError("navigation candidates must be within 2..50")
        if (
            not isinstance(self.verification_candidates, int)
            or not 1 <= self.verification_candidates <= self.candidates
        ):
            raise ValidationError("verification candidates must fit within the retrieval limit")


@dataclass(frozen=True, slots=True)
class Destination:
    label: str
    pose: Pose
    source: Literal["memory", "named_place", "coordinates"]
    memory: Memory | None = field(default=None, repr=False, compare=False)
    target: str = ""
    object_id: str = ""
    object_revision: int = 0


@dataclass(frozen=True, slots=True)
class Resolution:
    state: Literal["resolved", "ambiguous", "not_found"]
    message: str
    choices: tuple[Destination, ...] = ()


def load_named_places(path: str | Path) -> dict[str, Pose]:
    """Load operator-defined navigation poses: {name: {x, y, yaw, frame_id, map_id}}."""
    try:
        data = json.loads(Path(path).expanduser().read_text())
        if not isinstance(data, dict):
            raise ValidationError("named places must be a JSON object")
        result = {}
        for name, pose in data.items():
            if not name.strip() or not isinstance(pose, dict):
                raise ValidationError("each named place needs a name and pose")
            key = " ".join(name.casefold().split()).removeprefix("the ")
            if key in result:
                raise ValidationError(f"duplicate named place: {key}")
            result[key] = Pose(**pose)
        return result
    except (TypeError, ValueError) as e:
        raise ValidationError(f"invalid named places: {e}") from e


class DestinationResolver:
    def __init__(
        self,
        store: VectorStore,
        recall: Recall,
        *,
        robot_id: str,
        camera_id: str = "",
        frame_id: str = "map",
        map_id: str = "",
        places: Mapping[str, Pose] | None = None,
        policy: NavigationPolicy | None = None,
        verifier: SceneVerifier | None = None,
        objects: ObjectRecall | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not robot_id:
            raise ValidationError("navigation robot_id must not be empty")
        self._store, self._recall = store, recall
        self._scope = Filter(
            robot_id=robot_id, camera_id=camera_id or None, frame_id=frame_id, map_id=map_id, role="episodic"
        )
        self._origin = Pose(0, 0, frame_id=frame_id, map_id=map_id)
        self._places = {" ".join(k.casefold().split()).removeprefix("the "): v for k, v in (places or {}).items()}
        self._policy, self._clock = policy or NavigationPolicy(), clock
        self._verifier = verifier
        self._objects = objects

    def resolve(self, command: MovementCommand) -> Resolution:
        if command.kind != "go":
            raise ValidationError("only a go command has a destination")
        if command.coordinates is not None:
            x, y, yaw = command.coordinates
            pose = Pose(x, y, yaw, self._origin.frame_id, self._origin.map_id)
            return Resolution(
                "resolved", "Using the requested coordinates.", (Destination(command.destination, pose, "coordinates"),)
            )
        if command.destination in self._places:
            pose = self._places[command.destination]
            if not pose.same_frame(self._origin):
                return Resolution("not_found", "That named place belongs to a different map.")
            return Resolution(
                "resolved", "Using the named place.", (Destination(command.destination, pose, "named_place"),)
            )
        if self._verifier is None:
            return Resolution("not_found", "Visual destination verification is not configured.")
        if self._objects is not None:
            object_result = self._resolve_object(command.destination)
            if object_result is not None:
                return object_result
        hits = self._recall.similar(command.destination, k=self._policy.candidates, where=self._scope)
        now, p = self._clock(), self._policy
        hits = [
            h
            for h in hits
            if h.similarity is not None
            and h.similarity >= p.min_similarity
            and h.confidence >= p.min_confidence
            and 0 <= now - h.memory.last_seen <= p.max_age_s
            and h.memory.view_timestamp is not None
            and 0 <= now - h.memory.view_timestamp <= p.max_age_s
            and h.memory.localization_checked
            and h.memory.evidence is not None
        ]
        if not hits:
            return Resolution(
                "not_found", "I don't have a sufficiently recent, reliable location for that destination."
            )
        # Check distinct places rather than spending the entire budget on near-duplicate views.
        candidates: list[RankedMemory] = []
        for hit in hits:
            if all(
                hit.memory.camera_id != c.memory.camera_id
                or hit.memory.pose.distance_to(c.memory.pose) > p.same_place_radius_m
                or hit.memory.pose.heading_difference(c.memory.pose) > 0.5
                for c in candidates
            ):
                candidates.append(hit)
        # Do not silently ignore another plausible place just because the verification budget is small.
        if len(candidates) > p.verification_candidates:
            return Resolution("not_found", "Too many possible places. Please describe the destination more precisely.")
        choices_list = []
        for hit in candidates:
            memory = hit.memory
            assert memory.evidence is not None
            verdict = self.verify(command.destination, data_url(memory.evidence.uri))
            if verdict.result == "uncertain":
                return Resolution(
                    "not_found", "The images do not clearly identify the destination. Please give more detail."
                )
            if verdict.result == "matched":
                choices_list.append(Destination(memory.caption, memory.pose, "memory", memory, command.destination))
        choices = tuple(choices_list)
        if not choices:
            return Resolution("not_found", "The retrieved images do not show the requested destination.")
        if len(choices) > 1:
            return Resolution(
                "ambiguous", "I found several places. Say 'option one', 'option two', or give more detail.", choices
            )
        return Resolution("resolved", "Navigating to the remembered observation viewpoint.", choices)

    def _resolve_object(self, target: str) -> Resolution | None:
        assert self._objects is not None
        p = self._policy
        hits = self._objects.similar(
            target,
            robot_id=self._scope.robot_id or "",
            camera_id=self._scope.camera_id or "",
            frame_id=self._origin.frame_id,
            map_id=self._origin.map_id,
            k=p.candidates + 1,
            max_age_s=p.max_age_s,
        )
        hits = [h for h in hits if h.similarity >= p.min_similarity]
        if not hits:
            return None
        if len(hits) > p.verification_candidates:
            return Resolution("not_found", "Too many possible objects. Please describe the destination more precisely.")
        choices = []
        for hit in hits:
            memory = hit.view.memory
            verdict = self.verify(target, hit.view.image_url())
            if verdict.result == "not_matched":
                continue
            if verdict.result == "uncertain" or hit.object.status != "present" or hit.object.misses:
                return Resolution(
                    "not_found", "That object's identity or current presence is uncertain. Please revisit it."
                )
            if (
                not memory.localization_checked
                or self._recall.confidence(memory) < p.min_confidence
                or memory.view_timestamp is None
                or not 0 <= self._clock() - memory.view_timestamp <= p.max_age_s
            ):
                return Resolution("not_found", "That object has no reliable recent observation viewpoint.")
            choices.append(
                Destination(memory.caption, memory.pose, "memory", memory, target, hit.object.id, hit.object.revision)
            )
        if not choices:
            return None
        if len(choices) > 1:
            return Resolution(
                "ambiguous",
                "I found several objects. Say 'option one', 'option two', or give more detail.",
                tuple(choices),
            )
        return Resolution(
            "resolved", "Navigating to the object's latest verified observation viewpoint.", tuple(choices)
        )

    def verify(self, target: str, image_url: str) -> SceneVerdict:
        if self._verifier is None:
            return SceneVerdict("uncertain", "Visual verification is unavailable.")
        return self._verifier.verify(target, image_url)

    def current(self, destination: Destination) -> bool:
        """Recheck the source just before dispatch; later content changes must not silently change the goal."""
        if not destination.pose.same_frame(self._origin):
            return False
        if destination.memory is None:
            return True
        if destination.object_id:
            record = self._store.objects.get(destination.object_id)
            views = self._store.objects.views(destination.object_id, include_crops=False, limit=1)
            return bool(
                record is not None
                and record.revision == destination.object_revision
                and record.status == "present"
                and record.misses == 0
                and record.robot_id == self._scope.robot_id
                and record.camera_id == self._scope.camera_id
                and record.frame_id == self._origin.frame_id
                and record.map_id == self._origin.map_id
                and 0 <= self._clock() - record.last_seen <= self._policy.max_age_s
                and views
                and views[0].memory.id == destination.memory.id
                and views[0].memory == destination.memory
                and views[0].memory.same_embeddings(destination.memory)
                and views[0].memory.pose == destination.pose
                and views[0].memory.localization_checked
                and self._recall.confidence(views[0].memory) >= self._policy.min_confidence
            )
        before = destination.memory
        current = self._store.get(before.id)
        return (
            current is not None
            and self._scope.matches(current)
            and current.pose == destination.pose
            and current.caption == before.caption
            and current.evidence == before.evidence
            and current.view_timestamp == before.view_timestamp
            and current.view_timestamp is not None
            and 0 <= self._clock() - current.view_timestamp <= self._policy.max_age_s
            and current.localization_checked
            and current.embedding is not None
            and before.embedding is not None
            and current.same_embeddings(before)
            and 0 <= self._clock() - current.last_seen <= self._policy.max_age_s
            and self._recall.confidence(current) >= self._policy.min_confidence
        )


@dataclass(frozen=True, slots=True)
class NavigationEvent:
    state: str
    message: str = ""
    distance_remaining: float | None = None


class Navigator(Protocol):
    def send(self, request_id: str, destination: Destination, callback: Callable[[NavigationEvent], None]) -> None: ...

    def cancel(self, request_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class NavigationUpdate:
    request_id: str
    state: str
    message: str
    destination: Destination | None = None
    choices: tuple[Destination, ...] = ()
    distance_remaining: float | None = None


class NavigationCommands:
    """One active trip. Cancellation bypasses slow resolution; stopped work is never replayed."""

    def __init__(
        self,
        resolver: DestinationResolver,
        navigator: Navigator,
        submit: Callable[[Callable[[], None]], bool],
        publish: Callable[[NavigationUpdate], None],
        *,
        request_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        observation_clock: Callable[[], float] = time.time,
        localization_ready: Callable[[], bool] = lambda: True,
        arrival_timeout_s: float = 30.0,
    ) -> None:
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValidationError("command timeout must be finite and positive")
        if not math.isfinite(arrival_timeout_s) or arrival_timeout_s <= 0:
            raise ValidationError("arrival timeout must be finite and positive")
        self._resolver, self._navigator, self._submit, self._publish = resolver, navigator, submit, publish
        self._timeout, self._clock = request_timeout_s, clock
        self._requested_at = self._choices_at = 0.0
        self._lock = threading.RLock()
        self._active: str | None = None
        self._state = "idle"
        self._destination: Destination | None = None
        self._choices: tuple[Destination, ...] = ()
        self._closed = False
        self._canceling = False
        self._observation_clock, self._localization_ready = observation_clock, localization_ready
        self._arrival_timeout = arrival_timeout_s
        self._arrival_after = self._arrival_deadline = 0.0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None

    def handle(self, text: str) -> None:
        request_id = uuid.uuid4().hex
        try:
            command = parse_movement(text)
        except ValidationError as e:
            self._publish(NavigationUpdate(request_id, "invalid", str(e)))
            return
        if command.kind == "cancel":
            self.cancel()
            return
        with self._lock:
            if self._closed or self._active:
                self._publish(
                    NavigationUpdate(
                        request_id, "busy", "Navigation is busy or shutting down. Stop the current trip first."
                    )
                )
                return
            if not self._localization_ready():
                self._choices = ()
                self._publish(
                    NavigationUpdate(request_id, "unavailable", "Localization is missing, stale or uncertain.")
                )
                return
            chosen = None
            if command.kind == "choose":
                if not 1 <= command.choice <= len(self._choices) or self._clock() - self._choices_at > self._timeout:
                    self._publish(
                        NavigationUpdate(
                            request_id, "invalid", "There is no matching destination option. Give a destination first."
                        )
                    )
                    return
                chosen = self._choices[command.choice - 1]
            self._choices = ()
            self._active, self._state, self._destination = request_id, "resolving", None
            self._requested_at = self._clock()
            self._canceling = False
            self._publish(NavigationUpdate(request_id, "resolving", "Looking up the destination."))
            if not self._submit(lambda: self._resolve(request_id, command, chosen)):
                self._active = None
                self._publish(
                    NavigationUpdate(request_id, "busy", "The command worker is busy. Please repeat the command.")
                )

    def _resolve(self, request_id: str, command: MovementCommand, chosen: Destination | None) -> None:
        with self._lock:
            if request_id != self._active:
                return
            if self._clock() - self._requested_at > self._timeout:
                self._active = None
                self._publish(
                    NavigationUpdate(request_id, "not_found", "The command expired while waiting. Please repeat it.")
                )
                return
        try:
            result = (
                Resolution("resolved", "Using your selected destination.", (chosen,))
                if chosen
                else self._resolver.resolve(command)
            )
            if result.state == "resolved" and not self._resolver.current(result.choices[0]):
                result = Resolution("not_found", "That memory changed or expired. Please give the destination again.")
        except Exception as e:
            result = Resolution("not_found", f"Destination lookup failed: {e}")
        with self._lock:
            if request_id != self._active:
                return
            if self._clock() - self._requested_at > self._timeout:
                result = Resolution("not_found", "Destination lookup timed out. Please repeat the command.")
            if not self._localization_ready():
                result = Resolution("not_found", "Localization became unavailable during destination lookup.")
            if result.state != "resolved":
                self._active = None
                self._choices = result.choices
                self._choices_at = self._clock()
                self._publish(NavigationUpdate(request_id, result.state, result.message, choices=result.choices))
                return
            self._destination, self._state = result.choices[0], "submitting"
            self._publish(NavigationUpdate(request_id, "submitting", result.message, self._destination))
            try:
                self._navigator.send(request_id, self._destination, lambda e: self._event(request_id, e))
            except Exception as e:
                self._event(request_id, NavigationEvent("uncertain", f"Navigation transport failed: {e}"))

    def _event(self, request_id: str, event: NavigationEvent) -> None:
        with self._lock:
            if request_id != self._active:
                return
            if self._state in {"awaiting_observation", "verifying_arrival"}:
                return  # Late transport feedback cannot finish or restart visual verification.
            if event.state == "succeeded" and self._destination is not None and self._destination.source == "memory":
                if self._canceling or not self._localization_ready():
                    self._finish_arrival(
                        request_id, False, "Reached the pose; the destination was not visually verified."
                    )
                    return
                self._state = "awaiting_observation"
                self._arrival_after = self._observation_clock()
                self._arrival_deadline = self._clock() + self._arrival_timeout
                self._publish(
                    NavigationUpdate(
                        request_id,
                        self._state,
                        "Reached the pose. Waiting for a fresh view of the destination.",
                        self._destination,
                    )
                )
                return
            if self._canceling and event.state == "navigating":
                event = NavigationEvent("canceling", "Waiting for cancellation to finish.", event.distance_remaining)
            self._state = event.state
            if event.state in {"succeeded", "canceled", "failed", "rejected", "unavailable"}:
                self._active = None
            self._publish(
                NavigationUpdate(
                    request_id,
                    event.state,
                    event.message,
                    self._destination,
                    distance_remaining=event.distance_remaining,
                )
            )

    @property
    def needs_observation(self) -> bool:
        with self._lock:
            return self._active is not None and self._state == "awaiting_observation"

    def observe(self, observation: Observation) -> None:
        """Offer a live capture after arrival, independent of captioning and ingestion latency."""
        with self._lock:
            destination, request_id = self._destination, self._active
            if request_id is None or self._state != "awaiting_observation" or destination is None:
                return
            memory = destination.memory
            if (
                memory is None
                or observation.robot_id != memory.robot_id
                or observation.camera_id != memory.camera_id
                or not observation.localization_checked
                or not self._localization_ready()
                or not self._arrival_after < observation.timestamp <= self._observation_clock()
                or not observation.pose.same_frame(destination.pose)
                or observation.pose.distance_to(destination.pose) > 0.35
                or observation.pose.heading_difference(destination.pose) > 0.35
            ):
                return
            self._state = "verifying_arrival"
        try:
            # Snapshot bytes before ingestion can replace and clean up this keyframe.
            image = data_url(observation.evidence.uri)
            if not self._submit(lambda: self._verify_arrival(request_id, destination, image)):
                raise ValidationError("the verification worker is busy")
        except Exception as e:
            self._finish_arrival(request_id, False, f"Could not inspect the arrival image: {e}")

    def _verify_arrival(self, request_id: str, destination: Destination, image: str) -> None:
        with self._lock:
            if request_id != self._active or self._clock() >= self._arrival_deadline:
                return
        try:
            verdict = self._resolver.verify(destination.target, image)
            matched, reason = verdict.result == "matched", verdict.reason
        except Exception as e:
            matched, reason = False, f"Visual verification failed: {e}"
        self._finish_arrival(request_id, matched, reason)

    def _finish_arrival(self, request_id: str, matched: bool, reason: str) -> None:
        with self._lock:
            if request_id != self._active:
                return
            matched = matched and self._localization_ready() and self._clock() < self._arrival_deadline
            self._active = None
            self._state = "succeeded" if matched else "destination_unverified"
            message = (
                "Destination visible at the reached viewpoint. "
                if matched
                else "Reached the pose; destination unverified. "
            ) + reason
            self._publish(NavigationUpdate(request_id, self._state, message, self._destination))

    def poll(self) -> None:
        """Bound arrival waits and request cancellation if localization is lost during a trip."""
        with self._lock:
            if self._active is None:
                return
            if self._state == "resolving" and self._clock() - self._requested_at >= self._timeout:
                request_id, self._active = self._active, None
                self._publish(
                    NavigationUpdate(
                        request_id, "not_found", "Destination lookup timed out. Please repeat the command."
                    )
                )
                return
            ready = self._localization_ready()
            if self._state in {"awaiting_observation", "verifying_arrival"}:
                if not ready or self._clock() >= self._arrival_deadline:
                    self._finish_arrival(
                        self._active,
                        False,
                        "Fresh visual evidence or localization was unavailable before the deadline.",
                    )
            elif not ready and not self._canceling:
                self.cancel()

    def cancel(self) -> None:
        with self._lock:
            self._choices = ()
            if self._active is None:
                self._publish(NavigationUpdate("", "idle", "No navigation request is active."))
                return
            request_id = self._active
            if self._state in {"resolving", "awaiting_observation", "verifying_arrival"}:
                self._active = None
                self._publish(NavigationUpdate(request_id, "canceled", "Destination lookup or verification canceled."))
                return
            self._state = "canceling"
            self._canceling = True
            self._publish(
                NavigationUpdate(request_id, "canceling", "Requesting cancellation from Nav2.", self._destination)
            )
            try:
                self._navigator.cancel(request_id)
            except Exception as e:
                self._event(request_id, NavigationEvent("uncertain", f"Cancellation could not be confirmed: {e}"))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.cancel()
