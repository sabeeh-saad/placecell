"""Resolve explicit movement commands to map-scoped destinations and manage one trip."""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

from placecell.approach import ApproachPlan, ApproachPlanner
from placecell.errors import FailureStage, TargetValidationError, ValidationError
from placecell.memory import Memory, Pose
from placecell.mission_context import MissionContext
from placecell.missions import MissionPlan, MissionPlanner
from placecell.object_arrival import ObjectArrivalVerdict, ObjectArrivalVerifier, ObjectReference
from placecell.object_search import ObjectSearch
from placecell.objects import ObjectRecall
from placecell.pipeline import Observation
from placecell.providers.captioning import data_url
from placecell.retrieval import RankedMemory, Recall
from placecell.store.base import Filter, VectorStore
from placecell.tracing import (
    TraceContext,
    TraceStore,
    bind_trace,
    current_trace,
    trace_event,
    trace_scope,
    trace_span,
    traced,
)
from placecell.verification import ObjectSceneVerifier, SceneVerdict, SceneVerifier, SemanticQueryResolver


@dataclass(frozen=True, slots=True)
class MovementCommand:
    kind: Literal["go", "cancel", "choose"]
    destination: str = ""
    coordinates: tuple[float, float, float] | None = None
    choice: int = 0


def parse_movement(text: str) -> MovementCommand:
    """Accept direct movement requests; questions, negation and compound commands do not move the robot."""
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
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
    approach: ApproachPlan | None = field(default=None, repr=False, compare=False)
    object_reference: ObjectReference | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Resolution:
    state: Literal["resolved", "ambiguous", "not_found"]
    message: str
    choices: tuple[Destination, ...] = ()
    failure_stage: FailureStage = ""


def _trace_destination(destination: Destination | None) -> dict[str, object] | None:
    if destination is None:
        return None
    return {
        "label": destination.label,
        "target": destination.target,
        "source": destination.source,
        "pose": asdict(destination.pose),
        "memory_id": destination.memory.id if destination.memory else None,
        "object_id": destination.object_id,
        "object_revision": destination.object_revision,
    }


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
        approach: ApproachPlanner | None = None,
        object_arrival: ObjectArrivalVerifier | None = None,
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
        self._approach = approach
        self._object_arrival = object_arrival

    @property
    def configured_places(self) -> tuple[str, ...]:
        """Names only, in the current map; the planner cannot choose poses."""
        return tuple(sorted(name for name, pose in self._places.items() if pose.same_frame(self._origin)))

    @traced("approach_planning")
    def prepare_destination(self, destination: Destination, canceled: Callable[[], bool]) -> Destination:
        if not destination.object_id:
            return destination
        if self._object_arrival is not None:
            destination = replace(destination, object_reference=self._object_arrival.capture(destination.object_id))
        if self._approach is None:
            return destination
        record = self._store.objects.get(destination.object_id)
        views = self._store.objects.views(destination.object_id, include_crops=False, limit=1)
        if record is None or not views or not self.current(destination):
            raise TargetValidationError("object changed before approach planning", "retrieval")
        try:
            plan = self._approach.plan(record, views[0], canceled)
        except ValidationError as e:
            raise TargetValidationError(str(e), "geometry") from e
        return replace(destination, pose=plan.pose, approach=plan) if plan else destination

    def arrival_available(self, destination: Destination, checked_generation: int | None = None) -> bool:
        if not destination.object_id:
            return self.current(destination)
        return bool(
            self._object_arrival is not None
            and destination.object_reference is not None
            and destination.object_reference.record.id == destination.object_id
            and self._object_arrival.available(destination.object_reference)
            and destination.memory is not None
            and self._recall.confidence(destination.memory) >= self._policy.min_confidence
            and (checked_generation is None or self._store.objects.evidence_generation == checked_generation)
        )

    def verify_object_arrival(
        self, destination: Destination, observation: Observation, image: str, canceled: Callable[[], bool]
    ) -> ObjectArrivalVerdict:
        if not self.arrival_available(destination):
            return ObjectArrivalVerdict(
                "unavailable", "The selected object's saved reference is unavailable.", failure_stage="retrieval"
            )
        assert self._object_arrival is not None and destination.object_reference is not None
        if self._object_arrival.supports_comparison:
            # One image inspection binds detection, instance comparison and the
            # original request to the same box. Embeddings/geometry remain independent.
            return self._object_arrival.verify_image(
                destination.object_reference, observation, image, canceled, target=destination.target
            )
        # Independent image checks may run together. The object verifier validates
        # current identities and captures its evidence version AFTER both complete.
        # One bounded worker is joined before this verification attempt returns.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="arrival-request") as pool:
            request = pool.submit(bind_trace(current_trace(), lambda: self.verify(destination.target, image)))
            return self._object_arrival.verify_image(
                destination.object_reference, observation, image, canceled, request.result
            )

    @traced("destination_lookup")
    def resolve(self, command: MovementCommand) -> Resolution:
        trace_event(
            "lookup.request",
            target=command.destination,
            map_id=self._origin.map_id,
            frame_id=self._origin.frame_id,
            robot_id=self._scope.robot_id,
        )
        if command.kind != "go":
            raise ValidationError("only a go command has a destination")
        command = replace(command, destination=" ".join(command.destination.casefold().split()).removeprefix("the "))
        if command.coordinates is not None:
            x, y, yaw = command.coordinates
            pose = Pose(x, y, yaw, self._origin.frame_id, self._origin.map_id)
            return Resolution(
                "resolved", "Using the requested coordinates.", (Destination(command.destination, pose, "coordinates"),)
            )
        if command.destination in self._places:
            pose = self._places[command.destination]
            if not pose.same_frame(self._origin):
                return Resolution("not_found", "That named place belongs to a different map.", failure_stage="geometry")
            return Resolution(
                "resolved", "Using the named place.", (Destination(command.destination, pose, "named_place"),)
            )
        if self._verifier is None:
            return Resolution(
                "not_found", "Visual destination verification is not configured.", failure_stage="execution"
            )
        if self._objects is not None:
            object_result = self._resolve_object(command.destination)
            if object_result is not None:
                return object_result
        with trace_span("memory_retrieval"):
            hits = self._recall.similar(command.destination, k=self._policy.candidates, where=self._scope)
        trace_event(
            "retrieval.candidates",
            channel="scene",
            candidates=[
                {
                    "memory_id": h.memory.id,
                    "similarity": h.similarity,
                    "image_similarity": h.image_similarity,
                    "caption_similarity": h.caption_similarity,
                    "confidence": h.confidence,
                    "view_timestamp": h.memory.view_timestamp,
                    "pose": asdict(h.memory.pose),
                    "localization_checked": h.memory.localization_checked,
                }
                for h in hits
            ],
        )
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
        trace_event("retrieval.eligible", channel="scene", memory_ids=[h.memory.id for h in hits], policy=asdict(p))
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
            return Resolution(
                "not_found",
                "Too many possible places. Please describe the destination more precisely.",
                failure_stage="identity",
            )
        choices_list = []
        for hit in candidates:
            memory = hit.memory
            assert memory.evidence is not None
            with trace_span("candidate_verification", memory_id=memory.id):
                verdict = self.verify(command.destination, data_url(memory.evidence.uri))
            trace_event("candidate.verdict", memory_id=memory.id, verdict=verdict.result, reason=verdict.reason)
            if verdict.result == "uncertain":
                return Resolution(
                    "not_found",
                    "The images do not clearly identify the destination. Please give more detail.",
                    failure_stage="identity",
                )
            if verdict.result == "matched":
                choices_list.append(Destination(memory.caption, memory.pose, "memory", memory, command.destination))
        choices = tuple(choices_list)
        if not choices:
            return Resolution(
                "not_found", "The retrieved images do not show the requested destination.", failure_stage="identity"
            )
        if len(choices) > 1:
            return Resolution(
                "ambiguous", "I found several places. Say 'option one', 'option two', or give more detail.", choices
            )
        return Resolution("resolved", "Navigating to the remembered observation viewpoint.", choices)

    def _resolve_object(self, target: str, *, query: str | None = None) -> Resolution | None:
        assert self._objects is not None
        p = self._policy
        with trace_span("object_retrieval"):
            hits = self._objects.similar(
                query or target,
                robot_id=self._scope.robot_id or "",
                camera_id=self._scope.camera_id or "",
                frame_id=self._origin.frame_id,
                map_id=self._origin.map_id,
                k=p.candidates + 1,
                max_age_s=p.max_age_s,
            )
        trace_event(
            "retrieval.candidates",
            channel="object",
            candidates=[
                {
                    "object_id": h.object.id,
                    "revision": h.object.revision,
                    "memory_id": h.view.memory.id,
                    "similarity": h.similarity,
                    "status": h.object.status,
                    "misses": h.object.misses,
                }
                for h in hits
            ],
        )
        labels = tuple(sorted({h.object.label for h in hits}))
        hits = [h for h in hits if h.similarity >= p.min_similarity]
        trace_event("retrieval.eligible", channel="object", object_ids=[h.object.id for h in hits], policy=asdict(p))
        if not hits:
            if query is None and 1 <= len(labels) <= 13 and isinstance(self._verifier, SemanticQueryResolver):
                with trace_span("semantic_grounding"):
                    expanded = self._verifier.search_query(target, labels)
                trace_event("retrieval.query_expanded", target=target, query=expanded, observed_labels=labels)
                if expanded and expanded in labels and expanded.casefold() != target.casefold():
                    # Retain the original target for both candidate and arrival verification.
                    # Expansion never changes similarity, presence, geometry or identity gates.
                    return self._resolve_object(target, query=expanded)
            return None
        if len(hits) > p.verification_candidates:
            return Resolution(
                "not_found",
                "Too many possible objects. Please describe the destination more precisely.",
                failure_stage="identity",
            )
        choices = []
        for hit in hits:
            memory = hit.view.memory
            with trace_span("candidate_verification", object_id=hit.object.id, memory_id=memory.id):
                if isinstance(self._verifier, ObjectSceneVerifier) and memory.evidence is not None:
                    verdict = self._verifier.verify_object(target, hit.view.image_url(), data_url(memory.evidence.uri))
                else:
                    verdict = self.verify(target, hit.view.image_url())
            trace_event(
                "candidate.verdict",
                object_id=hit.object.id,
                memory_id=memory.id,
                verdict=verdict.result,
                reason=verdict.reason,
            )
            if verdict.result == "not_matched":
                continue
            if verdict.result == "uncertain" or hit.object.status != "present" or hit.object.misses:
                return Resolution(
                    "not_found",
                    "That object's identity or current presence is uncertain. Please revisit it.",
                    failure_stage="identity",
                )
            if (
                not memory.localization_checked
                or self._recall.confidence(memory) < p.min_confidence
                or memory.view_timestamp is None
                or not 0 <= self._clock() - memory.view_timestamp <= p.max_age_s
            ):
                return Resolution(
                    "not_found", "That object has no reliable recent observation viewpoint.", failure_stage="retrieval"
                )
            choices.append(
                Destination(memory.caption, memory.pose, "memory", memory, target, hit.object.id, hit.object.revision)
            )
        if not choices:
            return Resolution(
                "not_found",
                "The current object views do not verify that target. Please clarify or revisit it.",
                failure_stage="identity",
            )
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
            plan = destination.approach
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
                and views[0].memory.pose == (plan.viewpoint if plan else destination.pose)
                and (plan is None or (plan.object == record and plan.pose == destination.pose))
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
    cancel_requested: bool = False


@dataclass(frozen=True, slots=True)
class NavigationUpdate:
    request_id: str
    state: str
    message: str
    destination: Destination | None = None
    choices: tuple[Destination, ...] = ()
    distance_remaining: float | None = None
    object_result: str = ""
    search_attempt: int = 0
    mission_id: str = ""
    mission_step: int = 0
    mission_destinations: tuple[str, ...] = ()
    instance_id: str = ""
    sequence: int = 0
    failure_stage: FailureStage = ""


@dataclass(frozen=True, slots=True)
class NavigationSnapshot:
    """Read-only controller state; retained outcomes never resume work after restart."""

    status: NavigationUpdate
    sequence: int
    busy: bool
    closed: bool
    active_request_id: str | None
    choice_remaining_s: float | None


class Navigator(Protocol):
    def send(self, request_id: str, destination: Destination, callback: Callable[[NavigationEvent], None]) -> None: ...

    def cancel(self, request_id: str) -> None: ...


class NavigationCommands:
    """One active trip or ordered mission. Cancellation never waits for a model call."""

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
        localization_generation: Callable[[], int] = lambda: 0,
        sensor_ready: Callable[[Destination], bool] = lambda _: True,
        sensor_generation: Callable[[Destination], object] = lambda _: 0,
        arrival_timeout_s: float = 30.0,
        max_observation_age_s: float = 5.0,
        arrival_max_attempts: int = 1,
        search: ObjectSearch | None = None,
        mission_planner: MissionPlanner | None = None,
        mission_context: MissionContext | None = None,
        trace_store: TraceStore | None = None,
        startup_block_reason: Callable[[], str] = lambda: "",
    ) -> None:
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValidationError("command timeout must be finite and positive")
        if not math.isfinite(arrival_timeout_s) or arrival_timeout_s <= 0:
            raise ValidationError("arrival timeout must be finite and positive")
        if not math.isfinite(max_observation_age_s) or max_observation_age_s <= 0:
            raise ValidationError("arrival observation maximum age must be finite and positive")
        if type(arrival_max_attempts) is not int or not 1 <= arrival_max_attempts <= 5:
            raise ValidationError("arrival attempts must be an integer within 1..5")
        self._resolver, self._navigator = resolver, navigator
        self._submit_callback, self._publish_callback = submit, publish
        self._trace_store = trace_store
        self._trace_context: TraceContext | None = None
        self._timeout, self._clock = request_timeout_s, clock
        self._requested_at = self._choices_at = 0.0
        self._lock = threading.RLock()
        self._active: str | None = None
        self._state = "idle"
        self._destination: Destination | None = None
        self._choices: tuple[Destination, ...] = ()
        self._closed = False
        self._canceling = False
        self._observation_clock, self._localization_check = observation_clock, localization_ready
        self._localization_generation = localization_generation
        self._sensor_ready, self._sensor_generation = sensor_ready, sensor_generation
        self._accepted_localization: int | None = None
        self._accepted_sensors: object = None
        self._interruption_reason = ""
        self._arrival_timeout = arrival_timeout_s
        self._max_observation_age = max_observation_age_s
        self._arrival_max_attempts, self._arrival_attempts = arrival_max_attempts, 0
        self._arrival_stamp: float | None = None
        self._image_deadline = 0.0
        self._arrival_after = self._arrival_deadline = 0.0
        self._search = search
        self._search_deadline: float | None = None
        self._search_anchor: Pose | None = None
        self._search_visited: tuple[Pose, ...] = ()
        self._search_count = self._leg = 0
        self._transport_id = ""
        self._mission_planner = mission_planner
        self._mission: MissionPlan | None = None
        self._mission_id = ""
        self._mission_step = 0
        self._context = mission_context or (MissionContext() if mission_planner is not None else None)
        self._context_ok = True
        self._last_context_state: tuple[str, str] | None = None
        self._instance_id = uuid.uuid4().hex
        self._sequence = 0
        self._admission_epoch = 0
        self._snapshot_status = NavigationUpdate(
            "", "idle", "No navigation request is active.", instance_id=self._instance_id
        )
        self._startup_block_reason = startup_block_reason
        self._startup_reason = startup_block_reason()
        if self._startup_reason:
            self._snapshot_status = replace(
                self._snapshot_status, state="uncertain", message=self._startup_reason, failure_stage="execution"
            )

    def _refresh_startup(self) -> str:
        reason = self._startup_block_reason()
        if reason != self._startup_reason and self._active is None and not self._closed:
            self._startup_reason = reason
            self._publish(
                NavigationUpdate(
                    "",
                    "uncertain" if reason else "idle",
                    reason or "Previous Nav2 ownership reconciled. Submit a new command to move.",
                    failure_stage="execution" if reason else "",
                ),
                state_update=True,
            )
        return reason

    @property
    def admission_epoch(self) -> int:
        """An intervening stop invalidates commands still waiting for durable admission."""
        with self._lock:
            return self._admission_epoch

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None or self._mission is not None or bool(self._startup_block_reason())

    def snapshot(self) -> NavigationSnapshot:
        """Copy current state atomically without polling, dispatching or replaying commands."""
        with self._lock:
            remaining = max(0.0, self._timeout - (self._clock() - self._choices_at)) if self._choices else None
            return NavigationSnapshot(
                self._snapshot_status,
                self._sequence,
                self.busy,
                self._closed,
                self._active,
                remaining,
            )

    def reject_command(self, message: str) -> None:
        """Report a malformed transport envelope without changing the current mission."""
        self._publish(NavigationUpdate(uuid.uuid4().hex, "invalid", message))

    def _submit(self, task: Callable[[], None]) -> bool:
        return self._submit_callback(bind_trace(self._trace_context, task))

    def _publish(
        self, update: NavigationUpdate, context: TraceContext | None = None, *, state_update: bool = False
    ) -> None:
        with self._lock:
            self._sequence += 1
            update = replace(update, instance_id=self._instance_id, sequence=self._sequence)
            if state_update:
                self._snapshot_status = update
            self._publish_ordered(update, context)

    def _publish_ordered(self, update: NavigationUpdate, context: TraceContext | None) -> None:
        context = context or current_trace()
        if context is None and self._trace_context and update.request_id == self._trace_context.request_id:
            context = self._trace_context
        if context:
            context.emit(
                "status",
                state=update.state,
                message=update.message,
                destination=_trace_destination(update.destination),
                choices=[_trace_destination(choice) for choice in update.choices],
                distance_remaining=update.distance_remaining,
                object_result=update.object_result,
                search_attempt=update.search_attempt,
                mission_destinations=update.mission_destinations,
                failure_stage=update.failure_stage,
            )
        self._publish_callback(update)

    def _emit(self, update: NavigationUpdate, *, state_update: bool = True) -> None:
        if not update.failure_stage and update.state in {
            "failed",
            "rejected",
            "unavailable",
            "uncertain",
            "cancel_failed",
            "canceled",
            "canceling",
        }:
            update = replace(update, failure_stage="execution")
        if self._mission_id:
            update = replace(
                update,
                mission_id=self._mission_id,
                mission_step=self._mission_step + 1 if self._mission else 0,
                mission_destinations=self._mission.destinations if self._mission else (),
            )
        key = (update.request_id, update.state)
        if self._context is not None and key != self._last_context_state:
            destination = update.destination
            try:
                self._context.record(
                    update.request_id,
                    "status",
                    {
                        "state": update.state,
                        "message": update.message,
                        "mission_id": update.mission_id,
                        "step": update.mission_step,
                        "destinations": update.mission_destinations,
                        "target": destination.target if destination else "",
                        "memory_id": destination.memory.id if destination and destination.memory else "",
                        "object_id": destination.object_id if destination else "",
                        "failure_stage": update.failure_stage,
                    },
                )
                self._last_context_state = key
            except Exception as e:
                self._context_ok = False
                update = replace(update, message=f"{update.message} Context persistence failed: {e}")
        context = self._trace_context
        active_trace = current_trace()
        if active_trace and update.request_id == active_trace.request_id:
            context = active_trace
        self._publish(update, context, state_update=state_update)

    def _record_instruction(self, request_id: str, text: str, *, state_update: bool = False) -> bool:
        if self._context is None:
            return True
        try:
            self._context.record(request_id, "instruction", {"text": text})
            self._context_ok = True
            return True
        except Exception as e:
            self._context_ok = False
            self._publish(
                NavigationUpdate(request_id, "unavailable", f"Could not save the instruction context: {e}"),
                state_update=state_update,
            )
            return False

    def _clear_mission(self) -> None:
        self._mission, self._mission_id, self._mission_step = None, "", 0
        self._accepted_localization = None

    def _provenance_ready(self, destination: Destination | None = None) -> bool:
        ready = self._localization_check()
        generation = self._localization_generation()
        if not ready or (self._accepted_localization is not None and generation != self._accepted_localization):
            return False
        destination = destination or (self._destination if self._active else None)
        if destination is None:
            return True
        ready = self._sensor_ready(destination)
        token = self._sensor_generation(destination)
        return ready and (self._accepted_sensors is None or token == self._accepted_sensors)

    def _reset_leg(self, request_id: str) -> None:
        self._active, self._state, self._destination = request_id, "resolving", None
        self._requested_at = self._clock()
        self._canceling = False
        self._interruption_reason = ""
        self._accepted_sensors = None
        self._arrival_stamp = None
        self._search_deadline = None
        self._search_anchor = None
        self._search_visited = ()
        self._search_count = self._leg = 0
        self._transport_id = request_id
        if self._trace_context:
            self._trace_context = replace(
                self._trace_context,
                request_id=request_id,
                step=self._mission_step + 1 if self._mission or not self._mission_planner else 0,
            )

    def _complete(self, update: NavigationUpdate) -> None:
        """Called under the controller lock, only after a terminal trip outcome."""
        self._active = None
        if (
            update.state == "succeeded"
            and self._mission is not None
            and not self._canceling
            and self._mission_step + 1 < len(self._mission.destinations)
        ):
            self._emit(replace(update, state="step_succeeded"))
            if not self._context_ok:
                self._emit(NavigationUpdate(update.request_id, "unavailable", "Mission stopped: context unavailable."))
                self._clear_mission()
                return
            self._mission_step += 1
            request_id = uuid.uuid4().hex
            self._reset_leg(request_id)
            command = MovementCommand("go", self._mission.destinations[self._mission_step])
            self._emit(NavigationUpdate(request_id, "resolving", "Looking up the next mission destination."))
            if not self._submit(lambda: self._resolve(request_id, command)):
                self._active = None
                self._emit(NavigationUpdate(request_id, "failed", "Mission stopped: the command worker is busy."))
                self._clear_mission()
            return
        if self._canceling and update.state == "succeeded":
            update = replace(update, state="canceled", message="Navigation ended after cancellation; no further steps.")
        if self._interruption_reason:
            update = replace(update, message=f"{self._interruption_reason} {update.message}")
        self._state = update.state
        self._emit(update)
        self._clear_mission()

    def handle(
        self,
        text: str,
        *,
        request_id: str | None = None,
        target_request_id: str = "",
        admission_epoch: int | None = None,
    ) -> None:
        """Route once; callers supplying IDs must perform their own admission/deduplication."""
        request_id = request_id or uuid.uuid4().hex
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            self._publish(NavigationUpdate(request_id, "invalid", "Instructions must contain 1..2000 characters."))
            return
        with self._lock:
            try:
                continuation = parse_movement(text).kind in {"cancel", "choose"}
            except ValidationError:
                continuation = False
            context = None
            if self._trace_store:
                if continuation and self._trace_context and (self.busy or self._choices):
                    context = replace(self._trace_context, request_id=request_id)
                else:
                    context = self._trace_store.context(request_id, request_id)
        with trace_scope(context):
            trace_event("instruction", text=text)
            if target_request_id or admission_epoch is not None:
                with self._lock:
                    if (target_request_id and self._snapshot_status.request_id != target_request_id) or (
                        admission_epoch is not None and self._admission_epoch != admission_epoch
                    ):
                        self._publish(
                            NavigationUpdate(request_id, "stale_command", "The target changed or a stop intervened.")
                        )
                        return
                    self._handle(text, request_id)
            else:
                self._handle(text, request_id)

    def _handle(self, text: str, request_id: str) -> None:
        command = None
        try:
            command = parse_movement(text)
        except ValidationError as e:
            if self._mission_planner is None:
                self._publish(NavigationUpdate(request_id, "invalid", str(e)))
                return
        if command is not None and command.kind == "cancel":
            self.cancel()
            self._record_instruction(request_id, text)
            return
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            self._publish(NavigationUpdate(request_id, "invalid", "Instructions must contain 1..2000 characters."))
            return
        with self._lock:
            reason = self._refresh_startup()
            if reason:
                self._publish(NavigationUpdate(request_id, "uncertain", reason, failure_stage="execution"))
                return
            if self._closed or self._active:
                self._publish(
                    NavigationUpdate(
                        request_id, "busy", "Navigation is busy or shutting down. Stop the current trip first."
                    )
                )
                return
            if self._mission is not None and (command is None or command.kind != "choose"):
                self._emit(
                    NavigationUpdate(request_id, "busy", "Select a destination option or stop the mission first."),
                    state_update=False,
                )
                return
            if not self._provenance_ready():
                self._choices = ()
                self._emit(
                    NavigationUpdate(
                        request_id,
                        "unavailable",
                        "Localization or clock provenance is missing, stale or uncertain.",
                        failure_stage="geometry",
                    )
                )
                self._clear_mission()
                return
            chosen = None
            if command is not None and command.kind == "choose":
                if not 1 <= command.choice <= len(self._choices) or self._clock() - self._choices_at >= self._timeout:
                    self._publish(
                        NavigationUpdate(
                            request_id, "invalid", "There is no matching destination option. Give a destination first."
                        )
                    )
                    return
                chosen = self._choices[command.choice - 1]
            self._trace_context = current_trace()
            if not self._record_instruction(request_id, text, state_update=True):
                self._clear_mission()
                self._choices = ()
                return
            trace_event("instruction.accepted", command_kind=command.kind if command else "mission")
            self._choices = ()
            self._accepted_localization = self._localization_generation()
            self._reset_leg(request_id)
            if self._mission_planner is not None and chosen is None:
                self._state, self._mission_id = "planning", request_id
                self._emit(NavigationUpdate(request_id, "planning", "Planning and reviewing the requested mission."))
                submitted = self._submit(lambda: self._plan(request_id, text))
            else:
                assert command is not None
                self._emit(NavigationUpdate(request_id, "resolving", "Looking up the destination."))
                submitted = self._submit(lambda: self._resolve(request_id, command, chosen))
            if not submitted:
                self._active = None
                self._emit(
                    NavigationUpdate(request_id, "busy", "The command worker is busy. Please repeat the command.")
                )
                self._clear_mission()

    def _plan(self, request_id: str, text: str) -> None:
        assert self._mission_planner is not None

        def canceled() -> bool:
            with self._lock:
                return (
                    request_id != self._active
                    or self._clock() - self._requested_at >= self._timeout
                    or not self._provenance_ready()
                )

        try:
            context = self._context.recent(exclude_request_id=request_id) if self._context else []
            plan = self._mission_planner.plan(
                text, canceled, context=context, configured_places=self._resolver.configured_places
            )
        except Exception as e:
            trace_event("plan.failed", error_type=type(e).__name__)
            with self._lock:
                if request_id == self._active:
                    self._complete(NavigationUpdate(request_id, "rejected", f"Mission planning failed: {str(e)[:500]}"))
            return
        with self._lock:
            if request_id != self._active:
                trace_event("plan.discarded", reason="request no longer active")
                return
            if canceled():
                self._complete(
                    NavigationUpdate(request_id, "rejected", "Mission planning expired or became unavailable.")
                )
                return
            if plan.decision != "ready":
                self._complete(
                    NavigationUpdate(
                        request_id, "clarification_required" if plan.decision == "clarify" else "rejected", plan.message
                    )
                )
                return
            self._mission = plan
            self._reset_leg(request_id)
            self._emit(NavigationUpdate(request_id, "planned", plan.message))
            self._emit(NavigationUpdate(request_id, "resolving", "Looking up the first mission destination."))
        with trace_scope(self._trace_context):
            self._resolve(request_id, MovementCommand("go", plan.destinations[0]))

    @traced("destination_resolution")
    def _resolve(self, request_id: str, command: MovementCommand, chosen: Destination | None = None) -> None:
        with self._lock:
            if request_id != self._active:
                return
            if self._clock() - self._requested_at > self._timeout:
                self._complete(
                    NavigationUpdate(
                        request_id,
                        "not_found",
                        "The command expired while waiting. Please repeat it.",
                        failure_stage="execution",
                    )
                )
                return
        try:
            result = (
                Resolution("resolved", "Using your selected destination.", (chosen,))
                if chosen
                else self._resolver.resolve(command)
            )
            if result.state == "resolved":
                if not self._resolver.current(result.choices[0]):
                    result = Resolution(
                        "not_found", "That memory changed or expired. Please give the destination again."
                    )
                else:

                    def canceled() -> bool:
                        with self._lock:
                            return (
                                request_id != self._active
                                or self._clock() - self._requested_at > self._timeout
                                or not self._provenance_ready()
                            )

                    destination = self._resolver.prepare_destination(result.choices[0], canceled)
                    if not self._resolver.current(destination):
                        result = Resolution("not_found", "That object changed during approach planning.")
                    else:
                        message = (
                            "Navigating to a checked stopping pose near the object."
                            if destination.approach
                            else result.message
                        )
                        result = Resolution("resolved", message, (destination,))
        except Exception as e:
            trace_event("lookup.failed", error_type=type(e).__name__)
            result = Resolution(
                "not_found",
                f"Destination lookup failed: {str(e)[:500]}",
                failure_stage=e.failure_stage if isinstance(e, TargetValidationError) else "execution",
            )
        with self._lock:
            if request_id != self._active:
                trace_event("lookup.discarded", reason="request no longer active")
                return
            if self._clock() - self._requested_at > self._timeout:
                result = Resolution(
                    "not_found", "Destination lookup timed out. Please repeat the command.", failure_stage="execution"
                )
            if not self._provenance_ready():
                result = Resolution(
                    "not_found",
                    "Localization or sensor provenance became unavailable during destination lookup.",
                    failure_stage="geometry",
                )
            if result.state == "resolved" and not self._provenance_ready(result.choices[0]):
                self._complete(
                    NavigationUpdate(
                        request_id,
                        "unavailable",
                        "Required camera, aligned depth or localization is unavailable.",
                        failure_stage="geometry",
                    )
                )
                return
            if result.state != "resolved":
                self._active = None
                self._choices = result.choices
                self._choices_at = self._clock()
                self._state = result.state
                self._emit(
                    NavigationUpdate(
                        request_id,
                        result.state,
                        result.message,
                        choices=result.choices,
                        failure_stage=result.failure_stage
                        or ("identity" if result.state == "ambiguous" else "retrieval"),
                    )
                )
                if result.state != "ambiguous":
                    self._clear_mission()
                return
            self._destination, self._state = result.choices[0], "submitting"
            self._accepted_sensors = self._sensor_generation(self._destination)
            trace_event("destination.selected", destination=_trace_destination(self._destination))
            self._search_anchor = self._destination.pose
            self._search_visited = (self._destination.pose,)
            self._emit(NavigationUpdate(request_id, "submitting", result.message, self._destination))
            if not self._context_ok:
                self._complete(NavigationUpdate(request_id, "unavailable", "Navigation stopped: context unavailable."))
                return
            if not self._provenance_ready():
                self._complete(
                    NavigationUpdate(
                        request_id,
                        "unavailable",
                        "Navigation stopped: sensor provenance lost.",
                        failure_stage="geometry",
                    )
                )
                return
            if not self._resolver.current(self._destination):
                self._complete(
                    NavigationUpdate(
                        request_id,
                        "not_found",
                        "The selected target changed before dispatch. Select it again.",
                        failure_stage="retrieval",
                    )
                )
                return
            try:
                self._navigator.send(
                    request_id,
                    self._destination,
                    bind_trace(self._trace_context, lambda e: self._event(request_id, e, 0)),
                )
            except Exception as e:
                self._event(request_id, NavigationEvent("uncertain", f"Navigation transport failed: {e}"))

    def _event(self, request_id: str, event: NavigationEvent, leg: int | None = None) -> None:
        with self._lock:
            if request_id != self._active or (leg is not None and leg != self._leg):
                trace_event("callback.ignored", state=event.state, reason="request or search leg no longer active")
                return
            if self._state in {"awaiting_observation", "verifying_arrival"}:
                trace_event(
                    "callback.ignored", state=event.state, reason="arrival verification already owns completion"
                )
                return  # Late transport feedback cannot finish or restart visual verification.
            if event.cancel_requested or event.state in {"canceling", "cancel_failed", "uncertain"}:
                # Transport deadlines/errors can initiate cancellation independently.
                # A late success must not resume the mission after that decision.
                self._canceling = True
            if event.state == "succeeded" and not self._provenance_ready():
                self._canceling = True
                self._interruption_reason = (
                    "Sensor or localization provenance was lost; repeat the mission after recovery."
                )
            if event.state == "succeeded" and self._destination is not None and self._destination.source == "memory":
                if self._canceling or not self._provenance_ready():
                    self._finish_arrival(
                        request_id, False, "Reached the pose; the destination was not visually verified."
                    )
                    return
                self._state = "awaiting_observation"
                self._arrival_attempts = 0
                self._arrival_after = self._observation_clock()
                self._arrival_deadline = self._clock() + self._arrival_timeout
                if self._search_deadline is not None:
                    self._arrival_deadline = min(self._arrival_deadline, self._search_deadline)
                self._emit(
                    NavigationUpdate(
                        request_id,
                        self._state,
                        "Reached the pose. Waiting for a fresh view of the destination.",
                        self._destination,
                        search_attempt=self._search_count,
                    )
                )
                return
            if self._canceling and event.state == "navigating":
                event = NavigationEvent("canceling", "Waiting for cancellation to finish.", event.distance_remaining)
            self._state = event.state
            update = NavigationUpdate(
                request_id,
                event.state,
                event.message,
                self._destination,
                distance_remaining=event.distance_remaining,
                search_attempt=self._search_count,
            )
            if event.state in {"succeeded", "canceled", "failed", "rejected", "unavailable"}:
                self._complete(update)
            else:
                self._emit(update)

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
            arrival_trace = self._trace_context
            memory = destination.memory
            if (
                memory is None
                or observation.robot_id != memory.robot_id
                or observation.camera_id != memory.camera_id
                or not observation.localization_checked
                or (bool(destination.object_id) and observation.depth is None)
                or not self._provenance_ready()
                or not self._arrival_after < observation.timestamp <= self._observation_clock()
                or not 0 <= self._observation_clock() - observation.timestamp <= self._max_observation_age
                or not observation.pose.same_frame(destination.pose)
                or observation.pose.distance_to(destination.pose) > 0.35
                or observation.pose.heading_difference(destination.pose) > 0.35
            ):
                if arrival_trace:
                    arrival_trace.emit(
                        "arrival.observation_rejected",
                        robot_id=observation.robot_id,
                        camera_id=observation.camera_id,
                        timestamp=observation.timestamp,
                        pose=asdict(observation.pose),
                        localization_checked=observation.localization_checked,
                        aligned_depth_available=observation.depth is not None,
                        localization_ready=self._provenance_ready(),
                        after_timestamp=self._arrival_after,
                        observation_clock=self._observation_clock(),
                        destination=_trace_destination(destination),
                    )
                return
            if not self._resolver.arrival_available(destination):
                self._finish_arrival(
                    request_id,
                    False,
                    "The selected target became unavailable.",
                    "unavailable" if destination.object_id else "",
                    failure_stage="retrieval",
                )
                return
            self._arrival_stamp = observation.timestamp
            self._arrival_attempts += 1
            self._image_deadline = (
                self._clock() + self._max_observation_age - (self._observation_clock() - observation.timestamp)
            )
            self._state = "verifying_arrival"
            self._emit(
                NavigationUpdate(
                    request_id,
                    self._state,
                    "Checking a fresh view of the destination.",
                    destination,
                    search_attempt=self._search_count,
                )
            )
            if arrival_trace:
                arrival_trace.emit(
                    "arrival.observation_accepted",
                    attempt=self._arrival_attempts,
                    timestamp=observation.timestamp,
                    pose=asdict(observation.pose),
                    evidence_digest=observation.evidence.digest,
                )
        try:
            # Snapshot bytes before ingestion can replace and clean up this keyframe.
            image = data_url(observation.evidence.uri)
            if not self._submit(
                bind_trace(arrival_trace, lambda: self._verify_arrival(request_id, destination, image, observation))
            ):
                raise ValidationError("the verification worker is busy")
        except Exception as e:
            self._finish_arrival(
                request_id, False, f"Could not inspect the arrival image: {e}", failure_stage="execution"
            )

    @traced("arrival_verification")
    def _verify_arrival(self, request_id: str, destination: Destination, image: str, observation: Observation) -> None:
        with self._lock:
            if request_id != self._active:
                return
            if not self._arrival_fresh() or self._clock() >= self._arrival_deadline:
                self._finish_arrival(
                    request_id,
                    False,
                    "The arrival image expired before verification.",
                    "unavailable" if destination.object_id else "",
                    failure_stage="geometry",
                )
                return
        try:
            if destination.object_id:

                def canceled() -> bool:
                    with self._lock:
                        return (
                            request_id != self._active
                            or self._state != "verifying_arrival"
                            or self._clock() >= self._arrival_deadline
                            or not self._provenance_ready()
                            or not self._arrival_fresh()
                        )

                object_verdict = self._resolver.verify_object_arrival(destination, observation, image, canceled)
                if object_verdict.result in {"missing", "unobserved"} and self._search is not None and not canceled():
                    self._search_next(request_id, destination, object_verdict)
                    return
                self._finish_arrival(
                    request_id,
                    object_verdict.result == "matched",
                    object_verdict.reason,
                    object_verdict.result,
                    failure_stage=object_verdict.failure_stage,
                    checked_generation=object_verdict.checked_generation,
                )
                return
            verdict = self._resolver.verify(destination.target, image)
            matched, reason = verdict.result == "matched", verdict.reason
            failure_stage: FailureStage = "" if matched else "identity"
        except Exception as e:
            matched, reason = False, f"Visual verification failed: {e}"
            failure_stage = e.failure_stage if isinstance(e, TargetValidationError) else "execution"
        self._finish_arrival(
            request_id, matched, reason, "unavailable" if destination.object_id else "", failure_stage=failure_stage
        )

    def _arrival_fresh(self) -> bool:
        return (
            self._arrival_stamp is not None
            and 0 <= self._observation_clock() - self._arrival_stamp <= self._max_observation_age
            and self._clock() <= self._image_deadline
        )

    def _can_retry_arrival(self) -> bool:
        return (
            self._destination is not None
            and bool(self._destination.object_id)
            and self._arrival_attempts < self._arrival_max_attempts
            and self._clock() < self._arrival_deadline
            and self._provenance_ready()
        )

    def _search_next(self, request_id: str, destination: Destination, verdict: ObjectArrivalVerdict) -> None:
        assert self._search is not None
        with self._lock:
            if request_id != self._active:
                return
            if (
                self._state != "verifying_arrival"
                or self._clock() >= self._arrival_deadline
                or not self._provenance_ready()
            ):
                self._finish_arrival(
                    request_id, False, "Arrival verification expired or became unavailable.", "unavailable"
                )
                return
            if self._search_deadline is None:
                self._search_deadline = self._clock() + self._search.policy.timeout_s
            if self._search_count >= self._search.policy.max_viewpoints:
                self._finish_arrival(
                    request_id,
                    False,
                    "Local search reached its viewpoint limit.",
                    verdict.result,
                    failure_stage=verdict.failure_stage,
                )
                return
            self._state = "planning_search"
            self._leg += 1
            leg = self._leg
            self._emit(
                NavigationUpdate(
                    request_id,
                    self._state,
                    "Checking another nearby viewpoint.",
                    destination,
                    object_result=verdict.result,
                    search_attempt=self._search_count + 1,
                )
            )

        def canceled() -> bool:
            with self._lock:
                return (
                    request_id != self._active
                    or self._state != "planning_search"
                    or self._search_deadline is None
                    or self._clock() >= self._search_deadline
                    or not self._provenance_ready()
                )

        try:
            if not self._resolver.arrival_available(destination):
                raise TargetValidationError("the selected object reference is unavailable", "retrieval")
            assert destination.object_reference is not None and self._search_anchor is not None
            plan = self._search.next_view(
                destination.object_reference, self._search_anchor, self._search_visited, canceled
            )
            if not self._resolver.arrival_available(destination):
                raise TargetValidationError("the selected object changed", "retrieval")
            if not self._search.valid(plan, canceled):
                raise TargetValidationError("the search path changed", "geometry")
            goal = replace(destination, pose=plan.pose, approach=plan)
        except Exception as e:
            self._finish_arrival(
                request_id,
                False,
                f"Local search stopped: {e}",
                verdict.result,
                failure_stage=e.failure_stage if isinstance(e, TargetValidationError) else "geometry",
            )
            return
        with self._lock:
            if canceled():
                self._finish_arrival(request_id, False, "Local search canceled or expired.", verdict.result)
                return
            self._search_count += 1
            self._search_visited += (goal.pose,)
            self._destination, self._state = goal, "submitting"
            self._transport_id = f"{request_id}/search/{self._search_count}"
            self._emit(
                NavigationUpdate(
                    request_id,
                    "searching",
                    "Navigating to a checked local search viewpoint.",
                    goal,
                    object_result=verdict.result,
                    search_attempt=self._search_count,
                )
            )
            if not self._context_ok:
                self._finish_arrival(
                    request_id, False, "Local search stopped: context unavailable.", failure_stage="execution"
                )
                return
            if not self._provenance_ready():
                self._finish_arrival(
                    request_id, False, "Local search stopped: sensor provenance lost.", failure_stage="geometry"
                )
                return
            if not self._resolver.arrival_available(destination):
                self._finish_arrival(
                    request_id, False, "Local search stopped: target reference changed.", failure_stage="retrieval"
                )
                return
            try:
                trace_event(
                    "search.destination_selected", destination=_trace_destination(goal), attempt=self._search_count
                )
                self._navigator.send(
                    self._transport_id, goal, bind_trace(self._trace_context, lambda e: self._event(request_id, e, leg))
                )
            except Exception as e:
                self._event(request_id, NavigationEvent("uncertain", f"Search transport failed: {e}"), leg)

    def _finish_arrival(
        self,
        request_id: str,
        matched: bool,
        reason: str,
        object_result: str = "",
        *,
        failure_stage: FailureStage = "",
        checked_generation: int | None = None,
    ) -> None:
        with self._lock:
            if request_id != self._active:
                return
            if self._state == "verifying_arrival" and not self._arrival_fresh():
                if self._can_retry_arrival() and object_result not in {"ambiguous", "missing", "unobserved"}:
                    # The worker has returned. Discard its expired result before
                    # accepting a different capture; never overlap attempts or
                    # extend the original arrival deadline.
                    self._state, self._arrival_stamp = "awaiting_observation", None
                    self._arrival_after = self._observation_clock()
                    self._emit(
                        NavigationUpdate(
                            request_id,
                            self._state,
                            "The visual check expired. Waiting for a new capture within the arrival deadline.",
                            self._destination,
                            search_attempt=self._search_count,
                        )
                    )
                    return
                matched, reason, failure_stage = False, "The arrival image expired during verification.", "geometry"
                if object_result:
                    object_result = "unavailable"
            if matched and (not self._provenance_ready() or self._clock() >= self._arrival_deadline):
                matched = False
                reason = "Arrival verification expired or localization became unavailable."
                failure_stage = "geometry"
                if object_result:
                    object_result = "unavailable"
            if matched and self._destination is not None:
                matched = self._resolver.arrival_available(self._destination, checked_generation)
                if not matched:
                    object_result, reason = (
                        "unavailable" if self._destination.object_id else "",
                        "The selected target reference became unavailable.",
                    )
                    failure_stage = "retrieval"
            failure_stage = "" if matched else (failure_stage or "identity")
            self._state = "succeeded" if matched else "destination_unverified"
            if self._trace_context:
                self._trace_context.emit(
                    "arrival.verdict",
                    matched=matched,
                    reason=reason,
                    object_result=object_result,
                    failure_stage=failure_stage,
                )
            if not matched and object_result == "ambiguous":
                self._state = "destination_ambiguous"
            message = (
                "Destination visible at the reached viewpoint. "
                if matched
                else "Reached the pose; destination unverified. "
            ) + reason
            self._complete(
                NavigationUpdate(
                    request_id,
                    self._state,
                    message,
                    self._destination,
                    object_result=object_result,
                    search_attempt=self._search_count,
                    failure_stage=failure_stage,
                )
            )

    def poll(self) -> None:
        """Bound arrival waits and request cancellation if localization is lost during a trip."""
        with self._lock:
            if self._refresh_startup():
                return
            if self._active is None:
                if self._choices and not self._provenance_ready():
                    self._choices = ()
                    self._complete(
                        NavigationUpdate(
                            self._transport_id,
                            "unavailable",
                            "Destination provenance was lost.",
                            failure_stage="geometry",
                        )
                    )
                    return
                if self._choices and self._clock() - self._choices_at >= self._timeout:
                    self._choices = ()
                    self._complete(
                        NavigationUpdate(
                            self._transport_id, "not_found", "Destination choice expired.", failure_stage="retrieval"
                        )
                    )
                return
            if self._state in {"planning", "resolving"} and self._clock() - self._requested_at >= self._timeout:
                if self._trace_context:
                    self._trace_context.emit("deadline.expired", phase=self._state, limit_s=self._timeout)
                self._complete(
                    NavigationUpdate(
                        self._active,
                        "not_found",
                        "Planning or destination lookup timed out. Please repeat the command.",
                        failure_stage="execution",
                    )
                )
                return
            ready = self._provenance_ready()
            if not self._context_ok and not self._canceling:
                self.cancel("Context persistence became unavailable.")
                return
            if self._search_deadline is not None and self._clock() >= self._search_deadline:
                if self._state in {"planning_search", "awaiting_observation", "verifying_arrival"}:
                    self._finish_arrival(
                        self._active, False, "Local search time limit reached.", failure_stage="execution"
                    )
                elif not self._canceling:
                    self.cancel()
                return
            if self._state == "planning_search" and not ready:
                self._finish_arrival(
                    self._active,
                    False,
                    "Localization became unavailable during local search.",
                    failure_stage="geometry",
                )
                return
            if self._state in {"awaiting_observation", "verifying_arrival"}:
                if (
                    not ready
                    or self._clock() >= self._arrival_deadline
                    or (
                        self._state == "verifying_arrival"
                        and not self._arrival_fresh()
                        and not self._can_retry_arrival()
                    )
                ):
                    self._finish_arrival(
                        self._active,
                        False,
                        "Fresh visual evidence or sensor/localization provenance was unavailable before the deadline.",
                        failure_stage="geometry",
                    )
            elif not ready and not self._canceling:
                if self._trace_context:
                    self._trace_context.emit("localization.unavailable", phase=self._state)
                self.cancel("Sensor or localization provenance was lost; repeat the mission after recovery.")

    def cancel(self, reason: str = "") -> None:
        with self._lock:
            if reason:
                self._interruption_reason = reason
            self._admission_epoch += 1
            if self._refresh_startup():
                self._publish(self._snapshot_status)
                return
            if self._trace_context and (self._active or self._mission):
                self._trace_context.emit("cancellation.requested", phase=self._state, transport_id=self._transport_id)
            had_choices = bool(self._choices)
            self._choices = ()
            if self._active is None:
                if self._mission is not None or had_choices:
                    self._complete(NavigationUpdate(self._transport_id, "canceled", "Mission canceled."))
                else:
                    self._publish(NavigationUpdate("", "idle", "No navigation request is active."))
                return
            request_id = self._active
            if self._state in {"planning", "resolving", "planning_search", "awaiting_observation", "verifying_arrival"}:
                self._complete(NavigationUpdate(request_id, "canceled", "Destination lookup or verification canceled."))
                return
            self._state = "canceling"
            self._canceling = True
            try:
                self._navigator.cancel(self._transport_id)
            except Exception as e:
                self._event(request_id, NavigationEvent("uncertain", f"Cancellation could not be confirmed: {e}"))
            # Issue the transport request before context persistence/status publication.
            # An injected or already-complete future can finish this trip synchronously.
            if self._active == request_id and self._state == "canceling":
                self._emit(
                    NavigationUpdate(
                        request_id,
                        "canceling",
                        f"{self._interruption_reason} Requesting cancellation from Nav2.".strip(),
                        self._destination,
                    )
                )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.cancel()
