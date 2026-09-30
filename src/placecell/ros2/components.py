"""Everything the node is built from except its ROS entities, in the order it is built.

Providers, stores, workers and the navigation stack come from the node's settings; nothing
here imports rclpy, and the node passes its ROS-side factories in as `RosPorts`. API keys
come from the environment variables the parameters name, never from a parameter, so they
do not end up in launch files or logs. The shared `api_key_env` key goes only to the
scheme, host and port of `chat_base_url`; see `endpoint`.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from placecell.agent import Agent
from placecell.approach import ApproachPlanner, ApproachPolicy, PlanningEnvironment
from placecell.command_identity import CommandJournal, CommandScope
from placecell.consolidation import ChatSummarizer, Consolidator
from placecell.corrections import JsonlCorrectionLog
from placecell.errors import ValidationError
from placecell.lifecycle import Curator, RetentionPolicy, remove_local_file
from placecell.localization import LocalizationGate, LocalizationPolicy
from placecell.mission_context import MissionContext
from placecell.missions import MissionPlanner, PlanReviewAgent
from placecell.navigation import (
    DestinationResolver,
    NavigationCommands,
    NavigationPolicy,
    NavigationUpdate,
    load_named_places,
)
from placecell.navigation_ownership import NavigationOwnership, NavigationScope
from placecell.object_arrival import ObjectArrivalPolicy, ObjectArrivalVerifier
from placecell.object_search import ObjectSearch, ObjectSearchPolicy
from placecell.objects import ObjectPolicy, ObjectRecall, ObjectTracker
from placecell.observer import Observer
from placecell.pipeline import Ingester, SegmentationPolicy, Segmenter
from placecell.providers import Captioner, EmbeddingProvider, HashingEmbedder
from placecell.providers._http import RetryPolicy
from placecell.recordings import RecordingWriter
from placecell.refinement import REFINEMENT_PROMPT, MemoryRefiner, RefinementPolicy
from placecell.retrieval import Recall
from placecell.ros2.bridge import KeyframeWriter, ObservationBuilder
from placecell.ros2.config import NodeConfig
from placecell.ros2.depth import PendingImages
from placecell.ros2.navigation import Nav2Navigator
from placecell.ros2.workers import BoundedTasks, IngestWorker
from placecell.sensors import SensorHealth
from placecell.store import CollectionInfo, VectorStore
from placecell.store.limits import StoreLimits
from placecell.tracing import TraceStore
from placecell.verification import VisionVerifier


def build_embedder(
    base_url: str,
    model: str,
    api_key: str | None,
    dimension: int,
    *,
    backend: str = "auto",
    device: str = "cpu",
    revision: str = "",
    local_files_only: bool = False,
    batch_size: int = 16,
    cache_folder: str = "",
) -> EmbeddingProvider:
    if backend == "openrouter":
        from placecell.providers.openrouter import OPENROUTER_BASE_URL, OpenRouterGeminiEmbedder

        return OpenRouterGeminiEmbedder(
            model or "google/gemini-embedding-2",
            api_key=api_key,
            dimension=dimension or 768,
            base_url=base_url or OPENROUTER_BASE_URL,
            batch_size=batch_size,
            retry=RetryPolicy(attempts=1),
        )
    if backend == "gemini":
        from placecell.providers.gemini import DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, GeminiEmbedder

        return GeminiEmbedder(
            model or DEFAULT_GEMINI_MODEL,
            api_key=api_key,
            dimension=dimension or 768,
            base_url=base_url or GEMINI_BASE_URL,
            batch_size=batch_size,
            retry=RetryPolicy(attempts=1),
        )
    if backend == "clip":
        from placecell.providers.clip import DEFAULT_CLIP_MODEL, ClipEmbedder

        embedder = ClipEmbedder(
            model or DEFAULT_CLIP_MODEL,
            device=device,
            revision=revision or None,
            local_files_only=local_files_only,
            batch_size=batch_size,
            cache_folder=cache_folder or None,
        )
        if dimension and dimension != embedder.dimension:
            raise ValidationError("embed_dimension does not match the CLIP checkpoint")
        return embedder
    if backend != "auto":
        raise ValidationError("embed_backend must be auto, gemini, openrouter or clip")
    if not model:
        return HashingEmbedder()
    from placecell.providers import OpenAICompatibleEmbedder

    return OpenAICompatibleEmbedder(
        model,
        base_url or "https://api.openai.com/v1",
        api_key,
        dimension=dimension or None,
        retry=RetryPolicy(attempts=1),
    )


def build_store(
    db_path: str, collection: str, embedder: EmbeddingProvider, *, limits: StoreLimits | None = None
) -> VectorStore:
    info = CollectionInfo(collection, embedder.model_name, embedder.dimension)
    if not db_path:
        from placecell.store import InMemoryStore

        return InMemoryStore(info, limits=limits)
    from placecell.store.lancedb_store import LanceDBStore

    return LanceDBStore(Path(db_path).expanduser(), info, limits=limits)


# Group -> (base URL parameter, key env parameter, group an empty base URL falls back to).
ENDPOINTS = {
    "chat": ("chat_base_url", "chat_api_key_env", ""),
    "caption": ("caption_base_url", "caption_api_key_env", ""),
    "verification": ("verification_base_url", "verification_api_key_env", "caption"),
    "mission": ("mission_base_url", "mission_api_key_env", "chat"),
    "mission_review": ("mission_review_base_url", "mission_review_api_key_env", "mission"),
}


def endpoint(parameters: Mapping[str, Any], group: str) -> tuple[str, str | None]:
    """Base URL and API key of one endpoint group.

    The group's own `*_api_key_env` wins. An empty base URL uses the fallback group's URL
    and key. Otherwise the group gets the shared key only on `chat_base_url`'s origin.
    """
    url_parameter, key_parameter, fallback = ENDPOINTS[group]
    url = parameters[url_parameter]
    if not url and fallback:
        url, key = endpoint(parameters, fallback)
    else:
        key = shared_api_key(parameters, url)
    if parameters[key_parameter]:
        key = os.environ.get(parameters[key_parameter]) or None
    return url, key


def embedding_api_key(parameters: Mapping[str, Any]) -> str | None:
    """`embed_api_key_env`, then GEMINI_API_KEY for Gemini, then the shared key on the chat origin."""
    if parameters["embed_api_key_env"]:
        return os.environ.get(parameters["embed_api_key_env"]) or None
    backend = parameters["embed_backend"]
    if backend == "gemini" and os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    from placecell.providers.gemini import GEMINI_BASE_URL
    from placecell.providers.openrouter import OPENROUTER_BASE_URL

    defaults = {"gemini": GEMINI_BASE_URL, "openrouter": OPENROUTER_BASE_URL}
    url = parameters["embed_base_url"] or defaults.get(backend, "https://api.openai.com/v1")
    return shared_api_key(parameters, url)


def shared_api_key(parameters: Mapping[str, Any], base_url: str) -> str | None:
    """The `api_key_env` key, only for the scheme, host and port of `chat_base_url`."""
    origin = _origin(base_url)
    if origin is None or origin != _origin(parameters["chat_base_url"]):
        return None
    return os.environ.get(parameters["api_key_env"]) or None


def _origin(url: str) -> tuple[str, str, int | None] | None:
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    except ValueError:
        return None
    return (parts.scheme, parts.hostname, port) if parts.hostname else None


def chat_options(parameters: Mapping[str, Any], group: str) -> dict[str, Any]:
    """Request shape of a chat group. A negative `*_temperature` omits it, as reasoning models require."""
    temperature = parameters[f"{group}_temperature"]
    return {
        "max_tokens": parameters[f"{group}_max_tokens"],
        "token_parameter": parameters[f"{group}_token_parameter"],
        "temperature": None if temperature < 0 else temperature,
    }


def build_mission_planner(parameters: dict[str, Any]) -> MissionPlanner | None:
    if not parameters["mission_enabled"]:
        return None
    from placecell.providers import OpenAICompatibleChat

    model = parameters["mission_model"]
    if not model:
        raise ValidationError("mission_enabled requires mission_model with tool calling")
    base_url, api_key = endpoint(parameters, "mission")
    review_url, review_key = endpoint(parameters, "mission_review")
    options = {
        "timeout_s": parameters["mission_request_timeout_s"],
        "retry": RetryPolicy(attempts=1),
        **chat_options(parameters, "mission"),
    }
    planner = OpenAICompatibleChat(model, base_url, api_key, **options)
    reviewer = OpenAICompatibleChat(parameters["mission_review_model"] or model, review_url, review_key, **options)
    return MissionPlanner(planner, PlanReviewAgent(reviewer), max_destinations=parameters["mission_max_destinations"])


def build_trace_store(parameters: dict[str, Any]) -> TraceStore | None:
    path = parameters["mission_trace_path"]
    if not path:
        return None
    return TraceStore(
        path,
        max_events=parameters["mission_trace_max_events"],
        max_bytes=parameters["mission_trace_max_bytes"],
        queue_size=parameters["mission_trace_queue_size"],
        instruction_text=parameters.get("mission_trace_instruction_text", "raw"),
        # The Gemini embedding backend also reads GEMINI_API_KEY without a parameter naming it.
        secrets=[os.environ.get(value, "") for key, value in parameters.items() if key.endswith("api_key_env")]
        + [os.environ.get("GEMINI_API_KEY", "")],
    )


@dataclass(frozen=True)
class Components:
    """The memory pipeline and its workers. Built before the node creates any ROS entity."""

    embedder: EmbeddingProvider
    store: VectorStore
    admission: Segmenter
    object_policy: ObjectPolicy
    object_recall: ObjectRecall | None
    tracker: ObjectTracker | None
    detector_type: Any  # the object detector class, reused for arrival checks
    corrections: JsonlCorrectionLog
    recall: Recall
    agent: Agent | None
    answer_min_similarity: float
    consolidator: Consolidator | None
    refiner: MemoryRefiner | None
    worker: IngestWorker
    questions: BoundedTasks
    maintenance: BoundedTasks
    indexing: BoundedTasks
    curator: Curator
    writer: KeyframeWriter
    builder: ObservationBuilder
    recording: RecordingWriter | None
    sensors: SensorHealth


def build_components(config: NodeConfig, *, clock: Callable[[], float], log: Any, resources: ExitStack) -> Components:
    """Validate the settings and build the pipeline, in the order the node always built it.

    Each store and worker registers its release on `resources` as soon as it exists.
    """
    p = config.parameters()
    if config.navigation.enabled and (not config.localization.map_id.strip() or not config.localization.required):
        raise ValidationError("Navigation requires a versioned map_id and localization_required:=true.")
    embedder = build_embedder(
        config.embedding.base_url,
        config.embedding.model,
        embedding_api_key(p),
        config.embedding.dimension,
        backend=config.embedding.backend,
        device=config.embedding.device,
        revision=config.embedding.revision,
        local_files_only=config.embedding.local_files_only,
        batch_size=config.embedding.batch_size,
        cache_folder=config.embedding.cache_folder,
    )
    store = build_store(
        config.storage.db_path,
        config.storage.collection,
        embedder,
        limits=StoreLimits(
            config.storage.memory_max_records,
            config.storage.memory_max_sightings,
            config.storage.refine_max_pending,
            config.storage.cleanup_max_pending,
            config.storage.memory_evict_at_capacity,
        ),
    )
    resources.callback(store.close)
    captioner: Captioner | None = None
    caption_url, caption_key = endpoint(p, "caption")
    if config.caption.model:
        from placecell.providers import OpenAICompatibleCaptioner

        captioner = OpenAICompatibleCaptioner(
            config.caption.model,
            caption_url,
            caption_key,
            max_tokens=config.caption.max_tokens,
            retry=RetryPolicy(attempts=1),
        )
    if captioner is None and not embedder.capabilities.image:
        log.warning(
            "no caption_model and the embedder takes text only: frames cannot be stored. "
            "Set caption_model, or use an embedding model that accepts images."
        )
    policy = SegmentationPolicy(
        config.ingest.min_interval_s,
        config.ingest.min_travel_m,
        config.ingest.min_turn_rad,
        config.ingest.max_interval_s,
    )
    segmenter = Segmenter(policy)
    admission = Segmenter(policy)
    observer = Observer(store) if config.ingest.contradiction else None
    object_policy = ObjectPolicy(
        max_objects=config.objects.max_records,
        max_views=config.objects.max_views,
        retention_s=config.objects.retention_s,
        min_interval_s=config.objects.min_interval_s,
        require_position=bool(config.camera.depth_topic),
    )
    object_recall: ObjectRecall | None = None
    tracker = None
    detector_type: Any = None
    if config.objects.backend not in {"gemini", "chat"}:
        raise ValidationError("object_backend must be gemini or chat")
    if config.objects.enabled:
        from placecell.providers.object_detection import ChatObjectDetector, GeminiObjectDetector

        detector_type = ChatObjectDetector if config.objects.backend == "chat" else GeminiObjectDetector
        detector = detector_type(
            config.objects.model,
            api_key=os.environ.get(config.objects.api_key_env, ""),
            base_url=config.objects.base_url,
            retry=RetryPolicy(attempts=1),
        )
        tracker = ObjectTracker(store, embedder, detector, object_policy)
        object_recall = ObjectRecall(store, embedder, clock=clock)
    ingester = Ingester(
        embedder,
        store,
        captioner,
        segmenter,
        batch_size=config.ingest.batch_size,
        observer=observer,
        objects=tracker,
    )
    corrections = JsonlCorrectionLog(
        Path(config.storage.corrections_path).expanduser(),
        max_records=config.storage.correction_max_records,
        max_bytes=config.storage.correction_max_bytes,
    )
    recall = Recall(store, embedder, corrections=corrections, clock=clock)
    agent: Agent | None = None
    answer_min_similarity = float(config.questions.answer_min_similarity)
    if not 0 < answer_min_similarity <= 1:
        raise ValidationError("answer_min_similarity must be within (0, 1]")
    consolidator: Consolidator | None = None
    refiner: MemoryRefiner | None = None
    refinement_model = config.maintenance.refine_model or config.caption.model
    if config.maintenance.refine_interval_s > 0 and refinement_model:
        from placecell.providers import OpenAICompatibleCaptioner

        reviewer = OpenAICompatibleCaptioner(
            refinement_model,
            caption_url,
            caption_key,
            prompt=REFINEMENT_PROMPT,
            max_tokens=config.caption.max_tokens,
            detail="high",
        )
        refiner = MemoryRefiner(
            store,
            embedder,
            reviewer,
            RefinementPolicy(max_memories=config.maintenance.refine_batch_size),
            producer=refinement_model,
        )
    if config.chat.model:
        from placecell.providers import OpenAICompatibleChat

        chat = OpenAICompatibleChat(config.chat.model, *endpoint(p, "chat"), **chat_options(p, "chat"))
        agent = Agent(
            recall,
            chat,
            frame_id=config.localization.map_frame,
            map_id=config.localization.map_id,
            clock=clock,
            max_tool_calls=config.chat.max_tool_calls,
            max_context_chars=config.chat.max_context_chars,
        )
        if config.maintenance.consolidate_interval_s > 0:
            consolidator = Consolidator(store, embedder, ChatSummarizer(chat))
    worker = IngestWorker(
        ingester,
        None,
        config.ingest.batch_size,
        config.ingest.max_queue,
        log,
        max_attempts=config.ingest.attempts,
        retry_delay_s=config.ingest.retry_delay_s,
    )
    resources.callback(worker.stop)
    questions = BoundedTasks(config.questions.workers, config.questions.queue, log)
    resources.callback(questions.stop)
    maintenance = BoundedTasks(1, 1, log)
    resources.callback(maintenance.stop)
    # A separate worker, so frequent index syncs never crowd out hourly maintenance.
    indexing = BoundedTasks(1, 1, log)
    resources.callback(indexing.stop)
    curator = Curator(
        store,
        RetentionPolicy(max_idle_s=config.storage.memory_max_idle_s, history_age_s=config.storage.memory_history_age_s),
        corrections=corrections,
        remover=remove_local_file,
        clock=clock,
    )
    writer = KeyframeWriter(Path(config.storage.keyframe_dir).expanduser())
    writer.recover_pending(store)
    store.drain_cleanup(remove_local_file)
    builder = ObservationBuilder(config.camera.robot_id, config.camera.camera_id, writer)
    recording = RecordingWriter(config.camera.recording_dir) if config.camera.recording_dir else None
    sensors = SensorHealth(
        config.sensors.max_age_s,
        max_future_s=config.sensors.max_future_s,
        max_failures=config.sensors.max_failures,
        clock=clock,
    )
    return Components(
        embedder=embedder,
        store=store,
        admission=admission,
        object_policy=object_policy,
        object_recall=object_recall,
        tracker=tracker,
        detector_type=detector_type,
        corrections=corrections,
        recall=recall,
        agent=agent,
        answer_min_similarity=answer_min_similarity,
        consolidator=consolidator,
        refiner=refiner,
        worker=worker,
        questions=questions,
        maintenance=maintenance,
        indexing=indexing,
        curator=curator,
        writer=writer,
        builder=builder,
        recording=recording,
        sensors=sensors,
    )


def build_localization(config: NodeConfig, *, clock: Callable[[], float]) -> LocalizationGate:
    return LocalizationGate(
        config.localization.map_frame,
        config.localization.map_id,
        LocalizationPolicy(
            max_age_s=config.localization.max_age_s,
            max_position_std_m=config.localization.max_position_std_m,
            max_yaw_std_rad=config.localization.max_yaw_std_rad,
            max_capture_future_s=config.sensors.max_future_s,
            stationary_translation_m=config.localization.stationary_translation_m,
            stationary_rotation_rad=config.localization.stationary_rotation_rad,
            max_stationary_age_s=config.localization.max_stationary_age_s or math.inf,
        ),
        clock=clock,
    )


def build_pending_images(config: NodeConfig) -> PendingImages:
    return PendingImages(
        config.objects.depth_max_skew_s,
        wait_s=config.camera.rgbd_wait_s,
        max_age_s=config.sensors.max_age_s,
        max_message_bytes=config.camera.max_message_bytes,
        max_future_s=config.sensors.max_future_s,
    )


@dataclass(frozen=True)
class RosPorts:
    """ROS-side factories the navigation stack needs; the node binds them to itself."""

    create_navigator: Callable[..., Nav2Navigator]
    create_planning_environment: Callable[..., PlanningEnvironment]
    resolve_topic_name: Callable[[str], str]


@dataclass(frozen=True)
class Navigation:
    journal: CommandJournal
    traces: TraceStore | None
    context: MissionContext | None
    navigator: Nav2Navigator
    tasks: BoundedTasks
    commands: NavigationCommands


def build_navigation(
    config: NodeConfig,
    parts: Components,
    localization: LocalizationGate,
    *,
    clock: Callable[[], float],
    log: Any,
    ros: RosPorts,
    submit: Callable[[Callable[[], None]], bool],
    publish: Callable[[NavigationUpdate], None],
    references_available: Callable[[dict[str, Any]], bool],
    resources: ExitStack,
) -> Navigation:
    """The command journal, mission context and traces, destination resolver, Nav2 client and controller.

    Each journal, store, client and worker registers its release on `resources` as soon as it exists.
    """
    p = config.parameters()
    store, embedder, sensors = parts.store, parts.embedder, parts.sensors
    journal = CommandJournal(
        config.navigation.command_journal_path or ":memory:",
        CommandScope(config.camera.robot_id, config.localization.map_id, config.mission.conversation_id),
        retry_window_s=config.navigation.command_retry_window_s,
        max_records=config.navigation.command_max_records,
    )
    resources.callback(journal.close)
    traces = build_trace_store(p)
    if traces is not None:
        resources.callback(traces.close)
    context: MissionContext | None = None
    if config.mission.enabled:
        context = MissionContext(
            config.mission.context_path or ":memory:",
            scope=json.dumps([config.camera.robot_id, config.localization.map_id, config.mission.conversation_id]),
            max_events=config.mission.context_max_events,
            max_bytes=config.mission.context_max_bytes,
            retention_s=config.mission.context_retention_s,
            references_available=references_available,
        )
        resources.callback(context.close)
    places = load_named_places(config.navigation.places_file) if config.navigation.places_file else {}
    verification_model = config.verification.model or config.caption.model
    verifier = (
        VisionVerifier(
            verification_model,
            *endpoint(p, "verification"),
            timeout_s=config.verification.request_timeout_s,
            max_tokens=config.verification.max_tokens,
            structured_output=config.verification.structured_output,
        )
        if verification_model
        else None
    )
    object_arrival = None
    if parts.tracker is not None:
        arrival_detector = parts.detector_type(
            config.object_arrival.model or config.objects.model,
            api_key=os.environ.get(config.objects.api_key_env, ""),
            base_url=config.objects.base_url,
            timeout_s=config.object_arrival.request_timeout_s,
            retry=RetryPolicy(attempts=1),
        )
        object_arrival = ObjectArrivalVerifier(
            ObjectTracker(store, embedder, arrival_detector, parts.object_policy),
            arrival_detector,
            ObjectArrivalPolicy(
                max_observation_age_s=config.navigation.max_observation_age_s,
                min_similarity=config.object_arrival.min_similarity,
                moved_similarity=config.object_arrival.moved_similarity,
                similarity_margin=config.object_arrival.similarity_margin,
                max_uncertainty_m=config.object_arrival.max_uncertainty_m,
                max_position_age_s=config.object_arrival.max_position_age_s,
                max_move_m=config.object_arrival.max_move_m,
            ),
            clock=clock,
        )
    approach = None
    if config.approach.enabled or config.object_search.enabled:
        if parts.object_recall is None:
            raise ValidationError("approach planning requires objects_enabled")
        environment = ros.create_planning_environment(
            frame_id=config.localization.map_frame,
            map_id=config.localization.map_id,
            base_frame=config.localization.base_frame,
            costmap_topic=config.approach.costmap_topic,
            footprint_topic=config.approach.footprint_topic,
            action_name=config.approach.planner_action,
            planner_id=config.approach.planner_id,
            timeout_s=config.approach.request_timeout_s,
        )
        approach = ApproachPlanner(
            environment,
            ApproachPolicy(
                clearance_m=config.approach.clearance_m,
                max_uncertainty_m=config.approach.max_uncertainty_m,
                camera_yaw_offset_rad=config.approach.camera_yaw_offset_rad,
                max_sensor_age_s=config.approach.max_sensor_age_s,
                max_position_age_s=config.approach.max_position_age_s,
                planning_timeout_s=config.approach.planning_timeout_s,
            ),
            clock=clock,
        )
    resolver = DestinationResolver(
        store,
        parts.recall,
        robot_id=config.camera.robot_id,
        camera_id=config.camera.camera_id,
        frame_id=config.localization.map_frame,
        map_id=config.localization.map_id,
        clock=clock,
        places=places,
        verifier=verifier,
        objects=parts.object_recall,
        approach=approach if config.approach.enabled else None,
        object_arrival=object_arrival,
        policy=NavigationPolicy(
            min_similarity=config.navigation.min_similarity,
            min_confidence=config.navigation.min_confidence,
            max_age_s=config.navigation.max_memory_age_s,
        ),
    )
    navigator = ros.create_navigator(
        config.navigation.nav2_action,
        config.navigation.response_timeout_s,
        config.navigation.timeout_s,
        ownership=NavigationOwnership(
            config.navigation.ownership_path,
            NavigationScope(
                config.camera.robot_id,
                config.localization.map_id,
                ros.resolve_topic_name(config.navigation.nav2_action),
            ),
        ),
    )
    resources.callback(navigator.close)
    tasks = BoundedTasks(1, 1, log)
    resources.callback(tasks.stop)
    commands = NavigationCommands(
        resolver,
        navigator,
        submit,
        publish,
        request_timeout_s=config.navigation.lookup_timeout_s,
        observation_clock=clock,
        localization_ready=lambda: sensors.ready() and localization.ready(),
        localization_generation=lambda: localization.generation,
        sensor_ready=lambda d: sensors.ready(camera=d.source == "memory", depth=bool(d.object_id)),
        sensor_generation=lambda d: sensors.generation(depth=bool(d.object_id)) if d.source == "memory" else 0,
        arrival_timeout_s=config.navigation.arrival_timeout_s,
        max_observation_age_s=config.navigation.max_observation_age_s,
        arrival_max_attempts=config.navigation.arrival_max_attempts,
        mission_planner=build_mission_planner(p),
        mission_context=context,
        trace_store=traces,
        startup_block_reason=lambda: navigator.startup_block_reason,
        search=ObjectSearch(
            approach,
            ObjectSearchPolicy(
                max_viewpoints=config.object_search.max_viewpoints,
                timeout_s=config.object_search.timeout_s,
                radius_m=config.object_search.radius_m,
                max_path_m=config.object_search.max_path_m,
            ),
        )
        if config.object_search.enabled and approach is not None
        else None,
    )
    return Navigation(journal, traces, context, navigator, tasks, commands)
