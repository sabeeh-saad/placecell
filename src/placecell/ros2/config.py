"""The node's ROS parameters: names, defaults and declaration order, read into typed settings.

rclpy infers each parameter's type from its default, so a default's type is part of the
interface: 3600.0 must stay a float. `DEFAULTS` keeps the historical declaration order.
The settings are grouped by the part of the node that reads them; each field names its
ROS parameter in its metadata, and `NodeConfig.parameters()` gives back the flat mapping
that the endpoint, trace and storage helpers take.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

DEFAULTS: Mapping[str, Any] = {
    "robot_id": "robot",
    "camera_id": "front",
    "image_topic": "/camera/color/image_raw",
    "recording_dir": "",
    "compressed": False,
    "objects_enabled": False,
    "object_arrival_model": "",
    "object_arrival_request_timeout_s": 8.0,
    "object_arrival_min_similarity": 0.85,
    "object_arrival_moved_similarity": 0.95,
    "object_arrival_similarity_margin": 0.08,
    "object_arrival_max_uncertainty_m": 0.35,
    "object_arrival_max_position_age_s": 300.0,
    "object_arrival_max_move_m": 3.0,
    "object_search_enabled": False,
    "object_search_max_viewpoints": 3,
    "object_search_timeout_s": 60.0,
    "object_search_radius_m": 1.5,
    "object_search_max_path_m": 4.0,
    "approach_enabled": False,
    "approach_costmap_topic": "/global_costmap/costmap_raw",
    "approach_footprint_topic": "/local_costmap/published_footprint",
    "approach_planner_action": "/compute_path_to_pose",
    "approach_planner_id": "",
    "approach_request_timeout_s": 2.0,
    "approach_planning_timeout_s": 8.0,
    "approach_clearance_m": 0.5,
    "approach_max_uncertainty_m": 0.35,
    "approach_max_position_age_s": 300.0,
    "approach_max_sensor_age_s": 2.0,
    "approach_camera_yaw_offset_rad": 0.0,
    "object_model": "",
    "object_backend": "gemini",
    "object_base_url": "https://generativelanguage.googleapis.com/v1beta",
    "object_api_key_env": "GEMINI_API_KEY",
    "object_max_records": 1000,
    "object_max_views": 4,
    "object_retention_s": 2592000.0,
    "object_min_interval_s": 15.0,
    "depth_topic": "/camera/aligned_depth_to_color/image_raw",
    "camera_info_topic": "/camera/color/camera_info",
    "object_depth_max_skew_s": 0.08,
    "rgbd_wait_s": 0.3,
    "rgbd_reliable": False,
    "object_position_error_m": 0.1,
    "object_angular_error_rad": 0.05,
    "map_frame": "map",
    "base_frame": "base_footprint",
    "odom_frame": "odom",
    "map_id": "",
    "localization_required": True,
    "localization_topic": "/amcl_pose",
    "localization_max_age_s": 5.0,
    "sensor_max_age_s": 5.0,
    "sensor_max_future_s": 0.1,
    "sensor_max_failures": 3,
    "localization_max_position_std_m": 0.3,
    "localization_max_yaw_std_rad": 0.35,
    "localization_stationary_translation_m": 0.05,
    "localization_stationary_rotation_rad": 0.05,
    "localization_max_stationary_age_s": 0.0,
    "db_path": "~/.placecell/db",
    "collection": "default",
    "keyframe_dir": "~/.placecell/keyframes",
    "embed_base_url": "",
    "embed_api_key_env": "",
    "embed_model": "",
    "embed_backend": "auto",
    "embed_device": "cpu",
    "embed_revision": "",
    "embed_local_files_only": False,
    "embed_batch_size": 16,
    "embed_cache_folder": "",
    "embed_dimension": 0,
    "caption_base_url": "https://api.openai.com/v1",
    "caption_api_key_env": "",
    "caption_model": "",
    "caption_max_tokens": 1024,
    "chat_base_url": "https://api.openai.com/v1",
    "chat_api_key_env": "",
    "chat_model": "",
    "chat_max_tokens": 400,
    "chat_token_parameter": "max_tokens",
    "chat_temperature": 0.0,
    "chat_max_tool_calls": 16,
    "answer_min_similarity": 0.5,
    "chat_max_context_chars": 40000,
    "api_key_env": "PLACECELL_API_KEY",
    "min_interval_s": 2.0,
    "max_interval_s": 60.0,
    "ingest_attempts": 5,
    "ingest_retry_delay_s": 1.0,
    "camera_max_message_bytes": 8 * 1024 * 1024,
    "question_workers": 2,
    "question_queue": 8,
    "min_travel_m": 0.3,
    "min_turn_rad": 0.35,
    "batch_size": 8,
    "max_queue": 64,
    "memory_max_records": 10000,
    "memory_max_sightings": 1024,
    "memory_evict_at_capacity": True,
    "memory_max_idle_s": 7776000.0,
    "memory_history_age_s": 7776000.0,
    "refine_max_pending": 256,
    "cleanup_max_pending": 2048,
    "correction_max_records": 10000,
    "correction_max_bytes": 4194304,
    "mission_context_max_events": 1000,
    "mission_context_max_bytes": 2097152,
    "mission_context_retention_s": 2592000.0,
    "tf_timeout_s": 0.2,
    "curator_interval_s": 3600.0,
    "contradiction": True,
    "corrections_path": "~/.placecell/corrections.jsonl",
    "consolidate_interval_s": 0.0,
    "refine_interval_s": 3600.0,
    "refine_batch_size": 8,
    "refine_model": "",
    "navigation_enabled": False,
    "mission_enabled": False,
    "mission_model": "",
    "mission_base_url": "",
    "mission_api_key_env": "",
    "mission_review_model": "",
    "mission_review_base_url": "",
    "mission_review_api_key_env": "",
    "mission_request_timeout_s": 8.0,
    "mission_max_tokens": 2048,
    "mission_token_parameter": "max_tokens",
    "mission_temperature": 0.0,
    "mission_max_destinations": 8,
    "mission_context_path": "~/.placecell/missions.sqlite3",
    "mission_trace_path": "",
    "mission_trace_max_events": 10000,
    "mission_trace_max_bytes": 16777216,
    "mission_trace_queue_size": 256,
    "mission_trace_instruction_text": "raw",
    "mission_conversation_id": "default",
    "command_journal_path": "~/.placecell/commands.sqlite3",
    "command_retry_window_s": 86400.0,
    "command_max_records": 10000,
    "verification_model": "",
    "verification_base_url": "",
    "verification_api_key_env": "",
    "verification_request_timeout_s": 8.0,
    "verification_max_tokens": 2048,
    "verification_structured_output": True,
    "navigation_arrival_timeout_s": 30.0,
    "navigation_max_observation_age_s": 5.0,
    "navigation_arrival_max_attempts": 3,
    "nav2_action": "navigate_to_pose",
    "navigation_ownership_path": "~/.placecell/navigation.sqlite3",
    "places_file": "",
    "navigation_min_similarity": 0.5,
    "navigation_min_confidence": 0.2,
    "navigation_max_memory_age_s": 604800.0,
    "navigation_response_timeout_s": 10.0,
    "navigation_lookup_timeout_s": 30.0,
    "navigation_timeout_s": 600.0,
}


PARAMETER_ORDER = tuple(DEFAULTS)


def parameter(name: str) -> Any:
    """A settings field backed by the ROS parameter `name`, with that parameter's default."""
    return field(default=DEFAULTS[name], metadata={"ros": name})


@dataclass(frozen=True)
class CameraConfig:
    robot_id: str = parameter("robot_id")
    camera_id: str = parameter("camera_id")
    image_topic: str = parameter("image_topic")
    compressed: bool = parameter("compressed")
    recording_dir: str = parameter("recording_dir")
    max_message_bytes: int = parameter("camera_max_message_bytes")
    depth_topic: str = parameter("depth_topic")
    info_topic: str = parameter("camera_info_topic")
    rgbd_wait_s: float = parameter("rgbd_wait_s")
    rgbd_reliable: bool = parameter("rgbd_reliable")


@dataclass(frozen=True)
class LocalizationConfig:
    map_frame: str = parameter("map_frame")
    base_frame: str = parameter("base_frame")
    odom_frame: str = parameter("odom_frame")
    map_id: str = parameter("map_id")
    required: bool = parameter("localization_required")
    topic: str = parameter("localization_topic")
    max_age_s: float = parameter("localization_max_age_s")
    max_position_std_m: float = parameter("localization_max_position_std_m")
    max_yaw_std_rad: float = parameter("localization_max_yaw_std_rad")
    stationary_translation_m: float = parameter("localization_stationary_translation_m")
    stationary_rotation_rad: float = parameter("localization_stationary_rotation_rad")
    max_stationary_age_s: float = parameter("localization_max_stationary_age_s")
    tf_timeout_s: float = parameter("tf_timeout_s")


@dataclass(frozen=True)
class SensorConfig:
    max_age_s: float = parameter("sensor_max_age_s")
    max_future_s: float = parameter("sensor_max_future_s")
    max_failures: int = parameter("sensor_max_failures")


@dataclass(frozen=True)
class StorageConfig:
    db_path: str = parameter("db_path")
    collection: str = parameter("collection")
    keyframe_dir: str = parameter("keyframe_dir")
    corrections_path: str = parameter("corrections_path")
    memory_max_records: int = parameter("memory_max_records")
    memory_max_sightings: int = parameter("memory_max_sightings")
    memory_evict_at_capacity: bool = parameter("memory_evict_at_capacity")
    memory_max_idle_s: float = parameter("memory_max_idle_s")
    memory_history_age_s: float = parameter("memory_history_age_s")
    refine_max_pending: int = parameter("refine_max_pending")
    cleanup_max_pending: int = parameter("cleanup_max_pending")
    correction_max_records: int = parameter("correction_max_records")
    correction_max_bytes: int = parameter("correction_max_bytes")


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str = parameter("embed_base_url")
    api_key_env: str = parameter("embed_api_key_env")
    model: str = parameter("embed_model")
    backend: str = parameter("embed_backend")
    device: str = parameter("embed_device")
    revision: str = parameter("embed_revision")
    local_files_only: bool = parameter("embed_local_files_only")
    batch_size: int = parameter("embed_batch_size")
    cache_folder: str = parameter("embed_cache_folder")
    dimension: int = parameter("embed_dimension")


@dataclass(frozen=True)
class CaptionConfig:
    base_url: str = parameter("caption_base_url")
    api_key_env: str = parameter("caption_api_key_env")
    model: str = parameter("caption_model")
    max_tokens: int = parameter("caption_max_tokens")


@dataclass(frozen=True)
class ChatConfig:
    base_url: str = parameter("chat_base_url")
    api_key_env: str = parameter("chat_api_key_env")
    model: str = parameter("chat_model")
    max_tokens: int = parameter("chat_max_tokens")
    token_parameter: str = parameter("chat_token_parameter")
    temperature: float = parameter("chat_temperature")
    max_tool_calls: int = parameter("chat_max_tool_calls")
    max_context_chars: int = parameter("chat_max_context_chars")
    shared_api_key_env: str = parameter("api_key_env")


@dataclass(frozen=True)
class QuestionConfig:
    workers: int = parameter("question_workers")
    queue: int = parameter("question_queue")
    answer_min_similarity: float = parameter("answer_min_similarity")


@dataclass(frozen=True)
class IngestConfig:
    min_interval_s: float = parameter("min_interval_s")
    max_interval_s: float = parameter("max_interval_s")
    min_travel_m: float = parameter("min_travel_m")
    min_turn_rad: float = parameter("min_turn_rad")
    batch_size: int = parameter("batch_size")
    max_queue: int = parameter("max_queue")
    attempts: int = parameter("ingest_attempts")
    retry_delay_s: float = parameter("ingest_retry_delay_s")
    contradiction: bool = parameter("contradiction")


@dataclass(frozen=True)
class MaintenanceConfig:
    curator_interval_s: float = parameter("curator_interval_s")
    consolidate_interval_s: float = parameter("consolidate_interval_s")
    refine_interval_s: float = parameter("refine_interval_s")
    refine_batch_size: int = parameter("refine_batch_size")
    refine_model: str = parameter("refine_model")


@dataclass(frozen=True)
class ObjectConfig:
    enabled: bool = parameter("objects_enabled")
    model: str = parameter("object_model")
    backend: str = parameter("object_backend")
    base_url: str = parameter("object_base_url")
    api_key_env: str = parameter("object_api_key_env")
    max_records: int = parameter("object_max_records")
    max_views: int = parameter("object_max_views")
    retention_s: float = parameter("object_retention_s")
    min_interval_s: float = parameter("object_min_interval_s")
    depth_max_skew_s: float = parameter("object_depth_max_skew_s")
    position_error_m: float = parameter("object_position_error_m")
    angular_error_rad: float = parameter("object_angular_error_rad")


@dataclass(frozen=True)
class ObjectArrivalConfig:
    model: str = parameter("object_arrival_model")
    request_timeout_s: float = parameter("object_arrival_request_timeout_s")
    min_similarity: float = parameter("object_arrival_min_similarity")
    moved_similarity: float = parameter("object_arrival_moved_similarity")
    similarity_margin: float = parameter("object_arrival_similarity_margin")
    max_uncertainty_m: float = parameter("object_arrival_max_uncertainty_m")
    max_position_age_s: float = parameter("object_arrival_max_position_age_s")
    max_move_m: float = parameter("object_arrival_max_move_m")


@dataclass(frozen=True)
class ObjectSearchConfig:
    enabled: bool = parameter("object_search_enabled")
    max_viewpoints: int = parameter("object_search_max_viewpoints")
    timeout_s: float = parameter("object_search_timeout_s")
    radius_m: float = parameter("object_search_radius_m")
    max_path_m: float = parameter("object_search_max_path_m")


@dataclass(frozen=True)
class ApproachConfig:
    enabled: bool = parameter("approach_enabled")
    costmap_topic: str = parameter("approach_costmap_topic")
    footprint_topic: str = parameter("approach_footprint_topic")
    planner_action: str = parameter("approach_planner_action")
    planner_id: str = parameter("approach_planner_id")
    request_timeout_s: float = parameter("approach_request_timeout_s")
    planning_timeout_s: float = parameter("approach_planning_timeout_s")
    clearance_m: float = parameter("approach_clearance_m")
    max_uncertainty_m: float = parameter("approach_max_uncertainty_m")
    max_position_age_s: float = parameter("approach_max_position_age_s")
    max_sensor_age_s: float = parameter("approach_max_sensor_age_s")
    camera_yaw_offset_rad: float = parameter("approach_camera_yaw_offset_rad")


@dataclass(frozen=True)
class NavigationConfig:
    enabled: bool = parameter("navigation_enabled")
    nav2_action: str = parameter("nav2_action")
    ownership_path: str = parameter("navigation_ownership_path")
    places_file: str = parameter("places_file")
    min_similarity: float = parameter("navigation_min_similarity")
    min_confidence: float = parameter("navigation_min_confidence")
    max_memory_age_s: float = parameter("navigation_max_memory_age_s")
    response_timeout_s: float = parameter("navigation_response_timeout_s")
    lookup_timeout_s: float = parameter("navigation_lookup_timeout_s")
    timeout_s: float = parameter("navigation_timeout_s")
    arrival_timeout_s: float = parameter("navigation_arrival_timeout_s")
    max_observation_age_s: float = parameter("navigation_max_observation_age_s")
    arrival_max_attempts: int = parameter("navigation_arrival_max_attempts")
    command_journal_path: str = parameter("command_journal_path")
    command_retry_window_s: float = parameter("command_retry_window_s")
    command_max_records: int = parameter("command_max_records")


@dataclass(frozen=True)
class VerificationConfig:
    model: str = parameter("verification_model")
    base_url: str = parameter("verification_base_url")
    api_key_env: str = parameter("verification_api_key_env")
    request_timeout_s: float = parameter("verification_request_timeout_s")
    max_tokens: int = parameter("verification_max_tokens")
    structured_output: bool = parameter("verification_structured_output")


@dataclass(frozen=True)
class MissionConfig:
    enabled: bool = parameter("mission_enabled")
    model: str = parameter("mission_model")
    base_url: str = parameter("mission_base_url")
    api_key_env: str = parameter("mission_api_key_env")
    review_model: str = parameter("mission_review_model")
    review_base_url: str = parameter("mission_review_base_url")
    review_api_key_env: str = parameter("mission_review_api_key_env")
    request_timeout_s: float = parameter("mission_request_timeout_s")
    max_tokens: int = parameter("mission_max_tokens")
    token_parameter: str = parameter("mission_token_parameter")
    temperature: float = parameter("mission_temperature")
    max_destinations: int = parameter("mission_max_destinations")
    conversation_id: str = parameter("mission_conversation_id")
    context_path: str = parameter("mission_context_path")
    context_max_events: int = parameter("mission_context_max_events")
    context_max_bytes: int = parameter("mission_context_max_bytes")
    context_retention_s: float = parameter("mission_context_retention_s")
    trace_path: str = parameter("mission_trace_path")
    trace_max_events: int = parameter("mission_trace_max_events")
    trace_max_bytes: int = parameter("mission_trace_max_bytes")
    trace_queue_size: int = parameter("mission_trace_queue_size")
    trace_instruction_text: str = parameter("mission_trace_instruction_text")


@dataclass(frozen=True)
class NodeConfig:
    """Every node setting; `NodeConfig()` holds the defaults."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    questions: QuestionConfig = field(default_factory=QuestionConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    object_arrival: ObjectArrivalConfig = field(default_factory=ObjectArrivalConfig)
    object_search: ObjectSearchConfig = field(default_factory=ObjectSearchConfig)
    approach: ApproachConfig = field(default_factory=ApproachConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)

    @classmethod
    def from_parameters(cls, values: Mapping[str, Any]) -> NodeConfig:
        """Settings from a value for every name in PARAMETER_ORDER."""
        defaults, groups = cls(), {}
        for group in fields(cls):
            settings = getattr(defaults, group.name)
            groups[group.name] = type(settings)(**{f.name: values[f.metadata["ros"]] for f in fields(settings)})
        return cls(**groups)

    def parameters(self) -> dict[str, Any]:
        """The flat parameter mapping, in declaration order."""
        values = {
            f.metadata["ros"]: getattr(getattr(self, group.name), f.name)
            for group in fields(self)
            for f in fields(getattr(self, group.name))
        }
        return {name: values[name] for name in PARAMETER_ORDER}


def declare(node: Any) -> NodeConfig:
    """Declare every parameter on `node`, in PARAMETER_ORDER, and read the values it was given."""
    return NodeConfig.from_parameters({k: node.declare_parameter(k, v).value for k, v in DEFAULTS.items()})
