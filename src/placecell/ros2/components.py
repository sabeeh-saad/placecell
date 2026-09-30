"""Providers, stores and endpoint credentials built from the node's ROS parameters.

Nothing here imports rclpy. API keys come from the environment variables the parameters
name, never from a parameter, so they do not end up in launch files or logs. The shared
`api_key_env` key goes only to the scheme, host and port of `chat_base_url`; see `endpoint`.
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from placecell.errors import ValidationError
from placecell.missions import MissionPlanner, PlanReviewAgent
from placecell.providers import EmbeddingProvider, HashingEmbedder
from placecell.providers._http import RetryPolicy
from placecell.store import CollectionInfo, VectorStore
from placecell.store.limits import StoreLimits
from placecell.tracing import TraceStore


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
