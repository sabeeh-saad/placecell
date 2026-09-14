"""Embedding and captioning backends behind one small interface."""

from placecell.providers._http import RetryPolicy, Transport, UrllibTransport
from placecell.providers.base import Capabilities, Captioner, EmbeddingProvider, QueryEmbeddingProvider, normalise_rows
from placecell.providers.captioning import OpenAICompatibleCaptioner
from placecell.providers.chat import OpenAICompatibleChat
from placecell.providers.clip import ClipEmbedder
from placecell.providers.gemini import GeminiEmbedder
from placecell.providers.hashing import HashingEmbedder
from placecell.providers.object_detection import GeminiObjectDetector
from placecell.providers.openai_compatible import OpenAICompatibleEmbedder

__all__ = [
    "Capabilities",
    "Captioner",
    "ClipEmbedder",
    "EmbeddingProvider",
    "GeminiEmbedder",
    "GeminiObjectDetector",
    "HashingEmbedder",
    "OpenAICompatibleCaptioner",
    "OpenAICompatibleChat",
    "OpenAICompatibleEmbedder",
    "QueryEmbeddingProvider",
    "RetryPolicy",
    "Transport",
    "UrllibTransport",
    "normalise_rows",
]
