"""Embedding and captioning backends behind one small interface."""

from placecell.providers._http import RetryPolicy, Transport, UrllibTransport
from placecell.providers.base import Capabilities, Captioner, EmbeddingProvider, normalise_rows
from placecell.providers.captioning import OpenAICompatibleCaptioner
from placecell.providers.chat import OpenAICompatibleChat
from placecell.providers.hashing import HashingEmbedder
from placecell.providers.openai_compatible import OpenAICompatibleEmbedder

__all__ = [
    "Capabilities",
    "Captioner",
    "EmbeddingProvider",
    "HashingEmbedder",
    "OpenAICompatibleCaptioner",
    "OpenAICompatibleChat",
    "OpenAICompatibleEmbedder",
    "RetryPolicy",
    "Transport",
    "UrllibTransport",
    "normalise_rows",
]
