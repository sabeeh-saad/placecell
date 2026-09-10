"""Embedding and captioning backends behind one small interface."""

from placecell.providers.base import Capabilities, Captioner, EmbeddingProvider, normalise_rows
from placecell.providers.hashing import HashingEmbedder

__all__ = ["Capabilities", "Captioner", "EmbeddingProvider", "HashingEmbedder", "normalise_rows"]
