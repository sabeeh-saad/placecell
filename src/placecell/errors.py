"""Exception hierarchy. Everything placecell raises derives from PlacecellError."""

from __future__ import annotations


class PlacecellError(Exception):
    """Base class for all placecell errors."""


class ValidationError(PlacecellError, ValueError):
    """A value violates the data model (empty id, negative radius, wrong vector shape, ...)."""


class FrameMismatchError(ValidationError):
    """Two poses were compared although they live in different map frames."""


class ModelMismatchError(PlacecellError):
    """A vector produced by one embedding model was offered to a collection bound to another."""


class UnsupportedMediaError(PlacecellError):
    """A provider was asked to embed or caption a media kind it does not support."""


class ProviderError(PlacecellError):
    """An embedding or captioning backend failed."""


class RateLimitedError(ProviderError):
    """The backend refused the request because of rate limits after all retries were used."""
