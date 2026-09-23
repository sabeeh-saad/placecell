"""HTTP plumbing shared by provider adapters: transport, retries, error mapping.

The transport is injectable so tests never open a socket. Retries cover rate limits and
server errors with exponential backoff, honouring `Retry-After` when the server sends one.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from placecell.errors import ProviderError, RateLimitedError, ValidationError
from placecell.providers._contracts import strict_json
from placecell.tracing import current_trace, provider_usage, trace_span

MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _read_body(response: Any) -> Any:
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ProviderError("provider response exceeds the 8 MiB byte limit")
    return decode_body(raw)


class Transport(Protocol):
    """Minimal HTTP surface: post JSON, get status, headers and decoded body back."""

    def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, str], Any]: ...


class UrllibTransport:
    """Standard-library transport. HTTP errors are returned, not raised, so the caller can retry."""

    def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, str], Any]:
        check_http_url(url)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            url, data=body, method="POST", headers={**headers, "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - scheme checked above
                return response.status, dict(response.headers.items()), _read_body(response)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, dict(e.headers.items()), _read_body(e)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(f"request to {url} failed: {e}") from e


def check_http_url(url: str) -> None:
    if not url.startswith(("https://", "http://")):
        raise ValidationError(f"only http(s) endpoints are supported, got {url!r}")


def decode_body(raw: bytes) -> Any:
    try:
        return strict_json(raw.decode("utf-8"), max_chars=MAX_RESPONSE_BYTES) if raw else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")
    except ValueError as e:
        raise ProviderError(f"invalid provider JSON: {e}") from e


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0

    def __post_init__(self) -> None:
        if (
            type(self.attempts) is not int
            or not 1 <= self.attempts <= 32
            or not math.isfinite(self.base_delay_s)
            or not math.isfinite(self.max_delay_s)
            or self.base_delay_s < 0
            or self.max_delay_s < self.base_delay_s
        ):
            raise ValidationError("retry policy out of range")

    def delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                delay = float(retry_after)
                if math.isfinite(delay) and delay >= 0:
                    return min(delay, self.max_delay_s)
            except ValueError:
                pass
        return min(self.base_delay_s * 2.0**attempt, self.max_delay_s)


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Everything needed to call one URL repeatedly."""

    url: str
    headers: Mapping[str, str]
    timeout_s: float
    transport: Transport
    retry: RetryPolicy
    sleep: Callable[[float], None] = time.sleep

    @classmethod
    def build(
        cls,
        base_url: str,
        path: str,
        api_key: str | None,
        timeout_s: float,
        transport: Transport | None,
        retry: RetryPolicy | None,
        sleep: Callable[[float], None],
        extra_headers: Mapping[str, str] | None,
    ) -> Endpoint:
        check_http_url(base_url)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValidationError("timeout_s must be finite and greater than zero")
        headers = {**(extra_headers or {})}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return cls(
            base_url.rstrip("/") + path,
            headers,
            timeout_s,
            transport or UrllibTransport(),
            retry or RetryPolicy(),
            sleep,
        )

    def post(self, payload: Mapping[str, Any]) -> Any:
        """POST until a 200 comes back or the retry budget is spent. Returns the decoded body."""
        context = current_trace()
        if context:
            context.store.register_secrets(
                [
                    value.removeprefix("Bearer ")
                    for key, value in self.headers.items()
                    if key.casefold() in {"authorization", "x-api-key", "x-goog-api-key"}
                ]
            )
        with trace_span("provider_request", model=payload.get("model"), usage=provider_usage(None)) as details:
            return self._post(payload, details)

    def _post(self, payload: Mapping[str, Any], details: dict[str, Any]) -> Any:
        for attempt in range(self.retry.attempts):
            details["attempts"] = attempt + 1
            status, headers, body = self.transport.post_json(self.url, self.headers, payload, self.timeout_s)
            details["http_status"] = status
            details["usage"] = provider_usage(body)
            if status == 200:
                return body
            if status == 429 or status >= 500:
                if attempt + 1 < self.retry.attempts:
                    self.sleep(self.retry.delay(attempt, header(headers, "retry-after")))
                    continue
                if status == 429:
                    raise RateLimitedError(f"{self.url}: rate limited after {self.retry.attempts} attempts")
                raise ProviderError(f"{self.url}: server error {status} after {self.retry.attempts} attempts")
            raise ProviderError(f"{self.url}: HTTP {status}: {message(body)}")
        raise ProviderError("unreachable")  # pragma: no cover


def header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def message(body: Any) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])[:200]
    return str(body)[:200]
