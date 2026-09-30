"""HTTP plumbing shared by provider adapters: transport, retries, error mapping.

The transport is injectable so tests never open a socket. Retries use jittered exponential
backoff and honour `Retry-After` up to a cap; a longer requested wait fails at once. 429 and
503 are retried for every call. Other server errors and transport failures may follow work
the provider already did (and billed), so only idempotent calls such as embeddings repeat them.
Redirects are never followed, and credentials go over plain http only to this machine.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from http.client import HTTPMessage
from typing import IO, Any, NoReturn, Protocol

from placecell.errors import ProviderError, RateLimitedError, ValidationError
from placecell.providers._contracts import strict_json
from placecell.tracing import current_trace, provider_usage, trace_span

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key", "x-goog-api-key"})


def _read_body(response: Any) -> Any:
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ProviderError("provider response exceeds the 8 MiB byte limit")
    return decode_body(raw)


class TransportError(ProviderError):
    """The request failed on the network: it may or may not have reached the provider."""


class Transport(Protocol):
    """Minimal HTTP surface: post JSON, get status, headers and decoded body back."""

    def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, str], Any]: ...


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """urllib would resend the credential headers to the redirect target, on any host."""

    # Python 3.10 has no 308 handler, so a 308 would otherwise come back as a plain HTTP error.
    http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302

    def redirect_request(
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage, newurl: str
    ) -> NoReturn:
        fp.close()
        raise ProviderError(f"{req.full_url}: refused HTTP {code} redirect to {newurl[:200]}")


class UrllibTransport:
    """Standard-library transport. HTTP errors are returned, not raised, so the caller can retry.

    Network failures raise TransportError, including http.client errors that are not OSErrors.
    """

    def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, str], Any]:
        check_http_url(url)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            url, data=body, method="POST", headers={**headers, "Content-Type": "application/json"}
        )
        try:
            try:
                with urllib.request.build_opener(_RefuseRedirects).open(request, timeout=timeout_s) as response:
                    return response.status, dict(response.headers.items()), _read_body(response)
            except urllib.error.HTTPError as e:
                with e:
                    return e.code, dict(e.headers.items()), _read_body(e)
        # URLError, timeouts and resets are OSErrors; IncompleteRead and BadStatusLine are not.
        except (OSError, http.client.HTTPException) as e:
            raise TransportError(f"request to {url} failed: {e}") from e


def check_http_url(url: str) -> None:
    if not url.startswith(("https://", "http://")):
        raise ValidationError(f"only http(s) endpoints are supported, got {url!r}")


def check_credential_url(url: str) -> None:
    """Credentials need TLS unless they stay on this machine (localhost, 127.0.0.0/8, ::1)."""
    if url.startswith("https://"):
        return
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
    except ValueError as e:
        raise ValidationError(f"invalid endpoint URL: {e}") from e
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValidationError(
            f"refusing to send an API key over plain http to {host or 'an empty host'}; "
            "use https, a loopback address, or no key"
        )


def decode_body(raw: bytes) -> Any:
    try:
        return strict_json(raw.decode("utf-8"), max_chars=MAX_RESPONSE_BYTES) if raw else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")
    except ValueError as e:
        raise ProviderError(f"invalid provider JSON: {e}") from e


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded attempts with jittered exponential backoff.

    `max_retry_after_s` caps how long a server's `Retry-After` is waited out; a longer one
    fails at once. `jitter` is the randomized fraction of each backoff, so clients spread out.
    """

    attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    max_retry_after_s: float = 60.0
    jitter: float = 0.5

    def __post_init__(self) -> None:
        if (
            type(self.attempts) is not int
            or not 1 <= self.attempts <= 32
            or not all(math.isfinite(v) for v in (self.base_delay_s, self.max_delay_s, self.max_retry_after_s))
            or self.base_delay_s < 0
            or self.max_delay_s < self.base_delay_s
            or self.max_retry_after_s < 0
            or not 0 <= self.jitter <= 1
        ):
            raise ValidationError("retry policy out of range")

    def delay(self, attempt: int, retry_after: str | None = None) -> float:
        """The server's `Retry-After` up to its cap, otherwise jittered exponential backoff."""
        requested = _retry_after(retry_after)
        if requested is not None:
            return min(requested, self.max_retry_after_s)
        spread = self.jitter * random.random()  # noqa: S311 - timing jitter, not security
        return min(self.base_delay_s * 2.0**attempt, self.max_delay_s) * (1 - spread)


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Everything needed to call one URL repeatedly."""

    url: str
    headers: Mapping[str, str]
    timeout_s: float
    transport: Transport
    retry: RetryPolicy
    sleep: Callable[[float], None] = time.sleep
    idempotent: bool = False

    def __post_init__(self) -> None:
        if any(value and key.casefold() in CREDENTIAL_HEADERS for key, value in self.headers.items()):
            check_credential_url(self.url)

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
        *,
        idempotent: bool = False,
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
            idempotent,
        )

    def post(self, payload: Mapping[str, Any]) -> Any:
        """POST until a 200 comes back or the retry budget is spent. Returns the decoded body."""
        context = current_trace()
        if context:
            context.store.register_secrets(
                [
                    value.removeprefix("Bearer ")
                    for key, value in self.headers.items()
                    if key.casefold() in CREDENTIAL_HEADERS
                ]
            )
        with trace_span("provider_request", model=payload.get("model"), usage=provider_usage(None)) as details:
            return self._post(payload, details)

    def _post(self, payload: Mapping[str, Any], details: dict[str, Any]) -> Any:
        for attempt in range(self.retry.attempts):
            details["attempts"] = attempt + 1
            last = attempt + 1 == self.retry.attempts
            try:
                status, headers, body = self.transport.post_json(self.url, self.headers, payload, self.timeout_s)
            except TransportError:  # the provider may already have done the work
                if last or not self.idempotent:
                    raise
                self.sleep(self.retry.delay(attempt))
                continue
            details["http_status"] = status
            details["usage"] = provider_usage(body)
            if status == 200:
                return body
            if status != 429 and status < 500:
                raise ProviderError(f"{self.url}: HTTP {status}: {message(body)}")
            retry_after = header(headers, "retry-after")
            wait = retry_after_seconds(retry_after)
            # 429 and 503 mean the request was not processed; other server errors may follow billed work.
            if status not in (429, 503) and not self.idempotent:
                raise ProviderError(f"{self.url}: server error {status}, not retried", retry_after_s=wait)
            if wait > self.retry.max_retry_after_s:
                raise RateLimitedError(
                    f"{self.url}: HTTP {status} asks to wait {wait:.0f} s, "
                    f"longer than the {self.retry.max_retry_after_s:g} s retry limit",
                    retry_after_s=wait,
                )
            if not last:
                self.sleep(self.retry.delay(attempt, retry_after))
            elif status == 429:
                raise RateLimitedError(f"{self.url}: rate limited after {attempt + 1} attempts", retry_after_s=wait)
            else:
                raise ProviderError(
                    f"{self.url}: server error {status} after {attempt + 1} attempts", retry_after_s=wait
                )
        raise ProviderError("unreachable")  # pragma: no cover


def retry_after_seconds(value: str | None) -> float:
    """Preserve numeric/HTTP-date cooldowns instead of shortening them to the sleep budget."""
    return _retry_after(value) or 0


def _retry_after(value: str | None) -> float | None:
    """Seconds from a numeric or HTTP-date `Retry-After`; None when absent or invalid."""
    if not value:
        return None
    try:
        delay = float(value)
        return delay if math.isfinite(delay) and delay >= 0 else None
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            return max(0, date.timestamp() - time.time()) if date.tzinfo is not None else None
        except (ValueError, TypeError, OverflowError):
            return None


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
