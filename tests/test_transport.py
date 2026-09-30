from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from placecell.errors import ProviderError, RateLimitedError, ValidationError
from placecell.providers import GeminiEmbedder, OpenAICompatibleChat
from placecell.providers._http import Endpoint, RetryPolicy, TransportError, reported_usage
from placecell.providers.openai_compatible import UrllibTransport
from placecell.tracing import TraceStore, read_trace, trace_scope


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/ok":
            payload = json.dumps({"echo": body, "auth": self.headers.get("Authorization")}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Test", "yes")
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/limited":
            self.send_response(429)
            self.send_header("Retry-After", "3")
            self.end_headers()
            self.wfile.write(b"slow down")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture
def server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_transport_posts_json_and_returns_status_headers_body(server: str) -> None:
    status, headers, body = UrllibTransport().post_json(
        f"{server}/ok", {"Authorization": "Bearer k"}, {"input": ["a"]}, 5
    )
    assert status == 200
    assert body == {"echo": {"input": ["a"]}, "auth": "Bearer k"}
    assert {k.lower(): v for k, v in headers.items()}["x-test"] == "yes"


def test_transport_returns_http_errors_instead_of_raising(server: str) -> None:
    status, headers, body = UrllibTransport().post_json(f"{server}/limited", {}, {}, 5)
    assert status == 429 and body == "slow down" and headers.get("Retry-After") == "3"
    status, _, body = UrllibTransport().post_json(f"{server}/missing", {}, {}, 5)
    assert status == 404 and body is None


def test_transport_rejects_other_schemes_and_reports_connection_failures() -> None:
    with pytest.raises(ValidationError):
        UrllibTransport().post_json("file:///etc/hostname", {}, {}, 1)
    probe = HTTPServer(("127.0.0.1", 0), _Handler)
    port = probe.server_port
    probe.server_close()  # nothing listens on this port any more
    with pytest.raises(ProviderError):
        UrllibTransport().post_json(f"http://127.0.0.1:{port}/ok", {}, {}, 2)


class _Redirect(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.send_response(int(self.path.strip("/")))
        self.send_header("Location", self.server.target)  # type: ignore[attr-defined]
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


class _Record(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.server.seen.append(dict(self.headers.items()))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self) -> None:
        self.do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture
def redirect() -> Iterator[tuple[str, list[dict[str, str]]]]:
    """A redirecting server whose target is a second, recording server."""
    target = HTTPServer(("127.0.0.1", 0), _Record)
    target.seen = []  # type: ignore[attr-defined]
    source = HTTPServer(("127.0.0.1", 0), _Redirect)
    source.target = f"http://localhost:{target.server_port}/stolen"  # type: ignore[attr-defined]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (source, target)]
    for thread in threads:
        thread.start()
    try:
        yield f"http://127.0.0.1:{source.server_port}", target.seen  # type: ignore[attr-defined]
    finally:
        for s in (source, target):
            s.shutdown()
            s.server_close()
        for thread in threads:
            thread.join(timeout=5)


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_transport_refuses_redirects_so_credentials_stay_on_the_configured_host(redirect, code) -> None:
    url, seen = redirect
    headers = {"Authorization": "Bearer secret-value", "X-Goog-Api-Key": "secret-value"}
    with pytest.raises(ProviderError, match=f"refused HTTP {code} redirect to http://localhost") as error:
        UrllibTransport().post_json(f"{url}/{code}", headers, {"input": ["a"]}, 5)
    assert "secret-value" not in str(error.value)
    endpoint = Endpoint.build(url, f"/{code}", "secret-value", 5, None, None, lambda _: None, None)
    with pytest.raises(ProviderError, match="refused"):
        endpoint.post({"input": ["a"]})
    assert seen == []


@pytest.mark.parametrize(
    "build",
    [
        lambda url: OpenAICompatibleChat("m", url, "key"),
        lambda url: OpenAICompatibleChat("m", url, extra_headers={"X-Api-Key": "key"}),
        lambda url: GeminiEmbedder(api_key="key", base_url=url),
    ],
)
@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.5:8000/v1",
        "http://api.example.com/v1",
        "http://localhost.example.com/v1",
        "http://127.0.0.1.example.com/v1",
        "http://[::2]/v1",
        "http://",
    ],
)
def test_credentials_are_refused_over_plain_http_to_other_hosts(build, url) -> None:
    with pytest.raises(ValidationError, match="plain http"):
        build(url)


def test_plain_http_is_allowed_without_credentials_or_on_loopback() -> None:
    for url in ("http://10.0.0.5:8000/v1", "http://api.example.com/v1"):
        assert OpenAICompatibleChat("m", url).model_name == "m"
        assert OpenAICompatibleChat("m", url, "", extra_headers={"X-Api-Key": ""}).model_name == "m"
    for url in ("http://localhost:8000/v1", "http://LOCALHOST/v1", "http://127.0.0.2/v1", "http://[::1]:8000/v1"):
        assert OpenAICompatibleChat("m", url, "key").model_name == "m"
    assert OpenAICompatibleChat("m", "https://10.0.0.5:8000/v1", "key").model_name == "m"
    with pytest.raises(ValidationError, match="invalid endpoint URL"):
        OpenAICompatibleChat("m", "http://[::1/v1", "key")


class _Broken(BaseHTTPRequestHandler):
    """Replies that make http.client raise its own, non-OSError exceptions."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/status-line":
            self.wfile.write(b"garbage\r\n\r\n")
            return
        self.send_response(200 if self.path == "/short-body" else 503)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.write(b'64\r\n{"a":')  # announces 100 bytes, then the connection closes
        self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.mark.parametrize("path", ["/status-line", "/short-body", "/short-error-body"])
def test_http_client_failures_become_provider_errors(path) -> None:
    httpd = HTTPServer(("127.0.0.1", 0), _Broken)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(TransportError, match="failed"):
            UrllibTransport().post_json(f"http://127.0.0.1:{httpd.server_port}{path}", {}, {"input": ["a"]}, 5)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


class _Scripted:
    """Responses or exceptions, in order."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes, self.calls = list(outcomes), 0

    def post_json(self, url, headers, payload, timeout_s):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _endpoint(transport: Any, sleeps: list[float], *, idempotent: bool, **policy: Any) -> Endpoint:
    retry = RetryPolicy(**{"jitter": 0, **policy})
    return Endpoint.build(
        "https://x.test", "/v1", None, 5, transport, retry, sleeps.append, None, idempotent=idempotent
    )


@pytest.mark.parametrize("status", [500, 502, 504])
def test_server_errors_that_may_follow_billed_work_repeat_only_idempotent_calls(status) -> None:
    sleeps: list[float] = []
    transport = _Scripted((status, {}, {}), (200, {}, {"ok": True}))
    with pytest.raises(ProviderError, match=f"server error {status}, not retried") as error:
        _endpoint(transport, sleeps, idempotent=False).post({})
    assert transport.calls == 1 and not sleeps and not isinstance(error.value, RateLimitedError)
    transport = _Scripted((status, {}, {}), (200, {}, {"ok": True}))
    assert _endpoint(transport, sleeps, idempotent=True).post({}) == {"ok": True}
    assert transport.calls == 2 and sleeps == [0.5]


@pytest.mark.parametrize("idempotent", [False, True])
@pytest.mark.parametrize("status", [429, 503])
def test_unprocessed_requests_are_repeated_for_every_call(status, idempotent) -> None:
    sleeps: list[float] = []
    transport = _Scripted((status, {}, {}), (status, {"Retry-After": "30"}, {}), (200, {}, {"ok": True}))
    assert _endpoint(transport, sleeps, idempotent=idempotent).post({}) == {"ok": True}
    assert transport.calls == 3 and sleeps == [0.5, 30.0]


def test_transport_failures_are_repeated_only_for_idempotent_calls() -> None:
    sleeps: list[float] = []
    reset = TransportError("request to https://x.test/v1 failed: connection reset")
    transport = _Scripted(reset, (200, {}, {"ok": True}))
    with pytest.raises(TransportError):
        _endpoint(transport, sleeps, idempotent=False).post({})
    assert transport.calls == 1 and not sleeps
    transport = _Scripted(reset, (200, {}, {"ok": True}))
    assert _endpoint(transport, sleeps, idempotent=True).post({}) == {"ok": True}
    transport = _Scripted(reset, reset)
    with pytest.raises(TransportError):
        _endpoint(transport, sleeps, idempotent=True, attempts=2).post({})
    assert transport.calls == 2 and sleeps == [0.5, 0.5]


@pytest.mark.parametrize("idempotent", [False, True])
@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_beyond_the_cap_fails_at_once_without_hammering(status, idempotent) -> None:
    sleeps: list[float] = []
    for header, cap in (("61", 60.0), ("6", 5.0)):
        transport = _Scripted((status, {"Retry-After": header}, {}), (200, {}, {}))
        with pytest.raises(RateLimitedError, match="retry limit") as error:
            _endpoint(transport, sleeps, idempotent=idempotent, max_retry_after_s=cap).post({})
        assert error.value.retry_after_s == float(header) and transport.calls == 1 and not sleeps


def test_computed_backoff_is_jittered_below_its_bound() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=8.0, jitter=0.5)
    delays = [policy.delay(2) for _ in range(200)]
    assert all(2.0 <= d <= 4.0 for d in delays) and len(set(delays)) > 1
    assert all(4.0 <= policy.delay(5) <= 8.0 for _ in range(20))
    assert RetryPolicy(jitter=0.5).delay(0, "3") == 3.0  # a server-requested wait is never shortened


def test_provider_spans_name_the_model_when_the_payload_does_not(tmp_path) -> None:
    traces = TraceStore(tmp_path / "traces.sqlite3")
    body = {"embeddings": [{"values": [1.0] + [0.0] * 767}], "usageMetadata": {"totalTokenCount": 3}}
    embedder = GeminiEmbedder(api_key="key", transport=_Scripted((200, {}, body)))
    with trace_scope(traces.context("mission", "request")):
        embedder.embed_text(["printer"])
    assert traces.flush()
    span = next(e for e in read_trace(traces.path)["events"] if e["kind"] == "end")
    assert span["data"]["model"] == "gemini-embedding-2" and span["data"]["usage"]["total_tokens"] == 3
    assert traces.close()


@pytest.mark.parametrize(
    "url,cost",
    [
        ("https://openrouter.ai/api/v1/chat/completions", 0.0004),
        ("https://eu.openrouter.ai/api/v1/embeddings", 0.0004),
        ("https://api.example.com/v1/chat/completions", None),
        ("https://openrouter.ai.example.com/v1", None),
    ],
)
def test_openrouter_cost_is_read_as_dollars_and_other_costs_stay_unknown(url, cost) -> None:
    usage = reported_usage(url, {"usage": {"prompt_tokens": 3, "cost": 0.0004}})
    assert usage["cost_usd"] == cost and usage["input_tokens"] == 3
    assert reported_usage(url, {"usage": {"cost": 0.1, "cost_usd": 0.2}})["cost_usd"] == 0.2
    assert reported_usage(url, {"usage": {"cost": "free"}})["cost_usd"] is None
    assert reported_usage("http://[::1/v1", {"usage": {"cost": 0.1}})["cost_usd"] is None
