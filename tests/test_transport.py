from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from placecell.errors import ProviderError, ValidationError
from placecell.providers.openai_compatible import UrllibTransport


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
