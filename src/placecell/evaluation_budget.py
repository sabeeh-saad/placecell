"""Sequential, metered OpenRouter calls for explicitly budgeted evaluation runs."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeGuard

from placecell.errors import ProviderError, ValidationError
from placecell.providers._http import Transport, UrllibTransport

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
# USD per million tokens, USD per request/image. No server-side tools or model fallback.
PRICE_LIMITS = {"prompt": 1, "completion": 5, "request": 0, "image": 0.01}


class EvaluationStoppedError(ProviderError):
    """No further requests may be issued under this run's budget."""


def valid_cost(value: Any) -> TypeGuard[int | float]:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class EvaluationBudget:
    """One in-flight call, a flushed reservation before send, and fail-closed accounting.

    The reservation uses a conservative byte/token envelope plus image allowance and
    routing price ceilings. It is not a provider billing guarantee. Unknown billing or
    a price-envelope breach stops the run; neither is treated as zero spend.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_requests: int,
        max_usd: float,
        max_seconds: float,
        transport: Transport | None = None,
    ) -> None:
        if (
            type(max_requests) is not int
            or not 1 <= max_requests <= 10000
            or not valid_cost(max_usd)
            or max_usd <= 0
            or not valid_cost(max_seconds)
            or max_seconds <= 0
        ):
            raise ValidationError("evaluation needs positive request, cost and wall-time limits")
        self.max_requests, self.max_usd = max_requests, max_usd
        self.deadline = time.monotonic() + max_seconds
        self.transport = transport or UrllibTransport()
        self.path = path
        # Never resume/reset a prior ledger implicitly, including after interruption.
        with path.open("x"):
            pass
        self.lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self.known_cost = 0.0
        self.charged_or_reserved = 0.0
        self.stop_reason = ""
        self.stage = ""
        self.trial_id = ""

    def _append(self, row: dict[str, Any]) -> None:
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def prepare(payload: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        allowed = {"model", "temperature", "max_tokens", "messages", "tools", "tool_choice", "response_format"}
        if payload.keys() - allowed:
            raise ValidationError("unsupported evaluation request fields")
        tokens = payload.get("max_tokens")
        if type(tokens) is not int or not 1 <= tokens <= 2048:
            raise ValidationError("evaluation completions must be bounded to 1..2048 tokens")
        # Count text separately from image base64; bound both before transmission.
        copy = json.loads(json.dumps(payload, allow_nan=False))
        images = 0
        for message in copy.get("messages", []):
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if (
                            not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,"))
                            or len(url) > 2_000_000
                        ):
                            raise ValidationError("evaluation requires bounded inline PNG/JPEG images")
                        images += 1
                        part["image_url"]["url"] = "inline image"
        text_bytes = len(json.dumps(copy, ensure_ascii=False).encode())
        if text_bytes > 32768 or images > 2:
            raise ValidationError("evaluation request exceeds text/image bounds")
        # Byte-per-token allowance, serialization overhead, and 32768 tokens/image.
        reserve = (text_bytes + 16384 + images * 32768) / 1_000_000 + tokens * 5 / 1_000_000 + images * 0.01
        outgoing = {**payload, "provider": {"max_price": PRICE_LIMITS, "require_parameters": True}}
        return outgoing, reserve

    def summary(self) -> dict[str, Any]:
        return {
            "requests": len(self.records),
            "max_requests": self.max_requests,
            "max_usd": self.max_usd,
            "known_cost_usd": self.known_cost,
            "charged_or_reserved_usd": self.charged_or_reserved,
            "unknown_cost_requests": sum(r.get("cost_usd") is None for r in self.records),
            "stop_reason": self.stop_reason,
        }

    def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, str], Any]:
        if url != ENDPOINT:
            raise ValidationError("evaluation credentials may only be sent to the OpenRouter chat endpoint")
        outgoing, reserve = self.prepare(payload)
        with self.lock:
            if not self.stop_reason:
                if len(self.records) >= self.max_requests:
                    self.stop_reason = "request_limit"
                elif time.monotonic() >= self.deadline:
                    self.stop_reason = "wall_time_limit"
                elif self.charged_or_reserved + reserve > self.max_usd:
                    self.stop_reason = "insufficient_remaining_budget_for_reservation"
            if self.stop_reason:
                raise EvaluationStoppedError(self.stop_reason)
            row: dict[str, Any] = {
                "sequence": len(self.records) + 1,
                "stage": self.stage,
                "trial_id": self.trial_id,
                "model": payload.get("model"),
                "reserved_usd": reserve,
                "cost_usd": None,
                "started_unix_s": time.time(),
            }
            self.records.append(row)
            self.charged_or_reserved += reserve
            started = time.monotonic()
            try:
                self._append({**row, "event": "reserved"})
                status, response_headers, body = self.transport.post_json(
                    url, headers, outgoing, min(timeout_s, max(0.001, self.deadline - time.monotonic()))
                )
                row["http_status"] = status
                usage = body.get("usage", {}) if isinstance(body, dict) else {}
                usage = usage if isinstance(usage, dict) else {}
                cost = usage.get("cost")
                if valid_cost(cost):
                    row["cost_usd"] = float(cost)
                    self.known_cost += cost
                    self.charged_or_reserved += cost - reserve
                    if cost > reserve:
                        self.stop_reason = "provider_exceeded_reservation"
                else:
                    self.stop_reason = "unknown_cost"
                if status != 200:
                    self.stop_reason = f"http_{status}"
                # Retain billing/model provenance only. Never log headers, prompts or raw errors.
                for name in ("id", "model", "provider"):
                    value = body.get(name) if isinstance(body, dict) else None
                    if isinstance(value, str) and len(value) <= 200 and "sk-" not in value:
                        row[f"response_{name}"] = value
                row["tokens"] = {
                    key: usage[key]
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int and usage[key] >= 0
                }
                return status, response_headers, body
            except BaseException as error:
                self.stop_reason = self.stop_reason or "transport_or_accounting_error"
                row["error_type"] = type(error).__name__
                raise
            finally:
                row["elapsed_s"] = time.monotonic() - started
                row["stop_reason"] = self.stop_reason
                try:
                    self._append({**row, "event": "completed"})
                except BaseException:
                    self.stop_reason = "accounting_write_error"
                    raise
