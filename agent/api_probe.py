"""Shared, observable transport ceilings for the opt-in rescue probe.

This wrapper is not installed in the contest host. It performs no retries:
the diagnostic coordinator or v3 owns retry decisions. Reservations include
the complete serialized request and are never refunded for missing usage.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import time

from agent.gateways import official_gateway_name
from agent.llm import safe_error_details


class ProbeLimitReached(RuntimeError):
    pass


def timestamps():
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(), "msk": now.astimezone(timezone(timedelta(hours=3))).isoformat()}


def request_size(arguments):
    encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    output = arguments.get("max_tokens")
    if type(output) is not int or output <= 0:
        raise ValueError("A positive output token reservation is required")
    return {"request_bytes": len(encoded), "message_count": len(arguments.get("messages", [])),
            "estimated_tokens": math.ceil(len(encoded) / 3) + output, "max_tokens": output}


def reported_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    values = {key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return values if all(type(value) is int and value >= 0 for value in values.values()) else None


class ProbeBudget:
    def __init__(self, *, deadline, max_requests=6, max_tokens=30000, clock=time.monotonic, checkpoint=None):
        if not 1 <= max_requests <= 6 or not 1 <= max_tokens <= 30000:
            raise ValueError("Probe ceilings cannot exceed six requests or 30000 estimated tokens")
        self.deadline, self.max_requests, self.max_tokens = deadline, max_requests, max_tokens
        self.clock, self.checkpoint = clock, checkpoint or (lambda: None)
        self.attempts = []
        self.reserved = 0
        self.blocked = None
        self.stage = "text"

    def remaining(self):
        return max(0., self.deadline - self.clock())

    def reserve(self, arguments, route):
        size = request_size(arguments)
        reason = ("deadline" if self.remaining() <= 0 else
                  "request_limit" if len(self.attempts) >= self.max_requests else
                  "token_limit" if self.reserved + size["estimated_tokens"] > self.max_tokens else None)
        if reason:
            self.blocked = reason
            self.checkpoint()
            raise ProbeLimitReached("Shared probe ceiling reached")
        self.reserved += size["estimated_tokens"]
        row = {"attempt": len(self.attempts) + 1, "variant": self.stage,
               "route": route, "started_at": timestamps(), **size,
               "remaining_seconds": round(self.remaining(), 6), "result": "in_flight",
               "api_response_received": False, "usage": None, "duration_seconds": None}
        self.attempts.append(row)
        self.checkpoint()
        return row

    def summary(self):
        responses = [row for row in self.attempts if row["api_response_received"]]
        return {"attempts": len(self.attempts), "responses": len(responses),
                "max_requests": self.max_requests, "estimated_token_limit": self.max_tokens,
                "estimated_tokens_reserved": self.reserved, "blocked": self.blocked,
                "remaining_seconds": round(self.remaining(), 6),
                "responses_without_usage": sum(row["usage"] is None for row in responses),
                "usage_complete": bool(responses) and all(row["usage"] is not None for row in responses),
                "currency_cost": None,
                "estimate_method": "ceil(complete UTF-8 request JSON bytes / 3) + max_tokens"}


class ProbeClient:
    def __init__(self, client, budget):
        self.client, self.budget = client, budget
        self.chat = self.completions = self

    @property
    def base_url(self):
        return self.client.base_url

    def with_options(self, **kwargs):
        return ProbeClient(self.client.with_options(**kwargs), self.budget)

    def create(self, **kwargs):
        route = official_gateway_name(str(self.base_url)) or "custom"
        row = self.budget.reserve(kwargs, route)
        started = self.budget.clock()
        try:
            # v3 has already applied its original request/time/token guards.
            # The shared ceiling is an additional probe-only guard.
            response = self.client.chat.completions.create(**kwargs)
            row.update(result="response", api_response_received=True, usage=reported_usage(response))
            return response
        except Exception as error:
            row.update(result="transport_error", error=safe_error_details(error))
            raise
        finally:
            row.update(duration_seconds=round(max(0., self.budget.clock() - started), 6), finished_at=timestamps())
            self.budget.checkpoint()
