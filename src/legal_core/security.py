"""Authentication and lightweight rate limiting for the QA endpoint."""
from __future__ import annotations

import hmac
import os
import time
from collections import defaultdict, deque
from threading import Lock

API_KEY_ENV = "PAKISTANI_LAWYER_API_KEY"
RATE_LIMIT_ENV = "QA_RATE_LIMIT_PER_MINUTE"
TRUST_PROXY_ENV = "TRUST_PROXY"
DEFAULT_RATE_LIMIT = 30


def resolve_api_key(api_key: str | None = None, *, env=None) -> str:
    if api_key is not None:
        value = api_key.strip()
    else:
        env = os.environ if env is None else env
        value = (env.get(API_KEY_ENV) or "").strip()
    if not value:
        raise ValueError(
            f"API key is required: set {API_KEY_ENV} (there is no built-in default)"
        )
    return value


def resolve_rate_limit(limit: int | None = None, *, env=None) -> int:
    if limit is not None:
        value = limit
    else:
        env = os.environ if env is None else env
        raw = env.get(RATE_LIMIT_ENV)
        if raw is None or not raw.strip():
            return DEFAULT_RATE_LIMIT
        try:
            value = int(raw)
        except ValueError:
            raise ValueError("QA_RATE_LIMIT_PER_MINUTE must be a positive integer") from None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("rate limit must be a positive integer")
    return value


def api_key_matches(provided: str | None, expected: str) -> bool:
    return bool(provided) and hmac.compare_digest(provided, expected)


def client_identity(request, *, trust_proxy: bool | None = None, env=None) -> str:
    """Return the client IP, honouring X-Forwarded-For only when explicitly trusted."""
    if trust_proxy is None:
        env = os.environ if env is None else env
        trust_proxy = (env.get(TRUST_PROXY_ENV) or "").strip().lower() in {"1", "true", "yes"}
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window limiter keyed by client identity."""

    _SWEEP_EVERY = 256

    def __init__(self, limit: int, window_seconds: float = 60.0, *, clock=None):
        self.limit = resolve_rate_limit(limit)
        self.window_seconds = float(window_seconds)
        self.clock = clock or time.monotonic
        self._hits = defaultdict(deque)
        self._lock = Lock()
        self._calls = 0

    def _sweep(self, cutoff: float) -> None:
        for key in [k for k, b in self._hits.items() if not b or b[-1] <= cutoff]:
            del self._hits[key]

    def allow(self, key: str) -> bool:
        now = self.clock()
        cutoff = now - self.window_seconds
        with self._lock:
            self._calls += 1
            if self._calls % self._SWEEP_EVERY == 0:
                self._sweep(cutoff)
            bucket = self._hits[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


# Backwards-compatible alias (the old name was misleading).
FixedWindowRateLimiter = SlidingWindowRateLimiter
