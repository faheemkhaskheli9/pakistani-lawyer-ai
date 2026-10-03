"""Authentication and lightweight rate limiting for the QA endpoint."""
from __future__ import annotations

import hmac
import os
import time
from collections import defaultdict, deque
from threading import Lock

API_KEY_ENV = "PAKISTANI_LAWYER_API_KEY"
RATE_LIMIT_ENV = "QA_RATE_LIMIT_PER_MINUTE"
DEFAULT_DEV_API_KEY = "local-development-key"
DEFAULT_RATE_LIMIT = 30
SEARCH_RATE_LIMIT_ENV = "SEARCH_RATE_LIMIT_PER_MINUTE"
DEFAULT_SEARCH_RATE_LIMIT = 120
ENVIRONMENT_ENV = "PAKISTANI_LAWYER_ENV"
TRUSTED_PROXIES_ENV = "TRUSTED_PROXY_COUNT"


def is_production(*, env=None) -> bool:
    env = os.environ if env is None else env
    return env.get(ENVIRONMENT_ENV, "").strip().lower() in {"prod", "production"}


def resolve_api_key(api_key: str | None = None, *, env=None) -> str:
    if api_key is not None:
        value = api_key.strip()
    else:
        env = os.environ if env is None else env
        value = env.get(API_KEY_ENV, DEFAULT_DEV_API_KEY).strip()
    if not value:
        raise ValueError("API key must not be empty")
    if value == DEFAULT_DEV_API_KEY and is_production(env=env):
        raise ValueError(
            f"Refusing to start in production with the default development API key; set ${API_KEY_ENV}"
        )
    return value


def resolve_rate_limit(
    limit: int | None = None, *, env=None, env_var: str = RATE_LIMIT_ENV, default: int = DEFAULT_RATE_LIMIT
) -> int:
    if limit is not None:
        value = limit
    else:
        env = os.environ if env is None else env
        raw = env.get(env_var)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{env_var} must be a positive integer") from None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("rate limit must be a positive integer")
    return value


def resolve_trusted_proxies(count: int | None = None, *, env=None) -> int:
    if count is None:
        env = os.environ if env is None else env
        raw = env.get(TRUSTED_PROXIES_ENV, "").strip()
        if not raw:
            return 0
        try:
            count = int(raw)
        except ValueError:
            raise ValueError(f"{TRUSTED_PROXIES_ENV} must be a non-negative integer") from None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("trusted proxy count must be a non-negative integer")
    return count


def client_identity(peer: str | None, forwarded_for: str | None, trusted_proxies: int = 0) -> str:
    """Return the client address used as the rate-limit key.

    `X-Forwarded-For` is attacker-controlled except for the entries appended
    by proxies we operate, so with N trusted proxies only the Nth entry from
    the right is believed. With none configured the header is ignored.
    """
    if trusted_proxies > 0 and forwarded_for:
        hops = [hop.strip() for hop in forwarded_for.split(",") if hop.strip()]
        if len(hops) >= trusted_proxies:
            return hops[-trusted_proxies]
    return peer or "unknown"


def api_key_matches(provided: str | None, expected: str) -> bool:
    return bool(provided) and hmac.compare_digest(provided, expected)


class FixedWindowRateLimiter:
    """Thread-safe in-memory sliding-window limiter keyed by client identity."""

    def __init__(self, limit: int, window_seconds: float = 60.0, *, clock=None):
        self.limit = resolve_rate_limit(limit)
        self.window_seconds = float(window_seconds)
        self.clock = clock or time.monotonic
        self._hits = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True
