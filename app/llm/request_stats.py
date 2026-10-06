"""What the LLM calls of one request went through: which backend served each, how many 429s were seen,
how many calls fell back to Ollama.

The client records into the stats object of the request it runs under (a context variable, so the two parallel
paths of a MIXED question, which copy the context, share one object). Outside a request (eval runs, scripts)
nothing is recording and every call here does nothing.
"""

from __future__ import annotations

import contextvars
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager


class LLMRequestStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._served: Counter[str] = Counter()
        self._fallback_calls = 0
        self._rate_limited = 0

    def served(self, backend: str, *, fallback: bool = False) -> None:
        with self._lock:
            self._served[backend] += 1
            if fallback:
                self._fallback_calls += 1

    def rate_limited(self) -> None:
        with self._lock:
            self._rate_limited += 1

    def summary(self) -> dict:
        with self._lock:
            return {
                "calls": sum(self._served.values()),
                "backends": sorted(self._served),
                "fallback_calls": self._fallback_calls,
                "rate_limited": self._rate_limited,
            }


_current: contextvars.ContextVar[LLMRequestStats | None] = contextvars.ContextVar(
    "llm_request_stats", default=None
)


@contextmanager
def track() -> Iterator[LLMRequestStats]:
    stats = LLMRequestStats()
    token = _current.set(stats)
    try:
        yield stats
    finally:
        _current.reset(token)


def record_served(backend: str, *, fallback: bool = False) -> None:
    stats = _current.get()
    if stats is not None:
        stats.served(backend, fallback=fallback)


def record_rate_limited() -> None:
    stats = _current.get()
    if stats is not None:
        stats.rate_limited()
