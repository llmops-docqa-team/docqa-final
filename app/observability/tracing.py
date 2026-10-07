"""Langfuse tracing (design §15, nice-to-have #1): a span tree per request; 👍/👎 and judge scores on it.

    query                      (trace root)
    ├─ router                  generation: model, prompt version, token usage, route
    ├─ document                span: status, abstain reason, top score
    │  ├─ retrieve             span: retrieved chunk ids and scores
    │  └─ generate             generation: model, prompt version, token usage, backend
    └─ general
       └─ generate

Optional and off by default. It is on only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are both set
(LANGFUSE_BASE_URL picks the server; LANGFUSE_TRACING_ENABLED=false switches it off again) and the `langfuse`
package is installed. Whatever goes wrong, a request is never blocked or failed by it: every call into the SDK
is wrapped, and the SDK itself exports in a background thread.

Content policy: spans carry ids, versions, models, token counts, scores and timings. The question, the answer
and the cited snippets are `input` / `output` and are dropped unless `tracing.capture_content` is on; whole
chunks of an uploaded document are never sent.

The tracer is a process-wide object that is a no-op until `set_tracer` is called (the API's startup and the
judge script do), so tests and the eval tools stay untraced.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from app.observability.logging import get_logger

_CONTENT_KEYS = ("input", "output")


class Span:
    """What a traced block gets. This base class is the no-op version, so callers never test for None."""

    def update(self, **fields: Any) -> None:
        """Add `output`, `metadata`, `usage_details`, `model`, `level`, ... to the span. Never raises."""


_NULL_SPAN = Span()


class Tracer:
    """The no-op tracer: every block runs, nothing is recorded."""

    enabled = False
    capture_content = False

    @contextmanager
    def trace(self, name: str, trace_id: str, **fields: Any) -> Iterator[Span]:
        yield _NULL_SPAN

    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[Span]:
        yield _NULL_SPAN

    @contextmanager
    def generation(self, name: str, **fields: Any) -> Iterator[Span]:
        yield _NULL_SPAN

    def score(
        self, trace_id: str, name: str, value: float, *, boolean: bool = False, comment: str | None = None
    ) -> None:
        """Attach a score to the trace of request `trace_id`. Never raises."""

    def flush(self) -> None:
        """Send what is queued (call before a short-lived process exits). Never raises."""


class _LangfuseSpan(Span):
    def __init__(self, observation: Any, tracer: LangfuseTracer):
        self._obs = observation
        self._tracer = tracer

    def update(self, **fields: Any) -> None:
        try:
            self._obs.update(**self._tracer.clean(fields))
        except Exception as exc:  # noqa: BLE001  tracing must never fail a request
            self._tracer.failed("update", exc)


class LangfuseTracer(Tracer):
    enabled = True

    def __init__(self, client: Any, *, capture_content: bool = False):
        self._client = client
        self.capture_content = capture_content

    def failed(self, what: str, exc: Exception) -> None:
        get_logger().warning(
            "tracing_failed", what=what, error_type=type(exc).__name__, detail=str(exc)[:200]
        )

    def clean(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in fields.items() if v is not None}
        if not self.capture_content:
            for key in _CONTENT_KEYS:
                out.pop(key, None)
        return out

    def _lf_trace_id(self, trace_id: str) -> str:
        # Langfuse trace ids are 32 hex characters. Seeding keeps the mapping stable, so a 👍 or a judge score
        # that arrives later (or from another process) lands on the trace of the same request.
        return self._client.create_trace_id(seed=trace_id)

    @contextmanager
    def _observe(
        self, as_type: str, name: str, trace_id: str | None, fields: Mapping[str, Any]
    ) -> Iterator[Span]:
        try:
            kwargs = self.clean(fields)
            if trace_id is not None:
                kwargs["trace_context"] = {"trace_id": self._lf_trace_id(trace_id)}
            cm = self._client.start_as_current_observation(as_type=as_type, name=name, **kwargs)
            observation = cm.__enter__()
        except Exception as exc:  # noqa: BLE001
            self.failed(f"start {name}", exc)
            yield _NULL_SPAN
            return
        try:
            yield _LangfuseSpan(observation, self)
        except BaseException as exc:
            try:
                cm.__exit__(type(exc), exc, exc.__traceback__)
            except Exception as exit_exc:  # noqa: BLE001
                self.failed(f"end {name}", exit_exc)
            raise
        else:
            try:
                cm.__exit__(None, None, None)
            except Exception as exit_exc:  # noqa: BLE001
                self.failed(f"end {name}", exit_exc)

    def trace(self, name: str, trace_id: str, **fields: Any):
        return self._observe("span", name, trace_id, fields)

    def span(self, name: str, **fields: Any):
        return self._observe("span", name, None, fields)

    def generation(self, name: str, **fields: Any):
        return self._observe("generation", name, None, fields)

    def score(
        self, trace_id: str, name: str, value: float, *, boolean: bool = False, comment: str | None = None
    ) -> None:
        try:
            self._client.create_score(
                trace_id=self._lf_trace_id(trace_id),
                name=name,
                value=float(value),
                data_type="BOOLEAN" if boolean else "NUMERIC",
                comment=comment,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed("score", exc)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001
            self.failed("flush", exc)


def _falsey(v: str | None) -> bool:
    return v is not None and v.strip().lower() in ("0", "false", "no", "off")


def build_tracer(
    *, capture_content: bool = False, release: str | None = None, env: Mapping[str, str] | None = None
) -> Tracer:
    """A `LangfuseTracer` when the keys are set and the SDK works, else the no-op `Tracer`. Never raises."""
    env = os.environ if env is None else env
    log = get_logger()
    public, secret = env.get("LANGFUSE_PUBLIC_KEY"), env.get("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        return Tracer()
    if _falsey(env.get("LANGFUSE_TRACING_ENABLED")):
        log.info("tracing_off", reason="LANGFUSE_TRACING_ENABLED is false")
        return Tracer()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public,
            secret_key=secret,
            base_url=env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST") or None,
            release=release,
            timeout=10,
        )
    except ImportError:
        log.warning("tracing_off", reason="langfuse is not installed")
        return Tracer()
    except Exception as exc:  # noqa: BLE001
        log.warning("tracing_off", reason="langfuse failed to start", error_type=type(exc).__name__)
        return Tracer()
    log.info("tracing_on", capture_content=capture_content)
    return LangfuseTracer(client, capture_content=capture_content)


def usage_details(prompt: int, completion: int, total: int) -> dict[str, int]:
    """Token counts in the shape Langfuse's `usage_details` wants."""
    return {"input": prompt, "output": completion, "total": total}


_tracer: Tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    global _tracer
    _tracer = tracer
