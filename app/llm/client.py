"""One OpenAI-SDK client for every LLM call (Groq by default, Ollama or any OpenAI-compatible server).

- Timeouts and a small retry loop (429 / 5xx / timeout / connection errors) with exponential backoff.
- Token usage is returned to the caller, so each stage can log it.
- A dev/eval-only disk cache keyed by hash(model, messages, params). It is OFF unless `FINCHAT_LLM_CACHE=1`
  (or `llm.dev_cache: true`) and is forced off when `FINCHAT_ENV=prod`.
- Every failure the caller should treat as "service unavailable" is raised as `LLMError`.
- Optional fallback (`FallbackBackend`, built from config by `llm_client_from_settings`): when the primary
  backend is unavailable for good (timeouts, 429s, 5xx or connection errors, after the retries above) the call
  is repeated once on a local Ollama server, and the primary is skipped for a cooldown so the next requests do
  not each pay for the retries again. Which backend served a call is on `LLMResponse.backend` and in the
  request stats.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import openai

from app.config import Settings
from app.llm import request_stats
from app.observability.logging import get_logger

Message = dict[str, str]


class LLMError(Exception):
    """The LLM call failed for good (after retries). `status` is the HTTP status when there was one."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        # True when the backend itself was unavailable (timeout, 429, 5xx, connection error), as opposed to a
        # request it rejected (400, 401, ...). Only these are worth repeating on another backend.
        self.retryable = retryable


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class LLMResponse:
    content: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    cached: bool = False
    backend: str = ""  # which backend served the call ("" = replayed from the dev cache)


class DiskCache:
    """JSON files under `root`, one per request hash. Only successful responses are stored."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def key(model: str, messages: Sequence[Message], params: dict[str, Any]) -> str:
        payload = json.dumps({"model": model, "messages": list(messages), "params": params}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            return LLMResponse(
                content=data["content"],
                usage=Usage(**data["usage"]),
                model=data.get("model", ""),
                finish_reason=data.get("finish_reason"),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, key: str, resp: LLMResponse) -> None:
        path = self._path(key)
        data = {
            "content": resp.content,
            "usage": asdict(resp.usage),
            "model": resp.model,
            "finish_reason": resp.finish_reason,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass  # a cache that cannot write is just a cache miss


def cache_enabled(dev_cache: bool, env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    if env.get("FINCHAT_ENV", "").lower() in ("prod", "production"):
        return False
    flag = env.get("FINCHAT_LLM_CACHE", "")
    if flag:
        return flag.lower() in ("1", "true", "yes", "on")
    return dev_cache


@dataclass
class FallbackBackend:
    """Where a call goes when the primary backend is down: an OpenAI-compatible server and its model names."""

    name: str
    base_url: str
    answer_model: str
    router_model: str
    timeout_seconds: float = 120.0
    cooldown_seconds: float = 30.0

    def model_for(self, primary_model: str, primary_router_model: str) -> str:
        return self.router_model if primary_model == primary_router_model else self.answer_model


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    raw = response.headers.get("retry-after") if response is not None else None
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        cache: DiskCache | None = None,
        sdk_client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        backend: str | None = None,
        fallback: FallbackBackend | None = None,
        fallback_sdk_client: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = settings.llm
        self._api_key = settings.groq_api_key
        self._sdk = sdk_client
        self.cache = cache
        self._sleep = sleep
        self.backend = backend or settings.llm.backend
        self._fallback = fallback
        self._fallback_sdk = fallback_sdk_client
        self._clock = clock
        self._primary_down_until = 0.0  # while clock() < this, calls go straight to the fallback

    @property
    def sdk(self) -> Any:
        if self._sdk is None:
            # Ollama and other local servers ignore the key; Groq rejects a wrong one with a 401.
            self._sdk = openai.OpenAI(
                base_url=self.cfg.base_url,
                api_key=self._api_key or "not-set",
                max_retries=0,
                timeout=self.cfg.timeout_seconds,
            )
        return self._sdk

    @property
    def fallback_sdk(self) -> Any:
        if self._fallback_sdk is None and self._fallback is not None:
            self._fallback_sdk = openai.OpenAI(
                base_url=self._fallback.base_url,
                api_key="ollama",  # Ollama ignores the key, the SDK insists on one
                max_retries=0,
                timeout=self._fallback.timeout_seconds,
            )
        return self._fallback_sdk

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        model = model or self.cfg.answer_model
        params: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        if json_mode:
            params["response_format"] = {"type": "json_object"}

        key = None
        if self.cache is not None:
            key = DiskCache.key(model, messages, params)
            hit = self.cache.get(key)
            if hit is not None:
                hit.cached = True
                return hit

        resp = self._dispatch(list(messages), model, params, timeout or self.cfg.timeout_seconds)
        request_stats.record_served(resp.backend, fallback=resp.backend != self.backend)
        # Only the primary's answers are cached: a local model's answer must not be replayed as Groq's.
        if self.cache is not None and key is not None and resp.backend == self.backend:
            self.cache.put(key, resp)
        return resp

    def _dispatch(
        self, messages: list[Message], model: str, params: dict[str, Any], timeout: float
    ) -> LLMResponse:
        """The primary backend, then (when configured) the fallback if the primary is unavailable."""
        fb = self._fallback
        if fb is None:
            return self._call_with_retries(messages, model, params, timeout)
        if self._clock() < self._primary_down_until:
            return self._call_fallback(messages, model, params)
        try:
            return self._call_with_retries(messages, model, params, timeout)
        except LLMError as exc:
            if not exc.retryable:
                raise
            self._primary_down_until = self._clock() + fb.cooldown_seconds
            get_logger().warning(
                "llm_fallback",
                primary=self.backend,
                fallback=fb.name,
                status=exc.status,
                cooldown_s=fb.cooldown_seconds,
            )
            return self._call_fallback(messages, model, params)

    def _call_fallback(self, messages: list[Message], model: str, params: dict[str, Any]) -> LLMResponse:
        fb = self._fallback
        assert fb is not None
        return self._call_with_retries(
            messages,
            fb.model_for(model, self.cfg.router_model),
            params,
            fb.timeout_seconds,
            sdk=self.fallback_sdk,
            backend=fb.name,
            max_attempts=1,  # a local server either answers or it does not; no point waiting it out
        )

    def _call_with_retries(
        self,
        messages: list[Message],
        model: str,
        params: dict[str, Any],
        timeout: float,
        *,
        sdk: Any = None,
        backend: str | None = None,
        max_attempts: int | None = None,
    ) -> LLMResponse:
        log = get_logger()
        sdk = sdk or self.sdk
        backend = backend or self.backend
        attempts = max_attempts or self.cfg.max_retries + 1
        for attempt in range(1, attempts + 1):
            t0 = time.perf_counter()
            try:
                raw = sdk.chat.completions.create(model=model, messages=messages, timeout=timeout, **params)
                return self._to_response(raw, model, (time.perf_counter() - t0) * 1000, attempt, backend)
            except openai.APIStatusError as exc:  # 4xx / 5xx
                status = exc.status_code
                if status == 429:
                    request_stats.record_rate_limited()
                if not (status == 429 or status == 408 or status >= 500):
                    raise LLMError(f"LLM request rejected (HTTP {status}): {_short(exc)}", status) from exc
                error, wait = exc, _retry_after_seconds(exc)
            except openai.APIConnectionError as exc:  # includes APITimeoutError
                status = None
                error, wait = exc, None
            except Exception as exc:  # noqa: BLE001  anything else is not retryable
                raise LLMError(f"LLM call failed: {type(exc).__name__}: {_short(exc)}") from exc

            delay = wait if wait is not None else self.cfg.retry_backoff_seconds * 2 ** (attempt - 1)
            if attempt == attempts or delay > self.cfg.retry_max_wait_seconds:
                raise LLMError(
                    f"LLM unavailable after {attempt} attempt(s): {type(error).__name__}: {_short(error)}",
                    status,
                    retryable=True,
                ) from error
            log.warning(
                "llm_retry",
                attempt=attempt,
                status=status,
                wait_s=round(delay, 2),
                model=model,
                backend=backend,
            )
            self._sleep(delay)
        raise LLMError("LLM call failed")  # pragma: no cover  (loop always returns or raises)

    @staticmethod
    def _to_response(raw: Any, model: str, latency_ms: float, attempts: int, backend: str) -> LLMResponse:
        choice = raw.choices[0] if raw.choices else None
        content = (choice.message.content if choice and choice.message else None) or ""
        u = getattr(raw, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )
        return LLMResponse(
            content=content,
            usage=usage,
            model=getattr(raw, "model", None) or model,
            finish_reason=getattr(choice, "finish_reason", None),
            latency_ms=latency_ms,
            attempts=attempts,
            backend=backend,
        )


def _short(exc: Exception, limit: int = 800) -> str:
    return str(exc).replace("\n", " ")[:limit]


def llm_client_from_settings(settings: Settings) -> LLMClient:
    cache = None
    if cache_enabled(settings.llm.dev_cache):
        cache = DiskCache(settings.llm_cache_dir)
        get_logger().warning("llm_dev_cache_on", dir=str(settings.llm_cache_dir))
    fallback = None
    llm = settings.llm
    if llm.fallback_enabled and llm.backend == "groq":
        fallback = FallbackBackend(
            name="ollama",
            base_url=llm.ollama.base_url,
            answer_model=llm.ollama.answer_model,
            router_model=llm.ollama.router_model,
            timeout_seconds=llm.ollama.timeout_seconds,
            cooldown_seconds=llm.fallback_cooldown_seconds,
        )
        get_logger().info("llm_fallback_on", fallback="ollama", base_url=llm.ollama.base_url)
    return LLMClient(settings, cache=cache, fallback=fallback)
