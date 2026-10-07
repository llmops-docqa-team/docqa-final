"""LLM backend selection (groq | ollama) and the automatic Groq -> Ollama fallback (scripted fake SDKs)."""

from __future__ import annotations

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.llm import request_stats
from app.llm.client import DiskCache, FallbackBackend, LLMClient, LLMError, llm_client_from_settings
from app.main import create_app
from app.observability.request_log import build_record
from tests.conftest import FakeEmbedder
from tests.test_llm_client import FakeSDK, completion, status_error

MSGS = [{"role": "user", "content": "hello"}]
TIMEOUT = openai.APITimeoutError(request=httpx.Request("POST", "http://llm.test/v1/chat/completions"))


# ---------------------------------------------------------------- configuration


def test_default_is_groq_with_no_fallback():
    s = load_settings()
    assert s.llm.backend == "groq" and s.llm.fallback_enabled is False
    assert "groq.com" in s.llm.base_url
    assert llm_client_from_settings(s)._fallback is None


def test_llm_backend_ollama_makes_the_local_server_the_primary(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    s = load_settings()
    assert s.llm.backend == "ollama"
    assert s.llm.base_url == s.llm.ollama.base_url
    assert (s.llm.answer_model, s.llm.router_model) == (s.llm.ollama.answer_model, s.llm.ollama.router_model)
    assert s.llm.timeout_seconds == s.llm.ollama.timeout_seconds
    assert s.llm.fallback_enabled is False  # nothing left to fall back to
    client = llm_client_from_settings(s)
    assert client.backend == "ollama" and client._fallback is None


def test_ollama_base_url_env_turns_the_fallback_on_and_sets_the_address(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
    s = load_settings()
    assert s.llm.backend == "groq" and s.llm.fallback_enabled is True
    assert s.llm.ollama.base_url == "http://host.docker.internal:11434/v1"
    fb = llm_client_from_settings(s)._fallback
    assert fb is not None and fb.base_url == "http://host.docker.internal:11434/v1"
    assert fb.answer_model == s.llm.ollama.answer_model
    assert fb.cooldown_seconds == s.llm.fallback_cooldown_seconds


def test_llm_model_still_overrides_the_answer_model_on_either_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.2:3b")
    s = load_settings()
    assert s.llm.answer_model == "llama3.2:3b" and s.llm.router_model == s.llm.ollama.router_model


def test_an_unknown_backend_is_refused(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        load_settings()


def test_the_judge_client_never_gets_the_fallback(monkeypatch):
    from eval.judge import judge_llm_from_settings

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    s = load_settings()
    s.gemini_api_key = "g-key"
    assert llm_client_from_settings(s)._fallback is not None
    assert judge_llm_from_settings(s, use_cache=False)._fallback is None


# ---------------------------------------------------------------- the client


@pytest.fixture
def cfg():
    s = load_settings()
    s.llm.max_retries = 2
    s.llm.retry_backoff_seconds = 0.5
    s.llm.retry_max_wait_seconds = 5
    return s


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


FALLBACK = FallbackBackend(
    "ollama", "http://ollama.test/v1", answer_model="local-answer", router_model="local-router",
    timeout_seconds=99.0, cooldown_seconds=30.0,
)  # fmt: skip


def make(cfg, primary, fallback, cache=None, clock=None):
    sleeps: list[float] = []
    p, f = FakeSDK(primary), FakeSDK(fallback)
    client = LLMClient(
        cfg,
        cache=cache,
        sdk_client=p,
        sleep=sleeps.append,
        fallback=FALLBACK,
        fallback_sdk_client=f,
        clock=clock or Clock(),
    )
    return client, p, f, sleeps


def test_without_a_fallback_an_exhausted_429_is_a_retryable_error(cfg):
    sdk = FakeSDK([status_error(429)] * 3)
    client = LLMClient(cfg, sdk_client=sdk, sleep=lambda s: None)
    with pytest.raises(LLMError) as exc:
        client.chat(MSGS)
    assert exc.value.status == 429 and exc.value.retryable is True


def test_a_rejected_request_is_not_retryable(cfg):
    client = LLMClient(cfg, sdk_client=FakeSDK([status_error(400)]), sleep=lambda s: None)
    with pytest.raises(LLMError) as exc:
        client.chat(MSGS)
    assert exc.value.status == 400 and exc.value.retryable is False


def test_repeated_429s_are_repeated_on_ollama_with_its_model_and_timeout(cfg):
    client, primary, fallback, sleeps = make(cfg, [status_error(429)] * 3, [completion("local answer")])
    r = client.chat(MSGS, model=cfg.llm.answer_model, reasoning_effort="low", json_mode=True)
    assert r.content == "local answer" and r.backend == "ollama"
    assert len(primary.calls) == 3 and len(fallback.calls) == 1  # the primary's own retries came first
    kw = fallback.calls[0]
    assert kw["model"] == "local-answer" and kw["timeout"] == 99.0
    assert kw["response_format"] == {"type": "json_object"} and kw["reasoning_effort"] == "low"
    assert len(sleeps) == 2


def test_the_router_model_maps_to_the_fallback_router_model(cfg):
    client, _, fallback, _ = make(cfg, [status_error(429)] * 3, [completion()])
    client.chat(MSGS, model=cfg.llm.router_model)
    assert fallback.calls[0]["model"] == "local-router"


def test_timeouts_and_server_errors_fall_back_too(cfg):
    for failures in ([TIMEOUT] * 3, [status_error(503)] * 3):
        client, _, fallback, _ = make(cfg, failures, [completion("x")])
        assert client.chat(MSGS).backend == "ollama" and len(fallback.calls) == 1


def test_a_retry_after_longer_than_the_cap_falls_back_at_once(cfg):
    client, primary, fallback, sleeps = make(cfg, [status_error(429, retry_after=60)], [completion("x")])
    assert client.chat(MSGS).backend == "ollama"
    assert len(primary.calls) == 1 and sleeps == []  # no waiting a minute for Groq


def test_a_primary_that_recovers_inside_its_retries_never_touches_ollama(cfg):
    client, primary, fallback, _ = make(cfg, [status_error(429), completion("groq answer")], [])
    r = client.chat(MSGS)
    assert r.backend == "groq" and r.content == "groq answer" and fallback.calls == []


def test_a_rejected_request_is_not_sent_to_ollama(cfg):
    client, _, fallback, _ = make(cfg, [status_error(400)], [completion()])
    with pytest.raises(LLMError):
        client.chat(MSGS)
    assert fallback.calls == []


def test_when_ollama_also_fails_the_call_fails(cfg):
    down = openai.APIConnectionError(request=TIMEOUT.request)
    client, _, fallback, _ = make(cfg, [status_error(429)] * 3, [down])
    with pytest.raises(LLMError) as exc:
        client.chat(MSGS)
    assert exc.value.retryable is True
    assert len(fallback.calls) == 1  # one attempt: a local server either answers or it does not


def test_after_a_fallback_the_primary_is_skipped_until_the_cooldown_ends(cfg):
    clock = Clock()
    primary_outcomes = [status_error(429)] * 3 + [completion("groq is back")]
    client, primary, fallback, _ = make(
        cfg, primary_outcomes, [completion("o1"), completion("o2")], clock=clock
    )
    assert client.chat(MSGS).content == "o1"
    n_primary = len(primary.calls)
    clock.now += 10  # still cooling down: straight to Ollama, no retries burnt on Groq
    assert client.chat(MSGS).content == "o2"
    assert len(primary.calls) == n_primary
    clock.now += 25  # 35 s after the fallback: the primary is tried again
    r = client.chat(MSGS)
    assert r.backend == "groq" and r.content == "groq is back"


def test_only_the_primarys_answers_are_cached(cfg, tmp_path):
    cache = DiskCache(tmp_path / "cache")
    key = DiskCache.key("m", MSGS, {"temperature": 0.0})
    client, _, _, _ = make(cfg, [status_error(429)] * 3, [completion("from ollama")], cache)
    assert client.chat(MSGS, model="m").backend == "ollama"
    assert cache.get(key) is None  # a local model's answer must not be replayed later as Groq's

    groq = LLMClient(cfg, cache=cache, sdk_client=FakeSDK([completion("from groq")]), sleep=lambda s: None)
    assert groq.chat(MSGS, model="m").backend == "groq"
    assert cache.get(key).content == "from groq"


# ---------------------------------------------------------------- request stats


def test_request_stats_count_backends_fallbacks_and_429s(cfg):
    client, _, _, _ = make(cfg, [status_error(429)] * 4, [completion("a"), completion("b")])
    with request_stats.track() as stats:
        client.chat(MSGS)
        client.chat(MSGS)
    assert stats.summary() == {"calls": 2, "backends": ["ollama"], "fallback_calls": 2, "rate_limited": 3}


def test_mixed_backends_are_both_listed(cfg):
    clock = Clock()
    client, _, _, _ = make(cfg, [status_error(429)] * 3 + [completion("g")], [completion("o")], clock=clock)
    with request_stats.track() as stats:
        client.chat(MSGS)
        clock.now += 60
        client.chat(MSGS)
    s = stats.summary()
    assert s["backends"] == ["groq", "ollama"] and s["fallback_calls"] == 1 and s["calls"] == 2


def test_outside_a_request_nothing_is_recorded(cfg):
    client, _, _, _ = make(cfg, [completion("x")], [])
    client.chat(MSGS)  # no track(): must not raise
    request_stats.record_rate_limited()
    request_stats.record_served("groq")


def test_the_request_log_row_names_the_backend():
    from app.config import ObservabilityConfig

    base = {
        "trace_id": "t", "route": "GENERAL", "router_ok": True,
        "router": {"skipped": True, "model": None, "prompt_version": "v1", "fallback_reason": None},
        "sections": [], "timings": {},
        "tokens": {
            k: {"prompt": 0, "completion": 0, "total": 0} for k in ("router", "document", "general", "total")
        },
    }  # fmt: skip

    def row(llm):
        record = {**base, "llm": llm}
        return build_record(
            record, question_len=5, doc=None, gen=None, obs=ObservabilityConfig(), app_version="x"
        )

    r = row({"calls": 2, "backends": ["groq", "ollama"], "fallback_calls": 1, "rate_limited": 3})
    assert (r["llm_backend"], r["llm_fallbacks"], r["llm_rate_limited"]) == ("groq+ollama", 1, 3)
    r = row({"calls": 0, "backends": [], "fallback_calls": 0, "rate_limited": 0})
    assert (r["llm_backend"], r["llm_fallbacks"], r["llm_rate_limited"]) == (None, None, None)
    assert row(None)["llm_backend"] is None  # an old-shaped response


# ---------------------------------------------------------------- through the app


def test_query_served_by_the_fallback_is_logged_and_reported(settings, monkeypatch):
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    llm = LLMClient(
        settings,
        sdk_client=FakeSDK([status_error(429)] * 3),
        fallback=FALLBACK,
        fallback_sdk_client=FakeSDK([completion("Paris is the capital of France.", 40, 8)]),
        sleep=lambda s: None,
    )
    with TestClient(create_app(embedder=FakeEmbedder(), llm=llm)) as client:  # no documents -> general path
        r = client.post("/query", json={"question": "Capital of France?"}, headers={"X-Request-ID": "t-fb"})
        row = client.app.state.request_store.get("t-fb")
    body = r.json()
    assert r.status_code == 200 and body["sections"][0]["status"] == "answered"
    assert body["llm"] == {"calls": 1, "backends": ["ollama"], "fallback_calls": 1, "rate_limited": 3}
    assert (row["llm_backend"], row["llm_fallbacks"], row["llm_rate_limited"]) == ("ollama", 1, 3)
    assert row["status"] == "answered"
