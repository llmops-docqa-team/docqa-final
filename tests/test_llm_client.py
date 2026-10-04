"""LLM client: retries, usage, timeouts, the dev disk cache. The OpenAI SDK is replaced by a scripted fake."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from app.config import load_settings
from app.llm.client import DiskCache, LLMClient, LLMError, cache_enabled, llm_client_from_settings
from app.llm.prompts import load_prompt

REQ = httpx.Request("POST", "http://llm.test/v1/chat/completions")


def completion(text="hi", prompt=10, completion_tokens=5, finish="stop"):
    return SimpleNamespace(
        model="m-1",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion_tokens, total_tokens=prompt + completion_tokens
        ),
    )


def status_error(status, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(status, request=REQ, headers=headers)
    cls = (
        openai.RateLimitError
        if status == 429
        else openai.InternalServerError
        if status >= 500
        else openai.BadRequestError
    )
    return cls("boom", response=resp, body=None)


class FakeSDK:
    """`outcomes` are returned or raised in order; every call's kwargs are recorded."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def cfg():
    s = load_settings()
    s.llm.max_retries = 2
    s.llm.retry_backoff_seconds = 0.5
    s.llm.retry_max_wait_seconds = 5
    return s


def make(cfg, outcomes, cache=None):
    sleeps: list[float] = []
    sdk = FakeSDK(outcomes)
    return LLMClient(cfg, cache=cache, sdk_client=sdk, sleep=sleeps.append), sdk, sleeps


MSGS = [{"role": "user", "content": "hello"}]


def test_returns_content_usage_and_passes_params(cfg):
    client, sdk, _ = make(cfg, [completion("ok", 12, 7)])
    r = client.chat(MSGS, model="m", max_tokens=99, reasoning_effort="low", json_mode=True)
    assert r.content == "ok" and r.finish_reason == "stop" and not r.cached
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens) == (12, 7, 19)
    kw = sdk.calls[0]
    assert kw["model"] == "m" and kw["max_tokens"] == 99 and kw["reasoning_effort"] == "low"
    assert kw["response_format"] == {"type": "json_object"} and kw["temperature"] == 0.0
    assert kw["timeout"] == cfg.llm.timeout_seconds


def test_defaults_to_the_answer_model(cfg):
    client, sdk, _ = make(cfg, [completion()])
    client.chat(MSGS)
    assert sdk.calls[0]["model"] == cfg.llm.answer_model
    assert "response_format" not in sdk.calls[0] and "reasoning_effort" not in sdk.calls[0]


def test_empty_content_is_returned_not_raised(cfg):
    client, _, _ = make(cfg, [completion(None, finish="length")])
    r = client.chat(MSGS)
    assert r.content == "" and r.finish_reason == "length"


def test_retries_429_then_succeeds_with_backoff(cfg):
    client, sdk, sleeps = make(cfg, [status_error(429), status_error(503), completion("ok")])
    r = client.chat(MSGS)
    assert r.content == "ok" and r.attempts == 3 and len(sdk.calls) == 3
    assert sleeps == [0.5, 1.0]


def test_gives_up_after_max_retries(cfg):
    client, sdk, sleeps = make(cfg, [status_error(500)] * 3)
    with pytest.raises(LLMError) as e:
        client.chat(MSGS)
    assert e.value.status == 500 and len(sdk.calls) == 3 and len(sleeps) == 2


def test_client_error_is_not_retried(cfg):
    client, sdk, sleeps = make(cfg, [status_error(400)])
    with pytest.raises(LLMError) as e:
        client.chat(MSGS)
    assert e.value.status == 400 and len(sdk.calls) == 1 and sleeps == []


def test_timeout_and_connection_errors_are_retried(cfg):
    client, sdk, _ = make(
        cfg, [openai.APITimeoutError(request=REQ), openai.APIConnectionError(request=REQ), completion("ok")]
    )
    assert client.chat(MSGS).content == "ok" and len(sdk.calls) == 3


def test_timeout_every_time_raises_llm_error(cfg):
    client, _, _ = make(cfg, [openai.APITimeoutError(request=REQ)] * 3)
    with pytest.raises(LLMError) as e:
        client.chat(MSGS)
    assert e.value.status is None and "APITimeoutError" in str(e.value)


def test_retry_after_is_honoured_when_short(cfg):
    client, _, sleeps = make(cfg, [status_error(429, retry_after=2), completion()])
    client.chat(MSGS)
    assert sleeps == [2.0]


def test_long_retry_after_fails_fast(cfg):
    client, sdk, sleeps = make(cfg, [status_error(429, retry_after=60), completion()])
    with pytest.raises(LLMError) as e:
        client.chat(MSGS)
    assert e.value.status == 429 and len(sdk.calls) == 1 and sleeps == []


def test_unexpected_exception_becomes_llm_error(cfg):
    client, _, _ = make(cfg, [ValueError("weird")])
    with pytest.raises(LLMError):
        client.chat(MSGS)


# ---- dev disk cache


def test_cache_hit_skips_the_call_and_keeps_usage(cfg, tmp_path):
    client, sdk, _ = make(cfg, [completion("first", 11, 4)], cache=DiskCache(tmp_path))
    a = client.chat(MSGS, model="m", json_mode=True)
    b = client.chat(MSGS, model="m", json_mode=True)
    assert len(sdk.calls) == 1
    assert (a.content, a.cached) == ("first", False) and (b.content, b.cached) == ("first", True)
    assert b.usage.total_tokens == 15


def test_cache_key_covers_model_messages_and_params(cfg, tmp_path):
    client, sdk, _ = make(
        cfg, [completion("a"), completion("b"), completion("c"), completion("d")], cache=DiskCache(tmp_path)
    )
    client.chat(MSGS, model="m1")
    client.chat(MSGS, model="m2")
    client.chat([{"role": "user", "content": "other"}], model="m1")
    client.chat(MSGS, model="m1", temperature=0.5)
    assert len(sdk.calls) == 4


def test_failures_are_not_cached(cfg, tmp_path):
    client, sdk, _ = make(cfg, [status_error(400), completion("ok")], cache=DiskCache(tmp_path))
    with pytest.raises(LLMError):
        client.chat(MSGS)
    assert client.chat(MSGS).content == "ok" and len(sdk.calls) == 2


def test_corrupt_cache_file_is_a_miss(cfg, tmp_path):
    cache = DiskCache(tmp_path)
    key = DiskCache.key("m", MSGS, {"temperature": 0.0})
    cache._path(key).parent.mkdir(parents=True)
    cache._path(key).write_text("{not json")
    client, sdk, _ = make(cfg, [completion("fresh")], cache=cache)
    assert client.chat(MSGS, model="m").content == "fresh" and len(sdk.calls) == 1


def test_cache_is_off_by_default_and_prod_forces_it_off():
    assert cache_enabled(False, {}) is False
    assert cache_enabled(True, {}) is True
    assert cache_enabled(False, {"DOCQA_LLM_CACHE": "1"}) is True
    assert cache_enabled(True, {"DOCQA_LLM_CACHE": "0"}) is False
    assert cache_enabled(True, {"DOCQA_LLM_CACHE": "1", "DOCQA_ENV": "prod"}) is False


def test_client_from_settings_respects_the_env_switch(cfg, monkeypatch, tmp_path):
    cfg.llm.cache_dir = str(tmp_path / "c")
    monkeypatch.delenv("DOCQA_LLM_CACHE", raising=False)
    monkeypatch.delenv("DOCQA_ENV", raising=False)
    assert llm_client_from_settings(cfg).cache is None
    monkeypatch.setenv("DOCQA_LLM_CACHE", "1")
    assert llm_client_from_settings(cfg).cache is not None
    monkeypatch.setenv("DOCQA_ENV", "prod")
    assert llm_client_from_settings(cfg).cache is None


# ---- prompt loader


def test_prompt_loader_reads_version_and_renders_once():
    p = load_prompt("answer_doc_v1")
    assert p.version == "v1" and p.name == "answer_doc" and "INSUFFICIENT" in p.system
    # a value containing another placeholder must not be substituted a second time
    out = p.render_user(sources="<source>{question}</source>", question="what is {sources}?")
    assert "<source>{question}</source>" in out and "what is {sources}?" in out
