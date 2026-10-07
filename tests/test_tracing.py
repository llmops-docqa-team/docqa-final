"""Langfuse tracing: off without keys, never fails a request, the span tree, the content policy, scores.

The Langfuse client is replaced by `FakeLangfuse`, which keeps parent/child links the way OpenTelemetry's
context does (a context variable), so the tests also prove the tree is right across the MIXED path's threads.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import sys
from typing import Any

import pytest

from app.observability.tracing import LangfuseTracer, Tracer, build_tracer, get_tracer, set_tracer
from tests.fakes import FakeLLM
from tests.test_query_api import (
    INSUFFICIENT,
    PARIS,
    answer_json,
    ready_doc,
    route_json,
    running_app,
)

# ---------------------------------------------------------------- a fake Langfuse client


class Node:
    def __init__(self, as_type: str, name: str, parent: Node | None, kwargs: dict[str, Any]):
        self.as_type, self.name, self.parent, self.kwargs = as_type, name, parent, kwargs
        self.updates: list[dict[str, Any]] = []
        self.children: list[Node] = []
        self.ended = False

    def update(self, **fields: Any) -> None:
        self.updates.append(fields)

    def field(self, key: str) -> Any:
        """The last value set for `key`, at creation or by an update."""
        value = self.kwargs.get(key)
        for u in self.updates:
            if key in u:
                value = u[key]
        return value

    def meta(self) -> dict[str, Any]:
        merged = dict(self.kwargs.get("metadata") or {})
        for u in self.updates:
            merged.update(u.get("metadata") or {})
        return merged

    def has_content(self) -> bool:
        return any(k in d for d in [self.kwargs, *self.updates] for k in ("input", "output"))


class _Observation:
    def __init__(self, client: FakeLangfuse, node: Node):
        self._client, self._node = client, node

    def update(self, **fields: Any) -> None:
        if self._client.fail_update:
            raise RuntimeError("update exploded")
        self._node.update(**fields)


class _Scope:
    def __init__(self, client: FakeLangfuse, node: Node):
        self._client, self._node = client, node

    def __enter__(self):
        self._token = self._client.current.set(self._node)
        return _Observation(self._client, self._node)

    def __exit__(self, *exc):
        self._client.current.reset(self._token)
        self._node.ended = True
        if self._client.fail_exit:
            raise RuntimeError("exit exploded")
        return False


class FakeLangfuse:
    def __init__(self, *, fail_start=False, fail_update=False, fail_exit=False, fail_score=False):
        self.fail_start, self.fail_update, self.fail_exit, self.fail_score = (
            fail_start,
            fail_update,
            fail_exit,
            fail_score,
        )
        self.current: contextvars.ContextVar[Node | None] = contextvars.ContextVar("fake_lf", default=None)
        self.nodes: list[Node] = []
        self.scores: list[dict[str, Any]] = []
        self.flushed = 0

    @staticmethod
    def create_trace_id(*, seed: str | None = None) -> str:
        return hashlib.sha256((seed or "").encode()).hexdigest()[:32]

    def start_as_current_observation(self, *, as_type: str, name: str, **kwargs: Any) -> _Scope:
        if self.fail_start:
            raise RuntimeError("start exploded")
        node = Node(as_type, name, self.current.get(), kwargs)
        if node.parent is not None:
            node.parent.children.append(node)
        self.nodes.append(node)
        return _Scope(self, node)

    def create_score(self, **kwargs: Any) -> None:
        if self.fail_score:
            raise RuntimeError("score exploded")
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed += 1
        if self.fail_score:
            raise RuntimeError("flush exploded")

    def named(self, name: str) -> Node:
        [node] = [n for n in self.nodes if n.name == name]
        return node


def tree(node: Node) -> dict[str, Any]:
    return {node.name: [tree(c) for c in node.children]} if node.children else node.name


# ---------------------------------------------------------------- off by default


def test_no_keys_means_the_no_op_tracer():
    assert build_tracer(env={}).enabled is False
    assert build_tracer(env={"LANGFUSE_PUBLIC_KEY": "pk"}).enabled is False  # one key is not enough
    assert build_tracer(env={"LANGFUSE_SECRET_KEY": "sk"}).enabled is False
    assert build_tracer(env={"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}).enabled is False


def test_the_no_op_tracer_runs_the_block_and_swallows_everything():
    t = Tracer()
    ran = []
    with t.trace("q", "id", input="x") as root, t.span("s") as s, t.generation("g", model="m") as g:
        root.update(output="y")
        s.update(metadata={"a": 1})
        g.update(usage_details={"input": 1})
        ran.append(1)
    t.score("id", "n", 1)
    t.flush()
    assert ran == [1]


def test_keys_alone_do_not_switch_it_off_when_the_kill_switch_is_set():
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_TRACING_ENABLED": "false"}
    assert build_tracer(env=env).enabled is False


def test_a_missing_sdk_or_a_failing_constructor_falls_back_to_the_no_op_tracer(monkeypatch):
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
    monkeypatch.setitem(sys.modules, "langfuse", None)  # `from langfuse import Langfuse` -> ImportError
    assert build_tracer(env=env).enabled is False

    class Broken:
        def __init__(self, **kw):
            raise RuntimeError("bad config")

    monkeypatch.setitem(sys.modules, "langfuse", type("M", (), {"Langfuse": Broken}))
    assert build_tracer(env=env).enabled is False


def test_keys_build_a_langfuse_tracer_with_the_given_server(monkeypatch):
    seen: dict[str, Any] = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setitem(sys.modules, "langfuse", type("M", (), {"Langfuse": Recorder}))
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_BASE_URL": "https://lf.test"}
    t = build_tracer(env=env, capture_content=True, release="abc123")
    assert t.enabled and t.capture_content
    assert (seen["public_key"], seen["base_url"], seen["release"]) == ("pk", "https://lf.test", "abc123")


# ---------------------------------------------------------------- the wrapper never fails a request


@pytest.mark.parametrize("broken", ["fail_start", "fail_update", "fail_exit", "fail_score"])
def test_a_broken_sdk_never_raises_into_the_caller(broken):
    t = LangfuseTracer(FakeLangfuse(**{broken: True}), capture_content=True)
    ran = []
    with t.trace("q", "id", input="x") as root, t.generation("g", model="m") as g:
        root.update(output="y")
        g.update(usage_details={"input": 1})
        ran.append(1)
    t.score("id", "user_feedback", 1)
    t.flush()
    assert ran == [1]


def test_an_exception_in_the_block_propagates_unchanged_and_ends_the_span():
    client = FakeLangfuse(fail_exit=True)  # even a failing exit must not replace the caller's exception
    t = LangfuseTracer(client)
    with pytest.raises(ValueError, match="boom"):
        with t.span("s"):
            raise ValueError("boom")
    assert client.named("s").ended


def test_trace_ids_are_stable_so_late_scores_land_on_the_right_trace():
    client = FakeLangfuse()
    t = LangfuseTracer(client)
    with t.trace("query", "req-1"):
        pass
    t.score("req-1", "user_feedback", -1)
    t.score("req-1", "judge_correct", True, boolean=True)
    ctx = client.named("query").kwargs["trace_context"]["trace_id"]
    assert ctx == client.create_trace_id(seed="req-1")
    assert [s["trace_id"] for s in client.scores] == [ctx, ctx]
    assert client.scores[0]["data_type"] == "NUMERIC" and client.scores[0]["value"] == -1.0
    assert client.scores[1]["data_type"] == "BOOLEAN" and client.scores[1]["value"] == 1.0


def test_content_is_dropped_unless_capture_content_is_on():
    for capture in (False, True):
        client = FakeLangfuse()
        t = LangfuseTracer(client, capture_content=capture)
        with t.trace("q", "id", input="the question", metadata={"n": 1}) as root:
            root.update(output="the answer", metadata={"route": "DOCUMENT"})
        node = client.named("q")
        assert node.has_content() is capture
        assert node.meta() == {"n": 1, "route": "DOCUMENT"}  # metadata is never dropped


# ---------------------------------------------------------------- the span tree of a real request


@pytest.fixture
def lf():
    client = FakeLangfuse()
    set_tracer(LangfuseTracer(client))
    return client


def test_document_request_span_tree(settings, monkeypatch, tmp_path, lf):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        set_tracer(LangfuseTracer(lf))  # the app's startup installed the no-op tracer; swap ours in
        r = client.post("/query", json={"question": "What was revenue?"}, headers={"X-Request-ID": "t-doc"})
    assert r.status_code == 200 and r.json()["sections"][0]["status"] == "answered"

    [root] = [n for n in lf.nodes if n.parent is None]
    assert root.name == "query"
    assert root.kwargs["trace_context"]["trace_id"] == lf.create_trace_id(seed="t-doc")
    assert tree(root) == {"query": ["router", {"document": ["retrieve", "generate"]}]}

    router, generate = lf.named("router"), lf.named("generate")
    assert router.as_type == "generation" and generate.as_type == "generation"
    assert (router.meta()["prompt"], router.meta()["prompt_version"]) == ("router_v1", "v1")
    assert (generate.meta()["prompt"], generate.meta()["prompt_version"]) == ("answer_doc_v1", "v1")
    assert generate.field("model") == "fake-doc" and router.field("model") == "fake-router"
    assert generate.field("usage_details") == {"input": 50, "output": 10, "total": 60}
    assert router.field("usage_details") == {"input": 50, "output": 10, "total": 60}
    assert generate.meta()["source_ids"] and generate.meta()["llm_calls"] == 1

    retrieve = lf.named("retrieve")
    assert set(generate.meta()["source_ids"]) <= set(retrieve.meta()["source_ids"])
    assert lf.named("document").meta()["status"] == "answered"
    assert lf.named("document").meta()["cited_chunks"]
    assert root.meta()["route"] == "DOCUMENT" and root.meta()["tokens"]["total"] == 120
    assert all(n.ended for n in lf.nodes)
    # Default policy: no question / answer / snippet text went to the (fake) server.
    assert not any(n.has_content() for n in lf.nodes)
    assert "12,563" not in json.dumps([n.updates for n in lf.nodes])


def test_with_capture_content_the_question_answer_and_snippets_are_sent(settings, monkeypatch, tmp_path):
    client_lf = FakeLangfuse()
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        set_tracer(LangfuseTracer(client_lf, capture_content=True))
        client.post("/query", json={"question": "What was revenue?"})
    assert client_lf.named("query").field("input") == "What was revenue?"
    assert client_lf.named("query").field("output") == "Revenue was 12,563 crore."
    snippets = client_lf.named("retrieve").field("output")
    assert snippets and all({"chunk_id", "page", "snippet"} == set(s) for s in snippets)
    assert all(len(s["snippet"]) <= settings.answer.snippet_chars for s in snippets)


def test_mixed_request_keeps_both_paths_under_the_root(settings, monkeypatch, tmp_path, lf):
    llm = FakeLLM(
        settings,
        router=[route_json("MIXED", "What was revenue?", "What is the capital of France?")],
        doc=[answer_json()],
        general=[PARIS],
    )
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        set_tracer(LangfuseTracer(lf))
        client.post("/query", json={"question": "Revenue, and what is the capital of France?"})
    [root] = [n for n in lf.nodes if n.parent is None]
    assert sorted(c.name for c in root.children) == ["document", "general", "router"]
    assert tree(lf.named("general")) == {"general": ["generate"]}
    assert lf.named("general").meta()["status"] == "answered"
    assert all(n.ended for n in lf.nodes)


def test_abstention_and_llm_outage_are_visible_on_the_spans(settings, monkeypatch, tmp_path, lf):
    from app.llm.client import LLMError

    llm = FakeLLM(
        settings,
        router=[route_json("DOCUMENT"), route_json("DOCUMENT")],
        doc=[INSUFFICIENT, LLMError("down", 503, retryable=True)],
    )
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        set_tracer(LangfuseTracer(lf))
        client.post("/query", json={"question": "Something not in the report?"})
        client.post("/query", json={"question": "Revenue again?"})
    docs = [n for n in lf.nodes if n.name == "document"]
    assert [d.meta()["status"] for d in docs] == ["abstained", "error"]
    assert docs[0].meta()["abstain_reason"] == "insufficient"
    generates = [n for n in lf.nodes if n.name == "generate"]
    assert generates[1].field("level") == "ERROR"


def test_a_tracer_that_fails_everywhere_does_not_fail_the_request(settings, monkeypatch, tmp_path):
    broken = FakeLangfuse(fail_start=True, fail_update=True, fail_exit=True, fail_score=True)
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        set_tracer(LangfuseTracer(broken, capture_content=True))
        r = client.post("/query", json={"question": "What was revenue?"}, headers={"X-Request-ID": "t-b"})
        f = client.post("/feedback", json={"trace_id": "t-b", "value": 1})
    assert r.status_code == 200 and r.json()["sections"][0]["status"] == "answered"
    assert f.status_code == 200
    assert client.app.state.request_store.get("t-b")["feedback"] == 1


# ---------------------------------------------------------------- scores


def test_feedback_becomes_a_langfuse_score(settings, monkeypatch, lf):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        set_tracer(LangfuseTracer(lf))
        client.post("/query", json={"question": "Capital of France?"}, headers={"X-Request-ID": "t-fb"})
        client.post("/feedback", json={"trace_id": "t-fb", "value": -1})
        client.post("/feedback", json={"trace_id": "t-fb", "value": 1})
    ids = {s["trace_id"] for s in lf.scores}
    assert ids == {lf.create_trace_id(seed="t-fb")}
    assert [(s["name"], s["value"]) for s in lf.scores] == [("user_feedback", -1.0), ("user_feedback", 1.0)]


def test_judge_verdicts_become_langfuse_scores(settings, lf):
    from app.storage.db import init_db
    from app.storage.requests import RequestStore
    from eval.online_judge import judge_recent
    from tests.test_online_judge import FakeJudge, logged, section

    init_db(settings.sqlite_path)
    store = RequestStore(settings.sqlite_path)
    logged(store, "t-j", sections=[section()])
    set_tracer(LangfuseTracer(lf))
    judge_recent(store, FakeJudge(), 10)
    assert {(s["name"], s["value"], s["data_type"]) for s in lf.scores} == {
        ("judge_correct", 1.0, "BOOLEAN"),
        ("judge_grounded", 1.0, "BOOLEAN"),
    }
    assert {s["trace_id"] for s in lf.scores} == {lf.create_trace_id(seed="t-j")}


def test_the_app_starts_untraced_and_flushes_the_tracer_on_shutdown(settings, monkeypatch):
    flushed = []

    class Spy(Tracer):
        def flush(self):
            flushed.append(1)

    llm = FakeLLM(settings)
    with running_app(settings, monkeypatch, llm):
        assert get_tracer().enabled is False  # no LANGFUSE_* keys in the environment
        set_tracer(Spy())
    assert flushed == [1]
