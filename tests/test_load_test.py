"""scripts/load_test.py: the pure helpers, and the whole run against a scripted API (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from tests.fakes import FakeLLM
from tests.test_query_api import PARIS, running_app

_spec = importlib.util.spec_from_file_location(
    "load_test_script", Path(__file__).resolve().parent.parent / "scripts" / "load_test.py"
)
lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lt)

QUESTIONS = (
    [{"question": f"doc {i}?", "route": "DOCUMENT"} for i in range(10)]
    + [{"question": f"gen {i}?", "route": "GENERAL"} for i in range(6)]
    + [{"question": f"mix {i}?", "route": "MIXED"} for i in range(4)]
)


def reply(*, rate_limited=0, sections=("answered",), tokens=3000, route="DOCUMENT", fallback=0):
    return {
        "route": route,
        "sections": [{"kind": "document", "status": s} for s in sections],
        "tokens": {"total": {"prompt": tokens - 100, "completion": 100, "total": tokens}},
        "llm": {"calls": 2, "backends": ["groq"], "fallback_calls": fallback, "rate_limited": rate_limited},
    }


class Server:
    """A scripted /query: `script(n, question)` returns an httpx.Response, or raises an httpx error."""

    def __init__(self, script, delay=0.0):
        self.script, self.delay = script, delay
        self.count = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.questions: list[str] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/query":
            n, self.count = self.count, self.count + 1
            question = json.loads(request.content)["question"]
            self.questions.append(question)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                await asyncio.sleep(self.delay)
                return self.script(n, question)
            finally:
                self.in_flight -= 1
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/documents":
            return httpx.Response(200, json=[{"id": "d", "status": "READY"}])
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), base_url="http://api.test")


def ok(n, q):
    return httpx.Response(200, json=reply())


def run(server, **kw):
    args = dict(
        levels=[1, 3],
        duration=30,
        max_requests=4,
        token_budget=None,
        cooldown=0,
        timeout=5,
        seed=0,
        weights=lt.DEFAULT_WEIGHTS,
        log=lambda s: None,
    )
    args.update(kw)

    async def go():
        async with server.client() as client:
            return await lt.run_load_test(client, QUESTIONS, **args)

    return asyncio.run(go())


# ---------------------------------------------------------------- helpers


def test_percentile_interpolates():
    assert lt.percentile([], 50) is None
    assert lt.percentile([5], 95) == 5
    assert lt.percentile([1, 2, 3, 4], 50) == 2.5
    assert lt.percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48)


def test_weights_parse_and_reject_nonsense():
    assert lt.parse_weights("document=0.5, general=0.5") == {"DOCUMENT": 0.5, "GENERAL": 0.5}
    for bad in ("document=-1", "document=0", "oops"):
        with pytest.raises(ValueError):
            lt.parse_weights(bad)


def test_plan_is_reproducible_and_follows_the_mix():
    a = lt.build_plan(QUESTIONS, lt.DEFAULT_WEIGHTS, 200, seed=1)
    assert a == lt.build_plan(QUESTIONS, lt.DEFAULT_WEIGHTS, 200, seed=1)
    assert a != lt.build_plan(QUESTIONS, lt.DEFAULT_WEIGHTS, 200, seed=2)
    share = {r: sum(1 for q in a if q["route"] == r) / 200 for r in lt.DEFAULT_WEIGHTS}
    assert share["DOCUMENT"] > share["GENERAL"] > share["MIXED"]
    only_general = lt.build_plan(QUESTIONS, {"GENERAL": 1.0}, 20, seed=0)
    assert {q["route"] for q in only_general} == {"GENERAL"}
    with pytest.raises(ValueError):
        lt.build_plan(QUESTIONS, {"NOPE": 1.0}, 5, seed=0)


def test_classify_each_kind_of_outcome():
    c = lt.classify
    assert c(200, reply(), None)["kind"] == "ok"
    assert c(200, reply(), None)["tokens"] == 3000
    degraded = c(200, reply(sections=("error",), rate_limited=3), None)
    assert degraded["kind"] == "degraded" and degraded["rate_limited"] == 3
    assert c(200, reply(sections=("abstained",)), None)["kind"] == "ok"  # an abstention is an answer
    assert c(200, reply(rate_limited=2), None)["rate_limited"] == 2  # retried 429s still count
    assert c(200, reply(fallback=1), None)["fallback_calls"] == 1
    assert c(429, None, None) == {**c(429, None, None), "kind": "http_error", "rate_limited": 1}
    assert c(500, None, None)["kind"] == "http_error"
    assert c(None, None, "timeout")["kind"] == "timeout"
    assert c(None, None, "connection")["kind"] == "connection"
    assert c(200, {}, None)["kind"] == "ok"  # an old API without the llm block: nothing to count


def test_classify_reads_the_real_query_response(settings, monkeypatch):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        body = client.post("/query", json={"question": "Capital of France?"}).json()
    out = lt.classify(200, body, None)
    assert out["kind"] == "ok" and out["tokens"] == 60 and out["route"] == "GENERAL"
    assert out["rate_limited"] == 0


def test_level_verdicts():
    def level(outcomes, lat=(100.0,)):
        return lt.summarize_level(2, outcomes, list(lat) * len(outcomes), 60.0)

    good = {"kind": "ok", "rate_limited": 0, "fallback_calls": 0, "tokens": 10, "route": "DOCUMENT"}
    bad = {**good, "kind": "degraded", "rate_limited": 3}
    assert level([good] * 10)["verdict"] == "ok"
    strained = level([{**good, "rate_limited": 1}] * 10)
    assert strained["verdict"] == "strained" and strained["count_429"] == 10 and strained["error_rate"] == 0
    assert level([good] * 10, lat=(7000.0,))["verdict"] == "strained"  # slow, no errors
    broken = level([good] * 9 + [bad] * 1)  # 10% errors
    assert broken["verdict"] == "broken" and broken["error_rate"] == 0.1 and broken["degraded"] == 1
    assert level([good] * 19 + [bad])["verdict"] == "broken"  # exactly 5% counts
    assert level([good] * 10)["req_per_min"] == 10.0 and level([good] * 10)["p50_ms"] == 100.0
    assert level([])["requests"] == 0  # a level that sent nothing does not divide by zero


def test_bursts_are_flagged_as_extrapolated_and_the_table_says_so():
    good = {"kind": "ok", "rate_limited": 0, "fallback_calls": 0, "tokens": 3000, "route": "DOCUMENT"}
    burst = lt.summarize_level(5, [good] * 5, [100.0] * 5, 3.0)
    steady = lt.summarize_level(1, [good] * 5, [100.0] * 5, 60.0)
    assert burst["extrapolated"] is True and steady["extrapolated"] is False
    assert burst["tokens_per_min"] == 300000 and steady["tokens_per_min"] == 15000
    table = lt.markdown_table({"levels": [steady, burst]})
    assert "| 5 | 5 | 100.0* |" in table and "| 1 | 5 | 5.0 |" in table
    assert "extrapolated" in table and "extrapolated" not in lt.markdown_table({"levels": [steady]})


def test_find_breaks_names_the_first_level():
    levels = [
        {"concurrency": 1, "count_429": 0, "p95_ms": 4000, "verdict": "ok"},
        {"concurrency": 3, "count_429": 2, "p95_ms": 9000, "verdict": "strained"},
        {"concurrency": 5, "count_429": 9, "p95_ms": 12000, "verdict": "broken"},
        {"concurrency": 8, "skipped": "token budget reached"},
    ]
    s = lt.find_breaks(levels)
    assert (s["first_429_at"], s["first_slow_at"], s["breaks_at"]) == (3, 3, 5)
    assert lt.find_breaks(levels[:1])["breaks_at"] is None


# ---------------------------------------------------------------- whole runs


def test_each_level_sends_up_to_the_request_cap_with_that_many_users_at_once():
    server = Server(ok, delay=0.02)
    report = run(server, levels=[1, 3], max_requests=4)
    assert [lv["requests"] for lv in report["levels"]] == [4, 4]
    assert server.count == 8
    assert server.max_in_flight == 3  # level 1 never overlapped; level 3 did, and only 3 at a time
    assert report["summary"]["breaks_at"] is None
    assert report["meta"]["tokens_spent"] == 8 * 3000


def test_every_level_sees_the_same_questions():
    server = Server(ok)
    run(server, levels=[1, 1], max_requests=5)
    assert server.questions[:5] == server.questions[5:]


def test_the_duration_caps_a_level_too():
    server = Server(ok, delay=0.05)
    report = run(server, levels=[1], duration=0.12, max_requests=100)
    assert 1 <= report["levels"][0]["requests"] < 10


def test_429s_degraded_answers_timeouts_and_http_errors_are_all_counted():
    def script(n, q):
        if n == 0:
            return httpx.Response(200, json=reply(rate_limited=2))  # retried through, still a 429 seen
        if n == 1:
            return httpx.Response(200, json=reply(sections=("error",), rate_limited=3))
        if n == 2:
            raise httpx.ReadTimeout("slow")
        if n == 3:
            return httpx.Response(500)
        return httpx.Response(429)

    report = run(Server(script), levels=[1], max_requests=5)
    lv = report["levels"][0]
    assert lv["requests"] == 5 and lv["count_429"] == 2 + 3 + 1
    assert (lv["degraded"], lv["timeouts"], lv["http_errors"]) == (1, 1, 2)
    assert lv["error_rate"] == 0.8 and lv["failure_rate"] == 0.6 and lv["verdict"] == "broken"
    assert report["summary"]["breaks_at"] == 1 and report["summary"]["first_429_at"] == 1


def test_it_reports_where_it_breaks_across_levels():
    def script(n, q):  # fine until the third request overall, then 429s that turn into failed answers
        return httpx.Response(200, json=reply() if n < 4 else reply(sections=("error",), rate_limited=3))

    report = run(Server(script), levels=[1, 3], max_requests=4)
    assert [lv["verdict"] for lv in report["levels"]] == ["ok", "broken"]
    assert report["summary"]["breaks_at"] == 3 and report["summary"]["first_429_at"] == 3


def test_the_token_budget_stops_issuing_and_skips_later_levels():
    server = Server(ok)  # 3,000 tokens a request
    report = run(server, levels=[1, 3, 5], max_requests=10, token_budget=7000)
    first, second, third = report["levels"]
    assert first["requests"] == 3  # 3,000 + 3,000 + 3,000: the third is the one that crosses 7,000
    assert second == {"concurrency": 3, "skipped": "token budget reached"}
    assert third == {"concurrency": 5, "skipped": "token budget reached"}
    assert server.count == 3
    assert report["summary"]["breaks_at"] is None  # a skipped level is not a verdict


def test_cooldown_is_waited_between_levels_but_not_before_the_first():
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    server = Server(ok)

    async def go():
        async with server.client() as client:
            return await lt.run_load_test(
                client,
                QUESTIONS,
                levels=[1, 3, 5],
                duration=30,
                max_requests=1,
                token_budget=None,
                cooldown=65,
                timeout=5,
                seed=0,
                weights=lt.DEFAULT_WEIGHTS,
                sleep=fake_sleep,
                log=lambda s: None,
            )

    asyncio.run(go())
    assert slept == [65, 65]


def test_preflight_warns_when_nothing_is_ready_and_fails_when_the_api_is_down():
    async def empty(request):
        return httpx.Response(200, json=[] if request.url.path == "/documents" else {"status": "ok"})

    async def down(request):
        raise httpx.ConnectError("refused")

    async def go(handler):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://api.test") as c:
            return await lt.preflight(c)

    assert "no document is READY" in asyncio.run(go(empty))
    assert asyncio.run(go(Server(ok).handler)) is None
    with pytest.raises(httpx.HTTPError):
        asyncio.run(go(down))


def test_report_is_json_and_the_markdown_table_has_a_row_per_level():
    report = run(Server(ok), levels=[1, 3], max_requests=2)
    json.dumps(report)
    table = [line for line in lt.markdown_table(report).splitlines() if line.startswith("|")]
    assert len(table) == 2 + 2 and table[2].startswith("| 1 |") and table[3].startswith("| 3 |")


def test_cli_defaults_are_small():
    args = lt.parse_args([])
    assert args.levels == "1,3,5" and args.max_requests <= 8 and args.token_budget <= 80000
    assert args.out.name == "load_test.json" and args.out.parent.name == "results"
