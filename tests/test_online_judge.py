"""Online judging: the opt-in content log, the request store's judge methods, the judge_recent logic."""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.answering.document import DocAnswer
from app.observability.metrics import load_requests, quality_metrics
from app.storage.db import init_db
from app.storage.requests import RequestStore
from eval.judge import JudgeError, JudgeInput, Verdict
from eval.online_judge import judge_recent, summary
from tests.fakes import FakeLLM
from tests.test_query_api import (
    INSUFFICIENT,
    PARIS,
    answer_json,
    ready_doc,
    route_json,
    running_app,
)


class FakeJudge:
    model, prompt_version = "judge-m", "v1"

    def __init__(self, verdicts=None, fail_for=()):
        self.verdicts, self.fail_for, self.items = verdicts or {}, set(fail_for), []

    def judge(self, item: JudgeInput) -> Verdict:
        self.items.append(item)
        if item.question in self.fail_for:
            raise JudgeError("quota exhausted")
        return self.verdicts.get(item.question, Verdict(True, True if item.sources else None, "ok", 100))


def section(
    kind="document",
    status="answered",
    question="What was revenue?",
    answer="Revenue was 12,563 crore.",
    sources=None,
):
    if sources is None:
        sources = [
            {"chunk_id": "d1:1:0", "page_label": "1", "pdf_page": 1, "text": "Revenue was 12,563 crore."}
        ]
    return {"kind": kind, "status": status, "question": question, "answer": answer, "sources": sources}


@pytest.fixture
def store(settings):
    init_db(settings.sqlite_path)
    return RequestStore(settings.sqlite_path)


def logged(store, trace_id, *, status="answered", sections=None, question="What was revenue?", ts=None):
    store.log({"trace_id": trace_id, "question_len": len(question), "route": "DOCUMENT", "status": status})
    if ts:
        conn = sqlite3.connect(store.path)
        conn.execute("UPDATE requests SET ts = ? WHERE trace_id = ?", (ts, trace_id))
        conn.commit()
        conn.close()
    if sections is not None:
        store.log_content(trace_id, question, sections)


# ---------------------------------------------------------------- the store


def test_content_is_kept_apart_from_the_request_row(store):
    logged(store, "t1", sections=[section()])
    assert "question" not in store.get("t1") and "sections" not in store.get("t1")
    [row] = store.recent_answered(10)
    assert row["trace_id"] == "t1" and row["question"] == "What was revenue?"
    assert row["sections"][0]["sources"][0]["text"].startswith("Revenue was")


def test_log_content_replaces_the_text_for_a_repeated_trace_id(store):
    logged(store, "t1", sections=[section(answer="old")])
    store.log_content("t1", "What was revenue?", [section(answer="new")])
    assert store.recent_answered(10)[0]["sections"][0]["answer"] == "new"


def test_recent_answered_is_newest_first_answered_only_and_skips_judged_by_default(store):
    logged(store, "old", sections=[section()], ts="2026-10-01T10:00:00.000Z")
    logged(store, "new", sections=[section()], ts="2026-10-03T10:00:00.000Z")
    logged(
        store,
        "abst",
        status="abstained",
        sections=[section(status="abstained", answer=None)],
        ts="2026-10-04T10:00:00.000Z",
    )
    logged(store, "judged", sections=[section()], ts="2026-10-02T10:00:00.000Z")
    store.set_judge("judged", True, True)
    assert [r["trace_id"] for r in store.recent_answered(10)] == ["new", "old"]
    assert [r["trace_id"] for r in store.recent_answered(10, only_unjudged=False)] == ["new", "judged", "old"]
    assert [r["trace_id"] for r in store.recent_answered(1)] == ["new"]


def test_a_request_with_no_stored_text_comes_back_with_no_sections(store):
    logged(store, "t1")
    [row] = store.recent_answered(10)
    assert row["question"] is None and row["sections"] is None


def test_set_judge_touches_only_the_judge_columns_and_never_creates_a_row(store):
    logged(store, "t1")
    store.set_feedback("t1", -1)
    store.set_judge("t1", False, None)
    row = store.get("t1")
    assert (row["judge_correct"], row["judge_grounded"], row["feedback"], row["status"]) == (
        0,
        None,
        -1,
        "answered",
    )
    store.set_judge("ghost", True, True)
    assert store.get("ghost") is None
    # and re-logging the request does not erase the verdict (the logger never writes the judge columns)
    store.log({"trace_id": "t1", "question_len": 3, "status": "answered"})
    assert store.get("t1")["judge_correct"] == 0


def test_the_content_table_and_judge_columns_are_added_to_an_older_database(tmp_path):
    path = tmp_path / "old.sqlite"
    init_db(path)
    conn = sqlite3.connect(path)  # make it look like a database from before step 09
    conn.execute("DROP TABLE request_content")
    conn.execute("ALTER TABLE requests DROP COLUMN judge_correct")
    conn.execute("ALTER TABLE requests DROP COLUMN judge_grounded")
    conn.commit()
    conn.close()
    init_db(path)
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    assert "request_content" in tables and {"judge_correct", "judge_grounded"} <= cols


# ---------------------------------------------------------------- judging


def test_judge_recent_scores_the_document_answer_without_a_reference_and_stores_it(store):
    logged(store, "t1", sections=[section()])
    judge = FakeJudge()
    res = judge_recent(store, judge, 10, out=lambda *_: None)
    assert (res.judged, res.correct, res.grounded, res.grounded_n) == (1, 1, 1, 1)
    [item] = judge.items
    assert item.reference is None and item.question == "What was revenue?"
    assert item.answer == "Revenue was 12,563 crore." and item.sources[0].label == "p.1"
    row = store.get("t1")
    assert (row["judge_correct"], row["judge_grounded"]) == (1, 1)


def test_the_metrics_page_reads_the_stored_scores(store, settings):
    for i in range(3):
        logged(store, f"t{i}", sections=[section(question=f"q{i}")])
    judge = FakeJudge({"q0": Verdict(True, True), "q1": Verdict(False, True), "q2": Verdict(True, False)})
    judge_recent(store, judge, 10, out=lambda *_: None)
    q = quality_metrics(load_requests(settings.sqlite_path))
    assert (q["judge_correct"].count, q["judge_correct"].total) == (2, 3)
    assert (q["judge_grounded"].count, q["judge_grounded"].total) == (2, 3)


def test_requests_without_stored_text_and_general_only_answers_are_counted_not_judged(store):
    logged(store, "plain")  # log_content was off
    logged(store, "gen", sections=[section("general", answer=PARIS, sources=[], question="Capital?")])
    logged(store, "ok", sections=[section()])
    res = judge_recent(store, FakeJudge(), 10, out=lambda *_: None)
    assert (res.judged, res.no_content, res.not_document) == (1, 1, 1)
    assert store.get("plain")["judge_correct"] is None and store.get("gen")["judge_correct"] is None
    text = summary(res)
    assert (
        "no stored text" in text
        and "observability.log_content: true" in text
        and "general knowledge only" in text
    )


def test_a_mixed_request_is_judged_on_its_document_part_only(store):
    logged(
        store, "m", sections=[section(), section("general", answer=PARIS, sources=[], question="Capital?")]
    )
    judge = FakeJudge()
    judge_recent(store, judge, 10, out=lambda *_: None)
    assert [i.question for i in judge.items] == ["What was revenue?"]


def test_a_judge_failure_is_counted_and_leaves_the_row_unjudged_for_the_next_run(store):
    logged(store, "a", sections=[section(question="qa")], ts="2026-10-02T10:00:00.000Z")
    logged(store, "b", sections=[section(question="qb")], ts="2026-10-01T10:00:00.000Z")
    res = judge_recent(store, FakeJudge(fail_for=["qa"]), 10, out=lambda *_: None)
    assert (res.judged, res.failed) == (1, 1) and "quota exhausted" in res.details[0]
    assert store.get("a")["judge_correct"] is None and store.get("b")["judge_correct"] == 1
    assert "run the script again" in summary(res)
    res2 = judge_recent(store, FakeJudge(), 10, out=lambda *_: None)  # the retry picks up only "a"
    assert res2.judged == 1 and store.get("a")["judge_correct"] == 1


def test_rejudge_includes_rows_that_already_have_scores(store):
    logged(store, "t1", sections=[section()])
    judge_recent(store, FakeJudge(), 10, out=lambda *_: None)
    assert judge_recent(store, FakeJudge(), 10, out=lambda *_: None).judged == 0
    res = judge_recent(
        store, FakeJudge({"What was revenue?": Verdict(False, False)}), 10, rejudge=True, out=lambda *_: None
    )
    assert res.judged == 1 and store.get("t1")["judge_correct"] == 0


def test_last_n_limits_how_many_are_looked_at(store):
    for i in range(5):
        logged(store, f"t{i}", sections=[section()], ts=f"2026-10-0{i + 1}T10:00:00.000Z")
    res = judge_recent(store, FakeJudge(), 2, out=lambda *_: None)
    assert (
        res.judged == 2 and store.get("t4")["judge_correct"] == 1 and store.get("t0")["judge_correct"] is None
    )


# ---------------------------------------------------------------- the content log in the real app


def test_content_is_logged_only_when_the_flag_is_on(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:  # default: off
        ready_doc(client, tmp_path)
        client.post(
            "/query", json={"question": "What was revenue in FY25?"}, headers={"X-Request-ID": "off-1"}
        )
        assert client.app.state.request_store.recent_answered(10, only_unjudged=False)[0]["sections"] is None

    settings.observability.log_content = True
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        client.post(
            "/query", json={"question": "What was revenue in FY25?"}, headers={"X-Request-ID": "on-1"}
        )
        rows = {
            r["trace_id"]: r for r in client.app.state.request_store.recent_answered(10, only_unjudged=False)
        }
    sec = rows["on-1"]["sections"][0]
    assert rows["off-1"]["sections"] is None
    assert rows["on-1"]["question"] == "What was revenue in FY25?"
    assert (
        sec["kind"] == "document"
        and sec["status"] == "answered"
        and sec["answer"] == "Revenue was 12,563 crore."
    )
    [src] = sec["sources"]
    assert (
        src["chunk_id"] and src["page_label"] and "Revenue from operations" in src["text"]
    )  # the FULL chunk, not a snippet


def test_an_abstention_keeps_no_answer_text_and_the_request_row_stays_text_free(
    settings, monkeypatch, tmp_path
):
    settings.observability.log_content = True
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[INSUFFICIENT])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        client.post(
            "/query", json={"question": "What was the FY30 forecast?"}, headers={"X-Request-ID": "ab-1"}
        )
        store = client.app.state.request_store
        conn = sqlite3.connect(store.path)
        sections = json.loads(
            conn.execute("SELECT sections FROM request_content WHERE trace_id='ab-1'").fetchone()[0]
        )
        row = json.dumps(store.get("ab-1"))
    assert (
        sections[0]["status"] == "abstained"
        and sections[0]["answer"] is None
        and sections[0]["sources"] == []
    )
    assert "FY30" not in row and "forecast" not in row


def test_a_broken_content_log_never_fails_the_question(settings, monkeypatch, tmp_path):
    settings.observability.log_content = True
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        monkeypatch.setattr(
            client.app.state.request_store,
            "log_content",
            lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")),
        )
        r = client.post("/query", json={"question": "What is the capital of France?"})
    assert r.status_code == 200 and r.json()["sections"][0]["answer"] == PARIS


def test_cited_texts_never_reach_the_api_response_or_the_debug_json(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        body = client.post("/query", json={"question": "What was revenue in FY25?"}).json()
    assert "cited_texts" not in json.dumps(body)
    assert "cited_texts" not in DocAnswer(status="answered", message="x", cited_texts={"c": "t"}).to_dict()
