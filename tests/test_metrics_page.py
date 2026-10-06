"""The Streamlit Metrics page, headless (`AppTest`): it renders from the SQLite log, shows the five
categories, colours indicators from config thresholds, and copes with an empty or missing database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.storage.db import init_db
from app.storage.requests import RequestStore
from tests.test_ui_smoke import FakeClient, run, texts


def iso(delta: timedelta = timedelta()) -> str:
    return (datetime.now(UTC) - delta).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def record(i: int, **kw) -> dict:
    base = {
        "trace_id": f"t-{i}", "ts": iso(timedelta(minutes=i)), "question_len": 40 + i, "route": "DOCUMENT",
        "router_ok": 1, "t_router_ms": 400, "t_embed_ms": 12, "t_retrieve_ms": 20, "t_llm_ms": 1500,
        "t_total_ms": 2000, "top_score": 0.7, "n_sources": 5, "status": "answered", "citations_valid": 1,
        "number_check": "pass", "tokens_in": 2000, "tokens_out": 200, "cost_usd_equiv": 0.0004,
        "answer_chars": 180,
    }  # fmt: skip
    return {**base, **kw}


@pytest.fixture
def page(settings, monkeypatch):
    """Open the Metrics page against `settings`' (temp) database."""
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    init_db(settings.sqlite_path)

    def open_page():
        at = run(monkeypatch, FakeClient())
        at.switch_page("views/metrics.py").run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    return open_page


def labels(at) -> str:
    return " | ".join(m.label for m in at.metric)


def test_an_empty_log_shows_a_hint_not_a_crash(page):
    at = page()
    assert "No requests logged yet" in texts(at.info)
    assert not at.metric


def test_a_missing_database_file_is_handled_and_not_created(settings, monkeypatch):
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    assert not settings.sqlite_path.exists()
    at = run(monkeypatch, FakeClient())
    at.switch_page("views/metrics.py").run()
    assert not at.exception and "No requests logged yet" in texts(at.info)
    assert not settings.sqlite_path.exists()  # a page view must not create or write the database


def test_the_five_categories_render_from_the_log(settings, page):
    store = RequestStore(settings.sqlite_path)
    for i in range(8):
        store.log(record(i))
    store.log(record(8, status="abstained", abstain_reason="insufficient", t_llm_ms=1200, answer_chars=0))
    store.log(
        record(9, route="GENERAL", top_score=None, n_sources=None, citations_valid=None, number_check=None)
    )
    store.set_feedback("t-0", 1)
    store.set_feedback("t-1", -1)
    at = page()

    assert [h.value for h in at.main.header] == ["Operational", "Input", "Output", "Quality", "Drift"]
    seen = labels(at)
    for expected in (
        "p95 total latency",
        "Error rate",
        "Tokens / request",
        "Abstention rate",
        "Citation-invalid rate",
        "Number-check failures",
        "Router fallback rate",
        "Thumbs-down share",
        "Judge scores",
    ):
        assert expected in seen, expected
    values = {m.label.split(" ", 1)[-1]: m.value for m in at.metric}
    assert values["p95 total latency"] == "2.00 s"
    assert values["Abstention rate"] == "10.0%"  # 1 of 10
    assert values["Error rate"] == "0.0%"
    assert "10 requests" in texts(at.caption)
    assert "No judged answers yet." in texts(at.caption)  # the step 09 slot is there, empty


def test_a_slow_p95_turns_the_indicator_red(settings, page):
    store = RequestStore(settings.sqlite_path)
    for i in range(6):
        store.log(record(i, t_total_ms=9000))
    at = page()
    assert any(m.label.startswith("🔴") and "p95 total latency" in m.label for m in at.metric)
    assert "red > 6.00 s" in texts(at.caption)  # the rule is shown next to the number


def test_a_healthy_p95_is_green_and_too_few_requests_have_no_colour(settings, page):
    store = RequestStore(settings.sqlite_path)
    for i in range(6):
        store.log(record(i))
    at = page()
    assert any(m.label.startswith("🟢") and "p95 total latency" in m.label for m in at.metric)

    store = RequestStore(settings.sqlite_path)
    store.log(record(20, trace_id="x-abs", status="abstained"))
    # one abstention in 7 requests is 14%: within the threshold anyway; the thumbs tile has no ratings at all
    at = page()
    assert any(m.label.startswith("⚪") and "Thumbs-down share" in m.label for m in at.metric)


def test_abstention_over_forty_percent_is_amber(settings, page):
    store = RequestStore(settings.sqlite_path)
    for i in range(10):
        store.log(record(i, status="abstained" if i < 5 else "answered", abstain_reason="insufficient"))
    at = page()
    assert any(m.label.startswith("🟠") and "Abstention rate" in m.label for m in at.metric)


def test_the_time_range_filter_hides_old_requests(settings, page):
    store = RequestStore(settings.sqlite_path)
    store.log(record(1, trace_id="old", ts=iso(timedelta(days=3))))
    at = page()
    assert "1 requests" in texts(at.caption)  # the default range is 7 days
    at.selectbox[0].select("Last hour").run()
    assert not at.exception and "No requests logged yet" in texts(at.info)
    at.selectbox[0].select("All time").run()
    assert "1 requests" in texts(at.caption)


def test_ingestion_and_upload_failures_show_up(settings, page):
    conn_store = RequestStore(settings.sqlite_path)
    conn_store.log(record(1))
    conn_store.log_upload_failure(415, "not_a_pdf")
    import sqlite3

    c = sqlite3.connect(settings.sqlite_path)
    c.execute(
        "INSERT INTO documents (id, filename, sha256, status, pages_total, ingest_seconds) "
        "VALUES ('d1', 'a.pdf', 'x', 'READY', 10, 25.0)"
    )
    c.commit()
    c.close()
    at = page()
    assert "median 2.50 s/page" in texts(at.caption)
    assert len(at.dataframe) >= 2  # the latency table and the per-document ingestion table
