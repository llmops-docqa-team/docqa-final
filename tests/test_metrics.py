"""Metric computations (app/observability/metrics.py) on synthetic rows: percentiles, rates and their
denominators, drift, colour thresholds, and reading the SQLite log."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from app.config import ObservabilityConfig, Threshold
from app.observability import metrics as m
from app.observability.failure_reasons import classify_ingest_failure, classify_upload_rejection
from app.storage.db import init_db

CFG = ObservabilityConfig()


def row(**kw) -> dict:
    """A `requests` row; every column the metrics read is present, default NULL except status."""
    base = dict.fromkeys(
        "trace_id ts question_len route router_ok router_fallback t_router_ms t_embed_ms t_retrieve_ms "
        "t_llm_ms t_total_ms top_score n_sources abstain_reason citations_valid number_check tokens_in "
        "tokens_out cost_usd_equiv error answer_chars feedback judge_correct judge_grounded".split()
    )
    return {**base, "status": "answered", "ts": "2026-10-05T10:00:00.000Z", **kw}


# ---------------------------------------------------------------- statistics


def test_percentile_interpolates_like_numpy():
    xs = [10, 20, 30, 40, 50]
    assert m.percentile(xs, 50) == 30
    assert m.percentile(xs, 0) == 10 and m.percentile(xs, 100) == 50
    assert m.percentile(xs, 25) == 20
    assert m.percentile(xs, 95) == pytest.approx(48.0)
    assert m.percentile([1, 2], 50) == 1.5
    assert m.percentile([7], 99) == 7


def test_percentile_of_nothing_is_none_and_input_order_does_not_matter():
    assert m.percentile([], 50) is None
    assert m.percentile([50, 10, 30], 50) == 30


def test_p99_of_a_hundred_values():
    xs = list(range(1, 101))
    assert m.percentile(xs, 99) == pytest.approx(99.01)
    s = m.summarize([float(x) for x in xs])
    assert s["n"] == 100 and s["p50"] == pytest.approx(50.5) and s["mean"] == pytest.approx(50.5)


def test_summarize_empty():
    assert m.summarize([]) == {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None}


def test_ratio_counts_hits_among_the_rows_that_qualify():
    rows = [row(status="abstained"), row(), row(), row(status="error")]
    r = m.ratio(rows, lambda x: x["status"] == "abstained")
    assert (r.count, r.total, r.value) == (1, 4, 0.25)
    only_answered = m.ratio(
        rows, lambda x: x["status"] == "abstained", of=lambda x: x["status"] == "answered"
    )
    assert (only_answered.count, only_answered.total) == (0, 2)


def test_ratio_with_no_rows_has_no_value_instead_of_dividing_by_zero():
    assert m.ratio([], lambda x: True).value is None


# ---------------------------------------------------------------- operational


def test_stage_latencies_only_count_requests_that_ran_the_stage():
    rows = [
        row(t_total_ms=1000, t_router_ms=100, t_llm_ms=700),
        row(t_total_ms=3000, t_router_ms=None, t_llm_ms=None),  # no router call, abstained before the LLM
        row(t_total_ms=2000, t_router_ms=300, t_llm_ms=1500),
    ]
    lat = m.operational(rows, [])["latency_ms"]
    assert lat["total"]["n"] == 3 and lat["total"]["p50"] == 2000
    assert lat["router"]["n"] == 2 and lat["router"]["p50"] == 200
    assert lat["llm"]["n"] == 2 and lat["embed"]["n"] == 0 and lat["embed"]["p95"] is None


def test_error_rate_tokens_and_cost():
    rows = [
        row(tokens_in=100, tokens_out=50, cost_usd_equiv=0.002),
        row(tokens_in=300, tokens_out=50, cost_usd_equiv=0.004),
        row(status="error", tokens_in=None),
        row(status="abstained", tokens_in=0, tokens_out=0, cost_usd_equiv=0.0),
    ]
    op = m.operational(rows, [])
    assert op["error_rate"].count == 1 and op["error_rate"].total == 4
    assert op["tokens_per_request"] == pytest.approx((150 + 350 + 0) / 3)
    assert op["cost_total_usd"] == pytest.approx(0.006)
    assert op["cost_per_request_usd"] == pytest.approx(0.002)


def test_llm_backend_rates_only_count_requests_that_made_an_llm_call():
    rows = [
        {"status": "answered", "llm_backend": "groq", "llm_fallbacks": 0, "llm_rate_limited": 2},
        {"status": "answered", "llm_backend": "groq+ollama", "llm_fallbacks": 1, "llm_rate_limited": 3},
        {"status": "answered", "llm_backend": "groq", "llm_fallbacks": 0, "llm_rate_limited": 0},
        {"status": "answered", "llm_backend": None, "llm_fallbacks": None, "llm_rate_limited": None},
        {"status": "answered"},  # a row from before step 11
    ]
    op = m.operational(rows, [])
    assert (op["llm_fallback_rate"].count, op["llm_fallback_rate"].total) == (1, 3)
    assert (op["llm_rate_limited_rate"].count, op["llm_rate_limited_rate"].total) == (2, 3)
    assert m.operational([], [])["llm_fallback_rate"].value is None


def test_operational_with_no_rows():
    op = m.operational([], [])
    assert op["tokens_per_request"] is None and op["cost_per_request_usd"] is None
    assert op["cost_total_usd"] == 0 and op["error_rate"].value is None


def doc(name, status="READY", pages=10, secs=20.0, ocr=0, error=None):
    return {
        "filename": name,
        "status": status,
        "pages_total": pages,
        "ingest_seconds": secs,
        "n_ocr_pages": ocr,
        "error": error,
    }


def test_ingestion_seconds_per_page_uses_ready_documents_only():
    docs = [
        doc("a.pdf", pages=10, secs=20),
        doc("b.pdf", pages=20, secs=100, ocr=20),
        doc("c.pdf", status="PROCESSING", pages=50, secs=None),
        doc("d.pdf", status="FAILED", pages=5, secs=1, error="boom"),
    ]
    ing = m.ingestion_speed(docs)
    assert [d["filename"] for d in ing["documents"]] == ["a.pdf", "b.pdf"]
    assert [d["s_per_page"] for d in ing["documents"]] == [2.0, 5.0]
    assert ing["median_s_per_page"] == 3.5
    assert ing["pooled_s_per_page"] == pytest.approx(120 / 30)
    assert ing["documents"][1]["ocr_pages"] == 20


def test_ingestion_speed_with_nothing_finished():
    ing = m.ingestion_speed([doc("c.pdf", status="PROCESSING", secs=None)])
    assert ing["documents"] == [] and ing["median_s_per_page"] is None and ing["pooled_s_per_page"] is None


# ---------------------------------------------------------------- input


def test_route_mix_counts_and_ignores_rows_without_a_route():
    rows = [row(route="DOCUMENT"), row(route="DOCUMENT"), row(route="GENERAL"), row(route=None)]
    assert m.input_metrics(rows, [], [])["route_mix"] == {"DOCUMENT": 2, "GENERAL": 1}


def test_question_length_histogram_buckets_and_overflow():
    rows = [row(question_len=n) for n in (5, 49, 50, 120, 499, 500, 800)]
    hist = {b["bucket"]: b["count"] for b in m.question_length_histogram(rows)}
    assert hist["0-49"] == 2 and hist["50-99"] == 1 and hist["100-149"] == 1
    assert hist["450-499"] == 1 and hist["500+"] == 2  # over-limit lengths pile into the last bucket
    assert sum(hist.values()) == 7


def test_upload_failures_combine_refusals_and_failed_ingestions():
    rejections = [{"reason": "not_a_pdf"}, {"reason": "not_a_pdf"}, {"reason": "too_large"}]
    docs = [
        doc("x.pdf", status="FAILED", error="no text could be extracted from this PDF"),
        doc("y.pdf", status="FAILED", error="3 of 4 pages could not be read (OCR unavailable)"),
        doc("z.pdf", status="FAILED", error="something odd"),
        doc("ok.pdf", status="READY", error="2 pages could not be read"),  # a warning, not a failure
    ]
    out = m.upload_failures_by_reason(rejections, docs)
    assert out == {
        "not_a_pdf": 2,
        "too_large": 1,
        "no_text": 1,
        "pages_unreadable": 1,
        "processing_error": 1,
    }
    assert list(out)[0] == "not_a_pdf"  # most common first


@pytest.mark.parametrize(
    ("status", "detail", "reason"),
    [
        (415, "Not a PDF: the file is empty.", "not_a_pdf"),
        (413, "File is larger than the 25 MB limit.", "too_large"),
        (422, "PDF is password protected / encrypted. Remove the password and re-upload.", "encrypted"),
        (422, "PDF has 900 pages; the limit is 400.", "too_many_pages"),
        (422, "PDF has no pages.", "empty_pdf"),
        (422, "Could not read this PDF: broken xref", "unreadable"),
        (422, "something new", "rejected"),
    ],
)
def test_upload_rejection_reasons(status, detail, reason):
    assert classify_upload_rejection(status, detail) == reason


def test_ingest_failure_reasons():
    assert classify_ingest_failure(None) == "processing_error"
    assert classify_ingest_failure("No text could be extracted from this PDF") == "no_text"


# ---------------------------------------------------------------- output


def test_abstention_rate_and_reasons():
    rows = [
        row(status="abstained", abstain_reason="insufficient"),
        row(status="abstained", abstain_reason="insufficient"),
        row(status="abstained", abstain_reason="low_score"),
        row(),
        row(status="error", abstain_reason="llm_unavailable"),
    ]
    out = m.output_metrics(rows)
    assert (out["abstention_rate"].count, out["abstention_rate"].total) == (3, 5)
    assert out["abstain_reasons"] == {"insufficient": 2, "low_score": 1}  # errors are not abstentions


def test_citation_invalid_rate_ignores_requests_that_never_asked_the_llm():
    rows = [row(citations_valid=1), row(citations_valid=1), row(citations_valid=0), row(citations_valid=None)]
    r = m.output_metrics(rows)["citation_invalid_rate"]
    assert (r.count, r.total) == (1, 3)


def test_number_check_rate_only_counts_checked_answers():
    rows = [
        row(number_check="pass"),
        row(number_check="fail"),
        row(number_check="na"),
        row(number_check=None),
    ]
    r = m.output_metrics(rows)["number_check_fail_rate"]
    assert (r.count, r.total) == (1, 2)


def test_router_fallback_rate_excludes_requests_where_the_router_never_ran():
    rows = [
        row(route="DOCUMENT", router_ok=1, t_router_ms=400),
        row(route="DOCUMENT", router_ok=0, router_fallback="bad_json", t_router_ms=900),
        row(route="DOCUMENT", router_ok=0, router_fallback="llm_unavailable", t_router_ms=None),  # LLM down
        row(route="GENERAL", router_ok=1, t_router_ms=None),  # no documents uploaded: router skipped
        row(route=None, status="error"),  # blew up before routing
    ]
    out = m.output_metrics(rows)
    assert (out["router_fallback_rate"].count, out["router_fallback_rate"].total) == (2, 3)
    assert out["router_fallback_reasons"] == {"bad_json": 1, "llm_unavailable": 1}


def test_answer_length_is_over_answered_requests_only():
    rows = [row(answer_chars=100), row(answer_chars=300), row(status="abstained", answer_chars=0)]
    ac = m.output_metrics(rows)["answer_chars"]
    assert ac["n"] == 2 and ac["p50"] == 200


# ---------------------------------------------------------------- quality


def test_feedback_counts_and_shares():
    rows = [row(feedback=1), row(feedback=1), row(feedback=-1), row(), row(status="abstained", feedback=None)]
    q = m.quality_metrics(rows)
    assert (q["thumbs_up"], q["thumbs_down"]) == (2, 1)
    assert q["thumbs_down_share"].value == pytest.approx(1 / 3)
    assert (q["rated_share_of_answers"].count, q["rated_share_of_answers"].total) == (3, 4)


def test_judge_slot_is_empty_until_step_09_fills_it():
    assert m.quality_metrics([row()])["judge_correct"].total == 0
    rows = [row(judge_correct=1, judge_grounded=1), row(judge_correct=0, judge_grounded=1), row()]
    q = m.quality_metrics(rows)
    assert q["judge_correct"].value == 0.5 and q["judge_grounded"].value == 1.0


# ---------------------------------------------------------------- drift


def test_daily_median_top_score():
    rows = [
        row(ts="2026-10-05T09:00:00.000Z", top_score=0.6),
        row(ts="2026-10-05T11:00:00.000Z", top_score=0.8),
        row(ts="2026-10-05T12:00:00.000Z", top_score=0.7),
        row(ts="2026-10-06T08:00:00.000Z", top_score=0.5),
        row(ts="2026-10-06T09:00:00.000Z", top_score=None),  # general question: no retrieval
    ]
    daily = m.daily_median_top_score(rows)
    assert daily == [
        {"date": "2026-10-05", "median_top_score": 0.7, "n": 3},
        {"date": "2026-10-06", "median_top_score": 0.5, "n": 1},
    ]


def test_route_mix_shift_compares_the_latest_week_with_the_week_before():
    last = [row(route="DOCUMENT", ts="2026-09-29T10:00:00.000Z") for _ in range(4)]  # Tue, week of 09-28
    this = [row(route="GENERAL", ts="2026-10-06T10:00:00.000Z") for _ in range(2)]
    this += [row(route="DOCUMENT", ts="2026-10-07T10:00:00.000Z") for _ in range(2)]  # week of 10-05
    s = m.route_mix_shift(last + this)
    assert s["this_week"] == "2026-10-05" and s["last_week"] == "2026-09-28"
    assert s["mix_last"] == {"DOCUMENT": 1.0} and s["mix_this"] == {"GENERAL": 0.5, "DOCUMENT": 0.5}
    assert s["shift"] == pytest.approx(0.5)  # half of the traffic moved
    assert (s["n_this"], s["n_last"]) == (4, 4)


def test_route_mix_shift_is_zero_for_the_same_mix_and_none_without_a_previous_week():
    same = [
        row(route="DOCUMENT", ts="2026-09-29T10:00:00.000Z"),
        row(route="DOCUMENT", ts="2026-10-06T10:00:00.000Z"),
    ]
    assert m.route_mix_shift(same)["shift"] == 0
    gap = [
        row(route="DOCUMENT", ts="2026-09-15T10:00:00.000Z"),
        row(route="DOCUMENT", ts="2026-10-06T10:00:00.000Z"),
    ]
    assert m.route_mix_shift(gap) is None  # two weeks apart is not "week over week"
    assert m.route_mix_shift([]) is None


def test_weeks_start_on_monday_so_sunday_and_monday_are_different_weeks():
    rows = [
        row(route="DOCUMENT", ts="2026-10-04T23:59:00.000Z"),  # Sunday
        row(route="GENERAL", ts="2026-10-05T00:01:00.000Z"),  # Monday
    ]
    weeks = m.route_mix_by_week(rows)
    assert [w["week"] for w in weeks] == ["2026-09-28", "2026-10-05"]


# ---------------------------------------------------------------- thresholds


def test_indicator_colours_use_strict_comparisons_like_the_spec():
    p95 = Threshold(amber=4000, red=6000)
    assert m.indicator(3000, p95, 10, 5) == m.GREEN
    assert m.indicator(4000, p95, 10, 5) == m.GREEN  # "p95 > 6 s red": exactly 6 s is not red
    assert m.indicator(4001, p95, 10, 5) == m.AMBER
    assert m.indicator(6000, p95, 10, 5) == m.AMBER
    assert m.indicator(6001, p95, 10, 5) == m.RED


def test_abstention_over_forty_percent_is_amber_with_the_default_config():
    rule = CFG.thresholds["abstention_rate"]
    assert m.indicator(0.40, rule, 20, 5) == m.GREEN
    assert m.indicator(0.41, rule, 20, 5) == m.AMBER
    assert m.indicator(0.71, rule, 20, 5) == m.RED


def test_lower_is_worse_flips_the_comparison():
    score = Threshold(amber=0.55, red=0.45, lower_is_worse=True)
    assert m.indicator(0.70, score, 9, 5) == m.GREEN
    assert m.indicator(0.50, score, 9, 5) == m.AMBER
    assert m.indicator(0.40, score, 9, 5) == m.RED


def test_no_colour_without_a_rule_a_value_or_enough_requests():
    rule = Threshold(amber=1, red=2)
    assert m.indicator(None, rule, 50, 5) == m.GREY
    assert m.indicator(9, None, 50, 5) == m.GREY
    assert m.indicator(9, rule, 4, 5) == m.GREY  # a 100% abstention rate from 4 requests is noise
    assert m.indicator(9, rule, 5, 5) == m.RED


def test_amber_only_and_red_only_rules():
    assert m.indicator(0.5, Threshold(amber=0.4), 9, 5) == m.AMBER
    assert m.indicator(0.5, Threshold(red=0.4), 9, 5) == m.RED
    assert m.indicator(0.3, Threshold(red=0.4), 9, 5) == m.GREEN


def test_dashboard_colours_follow_config_thresholds():
    slow = [row(t_total_ms=7000) for _ in range(6)]
    d = m.build_dashboard(slow, [], [], CFG)
    assert d["indicators"]["p95_total_ms"] == {"value": 7000, "n": 6, "color": m.RED}
    tighter = ObservabilityConfig(thresholds={"p95_total_ms": Threshold(amber=8000, red=9000)})
    assert m.build_dashboard(slow, [], [], tighter).get("indicators")["p95_total_ms"]["color"] == m.GREEN


def test_dashboard_on_an_empty_log_does_not_crash_and_shows_no_colour():
    d = m.build_dashboard([], [], [], CFG)
    assert d["n_requests"] == 0
    assert {i["color"] for i in d["indicators"].values()} == {m.GREY}


def test_dashboard_drift_indicators_use_the_latest_day_and_the_weekly_shift():
    rows = [row(ts="2026-10-06T10:00:00.000Z", top_score=0.4, route="DOCUMENT") for _ in range(6)]
    rows += [row(ts="2026-09-29T10:00:00.000Z", top_score=0.7, route="GENERAL") for _ in range(6)]
    ind = m.build_dashboard(rows, [], [], CFG)["indicators"]
    assert ind["median_top_score"]["color"] == m.RED and ind["median_top_score"]["value"] == 0.4
    assert ind["route_mix_shift"]["color"] == m.RED and ind["route_mix_shift"]["value"] == 1.0


# ---------------------------------------------------------------- reading the database


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "log.sqlite"
    init_db(path)
    return path


def insert(path, table, **cols):
    conn = sqlite3.connect(path)
    conn.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        list(cols.values()),
    )
    conn.commit()
    conn.close()


def test_a_missing_database_or_table_gives_empty_lists(tmp_path):
    missing = tmp_path / "nope.sqlite"
    assert m.load_requests(missing) == [] and m.load_documents(missing) == []
    assert m.load_upload_failures(missing) == []
    assert not missing.exists()  # reading never creates the file
    bare = tmp_path / "bare.sqlite"
    sqlite3.connect(bare).close()  # a database with no tables yet
    assert m.load_requests(bare) == []


def test_feedback_only_rows_are_not_requests(db):
    insert(db, "requests", trace_id="real", status="answered", route="DOCUMENT", t_total_ms=900)
    insert(db, "requests", trace_id="ghost", feedback=1)  # a 👍 for a trace id that was never logged
    assert [r["trace_id"] for r in m.load_requests(db)] == ["real"]


def test_time_range_filters_on_ts(db):
    insert(db, "requests", trace_id="old", status="answered", ts="2026-10-01T10:00:00.000Z")
    insert(db, "requests", trace_id="new", status="answered", ts="2026-10-05T10:00:00.000Z")
    now = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    assert m.since_iso("All time", now) is None
    assert m.since_iso("Last 24 hours", now) == "2026-10-04T12:00:00.000Z"
    assert [r["trace_id"] for r in m.load_requests(db, m.since_iso("Last 24 hours", now))] == ["new"]
    assert [r["trace_id"] for r in m.load_requests(db, m.since_iso("Last 7 days", now))] == ["old", "new"]
    assert {r["trace_id"] for r in m.load_requests(db)} == {"old", "new"}


def test_documents_and_upload_failures_are_read_with_the_same_range(db):
    insert(db, "documents", id="d1", filename="a.pdf", sha256="x", created_at="2026-10-01T00:00:00.000Z")
    insert(db, "documents", id="d2", filename="b.pdf", sha256="y", created_at="2026-10-05T00:00:00.000Z")
    insert(db, "upload_failures", reason="not_a_pdf", status_code=415, ts="2026-10-05T01:00:00.000Z")
    insert(db, "upload_failures", reason="too_large", status_code=413, ts="2026-10-01T01:00:00.000Z")
    since = "2026-10-04T00:00:00.000Z"
    assert [d["id"] for d in m.load_documents(db, since)] == ["d2"]
    assert [u["reason"] for u in m.load_upload_failures(db, since)] == ["not_a_pdf"]
    assert len(m.load_documents(db)) == 2
