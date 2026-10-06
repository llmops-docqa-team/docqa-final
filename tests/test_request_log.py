"""The request log: one `requests` row per /query (real app, fake embedder, mocked LLM), logging never
failing a request, feedback and log writes not overwriting each other, upload failures counted."""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.config import ObservabilityConfig, Rate
from app.llm.client import LLMError
from app.observability.request_log import cost_usd, error_record, overall_status
from app.storage.db import init_db
from tests.conftest import make_pdf, upload
from tests.fakes import FakeLLM
from tests.test_query_api import (
    INSUFFICIENT,
    PARIS,
    answer_json,
    ready_doc,
    route_json,
    running_app,
)


def logged(client, trace_id) -> dict:
    row = client.app.state.request_store.get(trace_id)
    assert row is not None, f"no request row for {trace_id}"
    return row


def ask(client, question, trace_id):
    r = client.post("/query", json={"question": question}, headers={"X-Request-ID": trace_id})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- one row per query


def test_a_document_answer_is_logged_with_every_design_15_field(settings, monkeypatch, tmp_path):
    question = "What was revenue in FY25?"
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        body = ask(client, question, "t-doc")
        row = logged(client, "t-doc")

    assert row["route"] == "DOCUMENT" and row["router_ok"] == 1 and row["router_fallback"] is None
    assert row["status"] == "answered" and row["abstain_reason"] is None and row["error"] is None
    assert row["question_len"] == len(question)
    assert row["citations_valid"] == 1 and row["number_check"] == "pass"
    assert row["n_sources"] == 2 and row["top_score"] is not None  # both pages of the document were retrieved
    assert row["tokens_in"] == 100 and row["tokens_out"] == 20  # router + document, 50/10 each
    assert row["cost_usd_equiv"] > 0
    # (t_embed_ms is not asserted: the fake embedder is instant, and a 0 reading is stored as "did not run")
    for col in ("t_router_ms", "t_retrieve_ms", "t_llm_ms", "t_total_ms"):
        assert row[col] is not None and row[col] > 0, col
    assert row["t_total_ms"] == pytest.approx(body["timings"]["total_ms"], abs=0.1)
    assert row["answer_chars"] == len("Revenue was 12,563 crore.")
    assert row["app_version"]
    models = json.loads(row["model_ids"])
    assert models == {"router": "fake-router", "document": "fake-doc"}
    versions = json.loads(row["prompt_versions"])
    assert set(versions) == {"router", "document"} and all(v for v in versions.values())
    assert row["feedback"] is None and row["ts"]


def test_the_row_holds_chunk_ids_but_no_question_or_document_text(settings, monkeypatch, tmp_path):
    question = "What was revenue in FY25 for the unmistakable zebra division?"
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        doc_id = ready_doc(client, tmp_path)
        body = ask(client, question, "t-priv")
        row = logged(client, "t-priv")
    chunk_ids = json.loads(row["cited_chunks"])
    assert chunk_ids == [body["sections"][0]["citations"][0]["chunk_id"]]
    assert chunk_ids[0].startswith(doc_id)
    everything = json.dumps(row, default=str).lower()
    assert "zebra" not in everything  # the question is never stored
    assert "revenue from operations" not in everything  # nor document text
    assert "12,563" not in everything  # nor the answer


def test_no_documents_is_a_general_row_with_stages_that_did_not_run_left_null(settings, monkeypatch):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        ask(client, "What is the capital of France?", "t-gen")
        row = logged(client, "t-gen")
    assert row["route"] == "GENERAL" and row["router_ok"] == 1 and row["status"] == "answered"
    assert row["t_router_ms"] is None  # skipped: no documents, no router call
    assert row["t_embed_ms"] is None and row["t_retrieve_ms"] is None
    assert row["t_llm_ms"] > 0
    assert row["top_score"] is None and row["n_sources"] is None
    assert row["citations_valid"] is None and row["number_check"] is None and row["cited_chunks"] is None
    assert json.loads(row["model_ids"]) == {"general": "fake-general"}
    assert row["tokens_in"] == 50 and row["answer_chars"] == len(PARIS)


def test_abstention_is_logged_with_its_reason_and_zero_answer_length(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[INSUFFICIENT])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        ask(client, "What was revenue in FY2030?", "t-abs")
        row = logged(client, "t-abs")
    assert row["status"] == "abstained" and row["abstain_reason"] == "insufficient"
    assert row["answer_chars"] == 0 and row["top_score"] is not None


def test_gate_one_abstention_has_no_llm_time(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        settings.retrieval.theta = 2.0  # nothing can pass gate 1
        ask(client, "What was revenue in FY25?", "t-low")
        row = logged(client, "t-low")
    assert row["status"] == "abstained" and row["abstain_reason"] == "low_score"
    assert row["t_llm_ms"] is None and row["t_retrieve_ms"] > 0
    assert row["tokens_in"] == 50  # only the router call cost tokens


def test_mixed_takes_the_worst_status_and_the_slowest_llm_call(settings, monkeypatch, tmp_path):
    router = route_json("MIXED", "What was revenue in FY2030?", "What is the capital of France?")
    llm = FakeLLM(settings, router=[router], doc=[INSUFFICIENT], general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        ask(client, "FY2030 revenue and the capital of France?", "t-mix")
        row = logged(client, "t-mix")
    assert row["route"] == "MIXED"
    assert row["status"] == "abstained"  # the document half abstained, the general half answered
    assert row["abstain_reason"] == "insufficient"
    assert row["answer_chars"] == len(PARIS)  # only the answered half counts
    assert row["tokens_in"] == 150  # router + document + general
    assert set(json.loads(row["model_ids"])) == {"router", "document", "general"}


def test_a_failed_half_makes_the_request_an_error(settings, monkeypatch, tmp_path):
    router = route_json("MIXED", "What was revenue in FY25?", "What is the capital of France?")
    llm = FakeLLM(settings, router=[router], doc=[answer_json()], general=[LLMError("down", 503)])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        ask(client, "revenue FY25 and the capital of France?", "t-half")
        row = logged(client, "t-half")
    assert row["status"] == "error" and row["error"] == "llm_unavailable"
    assert row["number_check"] == "pass"  # the document half's details are still there


def test_llm_outage_on_the_document_path_is_an_error_row(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[LLMError("down", 503)])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        ask(client, "What was revenue in FY25?", "t-down")
        row = logged(client, "t-down")
    assert row["status"] == "error" and row["abstain_reason"] == "llm_unavailable"
    assert row["error"] == "llm_unavailable"


def test_router_fallback_is_logged_for_the_fallback_metric(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=["garbage", "still garbage"], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        ask(client, "What was revenue in FY25?", "t-fb")
        row = logged(client, "t-fb")
    assert row["route"] == "DOCUMENT" and row["router_ok"] == 0 and row["router_fallback"] == "bad_json"
    assert row["status"] == "answered"


def test_invalid_citations_are_logged(settings, monkeypatch, tmp_path):
    reply = answer_json(citations=["S1", "S99"])  # S99 was never sent: dropped, but S1 keeps the answer
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[reply])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        ask(client, "What was revenue in FY25?", "t-cite")
        row = logged(client, "t-cite")
    assert row["status"] == "answered" and row["citations_valid"] == 0


def test_a_repeated_trace_id_updates_the_row_instead_of_adding_one(settings, monkeypatch):
    llm = FakeLLM(settings, general=[PARIS, "Berlin"])
    with running_app(settings, monkeypatch, llm) as client:
        ask(client, "Capital of France?", "same")
        ask(client, "Capital of Germany? Please tell me.", "same")
        n = sqlite3.connect(settings.sqlite_path).execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert n == 1 and logged(client, "same")["question_len"] == len("Capital of Germany? Please tell me.")


# ---------------------------------------------------------------- logging must never fail a request


def test_a_broken_log_does_not_fail_the_query(settings, monkeypatch, capsys):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:

        def boom(record):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(client.app.state.request_store, "log", boom)
        r = client.post("/query", json={"question": "Capital of France?"}, headers={"X-Request-ID": "t-x"})
    assert r.status_code == 200 and r.json()["sections"][0]["answer"] == PARIS
    out = capsys.readouterr().out
    warn = [json.loads(ln) for ln in out.splitlines() if '"request_log_failed"' in ln]
    assert warn and warn[0]["error_type"] == "OperationalError"
    assert "Capital of France" not in out  # the question is not in the log line either


def test_a_record_that_cannot_be_built_does_not_fail_the_query(settings, monkeypatch):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        monkeypatch.setattr(
            "app.answering.query.build_record", lambda *a, **kw: (_ for _ in ()).throw(KeyError("oops"))
        )
        assert client.post("/query", json={"question": "Capital of France?"}).status_code == 200


def test_a_missing_database_file_does_not_fail_the_query(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        client.app.state.request_store.path = tmp_path / "no-such-dir" / "x" / "gone.sqlite"
        client.app.state.request_store.path.parent.mkdir(parents=True)
        # a directory where the file should be: every connect fails
        client.app.state.request_store.path.mkdir()
        assert client.post("/query", json={"question": "Capital of France?"}).status_code == 200


def test_a_crash_before_any_response_still_leaves_an_error_row(settings, monkeypatch):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:

        def boom(*a, **kw):
            raise RuntimeError("router exploded with the question text inside")

        monkeypatch.setattr(client.app.state.query_service.router, "route", boom)
        with pytest.raises(RuntimeError):
            client.post(
                "/query", json={"question": "Capital of France?"}, headers={"X-Request-ID": "t-crash"}
            )
        row = logged(client, "t-crash")
    assert row["status"] == "error" and row["error"] == "RuntimeError"  # the class name, not the message
    assert row["route"] is None and row["t_total_ms"] > 0 and row["question_len"] == len("Capital of France?")


# ---------------------------------------------------------------- feedback and the log share a row


def test_feedback_after_the_log_keeps_the_logged_fields(settings, monkeypatch):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        ask(client, "Capital of France?", "t-fbk")
        assert client.post("/feedback", json={"trace_id": "t-fbk", "value": -1}).status_code == 200
        row = logged(client, "t-fbk")
    assert row["feedback"] == -1 and row["route"] == "GENERAL" and row["status"] == "answered"


def test_a_log_write_never_erases_feedback(settings, monkeypatch):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        store = client.app.state.request_store
        store.set_feedback("t-early", 1)  # 👍 landed first (the row did not exist yet)
        ask(client, "Capital of France?", "t-early")
        row = logged(client, "t-early")
    assert row["feedback"] == 1 and row["status"] == "answered"


def test_the_store_rejects_unknown_columns_and_feedback(settings):
    init_db(settings.sqlite_path)
    from app.storage.requests import RequestStore

    store = RequestStore(settings.sqlite_path)
    for bad in (
        {"trace_id": "a", "feedback": 1},
        {"trace_id": "a", "x; DROP TABLE requests": 1},
        {"status": "x"},
    ):
        with pytest.raises(ValueError):
            store.log(bad)
    assert store.get("a") is None


# ---------------------------------------------------------------- upload failures


def failures(client) -> list[tuple]:
    conn = sqlite3.connect(client.app.state.settings.sqlite_path)
    return conn.execute("SELECT status_code, reason FROM upload_failures ORDER BY id").fetchall()


def test_refused_uploads_are_counted_by_reason(settings, monkeypatch, tmp_path):
    settings.upload.max_pages = 2
    llm = FakeLLM(settings)
    with running_app(settings, monkeypatch, llm) as client:
        assert upload(client, b"just some text", "notes.pdf").status_code == 415
        big = tmp_path / "big.pdf"
        make_pdf(big, 3)
        assert upload(client, big, "big.pdf").status_code == 422
        settings.upload.max_mb = 0
        assert upload(client, big, "big2.pdf").status_code == 413
        rows = failures(client)
    assert rows == [(415, "not_a_pdf"), (422, "too_many_pages"), (413, "too_large")]


def test_a_good_upload_and_a_duplicate_are_not_failures(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings)
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        pdf = tmp_path / "r.pdf"
        assert upload(client, pdf, "r.pdf").status_code == 200  # duplicate
        assert failures(client) == []


def test_a_broken_upload_failure_log_does_not_change_the_response(settings, monkeypatch):
    llm = FakeLLM(settings)
    with running_app(settings, monkeypatch, llm) as client:

        def boom(*a):
            raise sqlite3.OperationalError("locked")

        monkeypatch.setattr(client.app.state.request_store, "log_upload_failure", boom)
        r = upload(client, b"not a pdf", "x.pdf")
    assert r.status_code == 415 and "Not a PDF" in r.json()["detail"]


# ---------------------------------------------------------------- pure helpers and the schema


def test_overall_status_is_the_worst_one():
    assert overall_status(["answered"]) == "answered"
    assert overall_status(["answered", "abstained"]) == "abstained"
    assert overall_status(["abstained", "not_ready"]) == "not_ready"
    assert overall_status(["answered", "error", "abstained"]) == "error"
    assert overall_status([]) == "error"  # nothing ran: that is a failure
    assert overall_status(["something-new"]) == "error"


def test_cost_uses_per_model_rates_and_falls_back_to_default():
    cfg = ObservabilityConfig(
        pricing_per_mtok={"big": Rate(input=1.0, output=4.0), "default": Rate(input=0.5, output=0.5)}
    )
    assert cost_usd(cfg, [("big", 1_000_000, 500_000)]) == pytest.approx(3.0)
    assert cost_usd(cfg, [("other", 1_000_000, 0), (None, 0, 2_000_000)]) == pytest.approx(1.5)
    assert cost_usd(cfg, []) == 0
    assert cost_usd(ObservabilityConfig(pricing_per_mtok={}), [("big", 10, 10)]) == 0


def test_error_record_is_short_and_has_the_columns_the_store_accepts(settings):
    init_db(settings.sqlite_path)
    from app.storage.requests import RequestStore

    rec = error_record("t-e", question_len=12, error="X" * 500, total_ms=42.04, app_version="abc")
    assert len(rec["error"]) == 200 and rec["status"] == "error" and rec["t_total_ms"] == 42.0
    store = RequestStore(settings.sqlite_path)
    store.log(rec)
    assert store.get("t-e")["status"] == "error"


def test_an_older_database_gets_the_new_columns(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE requests (trace_id TEXT PRIMARY KEY, ts TEXT, status TEXT, feedback INTEGER);"
        "INSERT INTO requests (trace_id, status, feedback) VALUES ('keep', 'answered', 1);"
    )
    conn.commit()
    conn.close()
    init_db(path)
    init_db(path)  # idempotent
    conn = sqlite3.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    assert {"app_version", "router_fallback", "answer_chars", "cited_chunks", "judge_correct"} <= cols
    assert conn.execute("SELECT feedback FROM requests WHERE trace_id='keep'").fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM upload_failures").fetchone() == (0,)
