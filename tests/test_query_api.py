"""POST /query through the real app (fake embedder, real Chroma/SQLite, mocked LLM)."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.answering.document import DocAnswer
from app.answering.general import GENERAL_LABEL
from app.answering.query import NO_DOCS_NOTICE, SPECIFIC_HINT, _document_section
from app.llm.client import LLMError
from app.main import create_app
from tests.conftest import FakeEmbedder, make_pdf, upload, wait_for
from tests.fakes import FakeLLM

BODY = "Revenue from operations was 12,563 crore in FY25. " * 3
PARIS = "Paris is the capital of France."


def route_json(route, doc_q="", gen_q=""):
    return json.dumps({"route": route, "document_question": doc_q, "general_question": gen_q})


def answer_json(**kw):
    reply = {"answer": "Revenue was 12,563 crore.", "citations": ["S1"], "status": "ANSWERED", **kw}
    return json.dumps(reply)


INSUFFICIENT = json.dumps({"answer": "", "citations": [], "status": "INSUFFICIENT"})


@contextmanager
def running_app(settings, monkeypatch, llm):
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    settings.retrieval.theta = -2.0  # the fake embedder's cosine scores mean nothing
    with TestClient(create_app(embedder=FakeEmbedder(), llm=llm)) as client:
        yield client


def ready_doc(client, tmp_path, name="r.pdf") -> str:
    pdf = tmp_path / name
    make_pdf(pdf, [BODY, "Other page about something else entirely."])
    doc_id = upload(client, pdf, name).json()["doc_id"]
    wait_for(client, doc_id, ("READY",))
    return doc_id


def add_doc_row(client, filename, status, done=0, total=10, error=None) -> str:
    """A document row that is not queued anywhere (so it stays in `status`)."""
    store = client.app.state.store
    doc_id = f"fake-{filename}"
    assert store.insert(doc_id, filename, f"sha-{filename}", total)
    store.update(doc_id, status=status, pages_done=done, error=error)
    return doc_id


# ---------------------------------------------------------------- routes


def test_no_documents_skips_the_router_and_answers_general_with_the_label(settings, monkeypatch):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        r = client.post(
            "/query", json={"question": "What is the capital of France?"}, headers={"X-Request-ID": "t-1"}
        )
    assert r.status_code == 200
    body = r.json()
    assert body["trace_id"] == "t-1" and r.headers["X-Request-ID"] == "t-1"
    assert body["route"] == "GENERAL" and body["router_ok"] is True
    assert body["router"]["skipped"] is True and body["router"]["model"] is None
    assert llm.calls_for("router") == [] and len(llm.calls_for("general")) == 1
    [section] = body["sections"]
    assert section["kind"] == "general" and section["status"] == "answered" and section["answer"] == PARIS
    assert section["label"] == GENERAL_LABEL and section["citations"] == []
    assert section["warnings"] == [NO_DOCS_NOTICE]
    assert body["tokens"]["router"]["total"] == 0 and body["tokens"]["total"]["total"] == 60


def test_document_route_answers_with_citations(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        doc_id = ready_doc(client, tmp_path)
        body = client.post("/query", json={"question": "What was revenue in FY25?"}).json()
    assert body["route"] == "DOCUMENT" and body["router_ok"] is True and body["router"]["model"] is not None
    [section] = body["sections"]
    assert section["kind"] == "document" and section["status"] == "answered"
    assert section["answer"] == "Revenue was 12,563 crore." and section["label"] is None
    assert section["question"] == "What was revenue in FY25?" and section["number_check"] == "pass"
    [c] = section["citations"]
    assert c["doc_id"] == doc_id and c["filename"] == "r.pdf" and c["display"].startswith("p.")
    assert section["warnings"] == [] and section["hint"] is None
    assert llm.calls_for("general") == []  # the general path did not run
    t = body["timings"]
    assert set(t) == {
        "router_ms", "embed_ms", "retrieve_ms", "document_llm_ms", "document_ms", "general_ms", "total_ms"
    }
    assert t["general_ms"] == 0.0 and t["total_ms"] >= t["document_ms"] > 0
    assert body["tokens"]["router"]["total"] == 60 and body["tokens"]["document"]["total"] == 60
    assert body["tokens"]["total"]["total"] == 120
    # the router saw the document title and its state
    assert "r.pdf (READY)" in llm.calls_for("router")[0]["messages"][1]["content"]


def test_general_route_with_documents_uploaded(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=[route_json("GENERAL")], general=["EBITDA means ..."])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        body = client.post("/query", json={"question": "What is EBITDA?"}).json()
    assert body["route"] == "GENERAL"
    [section] = body["sections"]
    assert section["kind"] == "general" and section["label"] == GENERAL_LABEL
    assert section["warnings"] == []  # documents exist, so no "upload a PDF" notice
    assert llm.calls_for("doc") == []


def test_mixed_runs_both_paths_in_parallel_and_composes_two_sections(settings, monkeypatch, tmp_path):
    router = route_json("MIXED", "What was revenue in FY25?", "What is the capital of France?")
    # the barrier only releases when the doc call and the general call are in flight at the same time
    llm = FakeLLM(
        settings, router=[router], doc=[answer_json()], general=[PARIS], barrier=threading.Barrier(2)
    )
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        body = client.post(
            "/query", json={"question": "What was revenue in FY25 and what is the capital of France?"}
        ).json()
    assert body["route"] == "MIXED" and body["router_ok"] is True
    doc, gen = body["sections"]  # document first, general second
    assert (doc["kind"], doc["status"]) == ("document", "answered")
    assert doc["answer"] == "Revenue was 12,563 crore."
    assert doc["question"] == "What was revenue in FY25?"
    assert (gen["kind"], gen["status"], gen["answer"]) == ("general", "answered", PARIS)
    assert gen["question"] == "What is the capital of France?" and gen["label"] == GENERAL_LABEL
    # each path got only its own sub-question
    assert "Question: What was revenue in FY25?" in llm.calls_for("doc")[0]["messages"][1]["content"]
    assert llm.calls_for("general")[0]["messages"][1]["content"] == "Question: What is the capital of France?"
    assert body["tokens"]["total"]["total"] == 180
    assert body["router"]["document_question"] == "What was revenue in FY25?"


def test_mixed_document_part_can_abstain_while_the_general_part_answers(settings, monkeypatch, tmp_path):
    router = route_json("MIXED", "What was revenue in FY2030?", "What is the capital of France?")
    llm = FakeLLM(settings, router=[router], doc=[INSUFFICIENT], general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        body = client.post("/query", json={"question": "FY2030 revenue and the capital of France?"}).json()
    doc, gen = body["sections"]
    assert doc["status"] == "abstained" and doc["abstain_reason"] == "insufficient"
    assert "I won't guess" in doc["answer"] and doc["hint"] == SPECIFIC_HINT and doc["closest_pages"]
    assert gen["status"] == "answered" and gen["answer"] == PARIS


def test_mixed_general_part_failing_does_not_hurt_the_document_part(settings, monkeypatch, tmp_path):
    router = route_json("MIXED", "What was revenue in FY25?", "What is the capital of France?")
    llm = FakeLLM(settings, router=[router], doc=[answer_json()], general=[LLMError("down", 503)])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        resp = client.post("/query", json={"question": "revenue FY25 and the capital of France?"})
    assert resp.status_code == 200
    doc, gen = resp.json()["sections"]
    assert doc["status"] == "answered"
    assert gen["status"] == "error" and gen["abstain_reason"] == "llm_unavailable"
    assert gen["answer"] == "The answer service is temporarily unavailable." and gen["label"] == GENERAL_LABEL


def test_an_unexpected_crash_in_one_path_becomes_an_error_section(settings, monkeypatch, tmp_path):
    router = route_json("MIXED", "What was revenue in FY25?", "What is the capital of France?")
    llm = FakeLLM(settings, router=[router], general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)

        def boom(*a, **kw):
            raise RuntimeError("chroma exploded")

        monkeypatch.setattr(client.app.state.doc_answerer, "answer", boom)
        resp = client.post("/query", json={"question": "revenue FY25 and the capital of France?"})
    assert resp.status_code == 200
    doc, gen = resp.json()["sections"]
    assert doc["status"] == "error" and doc["abstain_reason"] == "internal_error"
    assert "chroma" not in doc["answer"]  # internals are not shown to the user
    assert gen["status"] == "answered"


# ---------------------------------------------------------------- router failure


def test_router_fallback_goes_to_document_and_is_visible_in_the_response(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=["garbage", "still garbage"], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        body = client.post("/query", json={"question": "What was revenue in FY25?"}).json()
    assert body["route"] == "DOCUMENT" and body["router_ok"] is False
    assert body["router"]["ok"] is False and body["router"]["fallback_reason"] == "bad_json"
    assert body["sections"][0]["status"] == "answered"
    assert body["tokens"]["router"]["total"] == 120 and llm.calls_for("general") == []


# ---------------------------------------------------------------- documents not ready (design §11)


def test_processing_document_gets_the_pages_message(settings, monkeypatch):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "big.pdf", "PROCESSING", done=140, total=312)
        body = client.post("/query", json={"question": "What was revenue?"}).json()
    [section] = body["sections"]
    assert section["status"] == "not_ready" and section["abstain_reason"] == "documents_not_ready"
    assert "big.pdf is still processing (140/312 pages)" in section["answer"]
    assert llm.calls_for("doc") == []  # nothing to read, so no answer call
    assert "big.pdf (processing 140/312 pages)" in llm.calls_for("router")[0]["messages"][1]["content"]


def test_failed_document_message_includes_the_reason(settings, monkeypatch):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "scan.pdf", "FAILED", error="no text could be extracted from this PDF")
        [section] = client.post("/query", json={"question": "What was revenue?"}).json()["sections"]
    assert section["status"] == "not_ready"
    assert "scan.pdf couldn't be processed: no text could be extracted from this PDF" in section["answer"]


def test_mixed_with_a_processing_document_still_answers_the_general_part(settings, monkeypatch):
    router = route_json("MIXED", "What was revenue?", "What is the capital of France?")
    llm = FakeLLM(settings, router=[router], general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "big.pdf", "PROCESSING", done=3, total=9)
        resp = client.post("/query", json={"question": "revenue, and the capital of France?"})
        doc, gen = resp.json()["sections"]
    assert doc["status"] == "not_ready" and "3/9 pages" in doc["answer"]
    assert gen["status"] == "answered"


def test_partial_document_abstention_carries_the_coverage_note(settings, monkeypatch):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "big.pdf", "PARTIAL", done=20, total=312)  # queryable, but nothing indexed here
        [section] = client.post("/query", json={"question": "What was revenue?"}).json()["sections"]
    assert section["status"] == "abstained" and section["abstain_reason"] == "no_results"
    assert section["coverage_note"] == "big.pdf: searched 20 of 312 pages; the rest is still processing."
    assert section["coverage_note"] in section["answer"] and section["hint"] == SPECIFIC_HINT


def test_vague_question_with_weak_evidence_abstains_with_a_hint(settings, monkeypatch, tmp_path):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[INSUFFICIENT])
    with running_app(settings, monkeypatch, llm) as client:
        ready_doc(client, tmp_path)
        [section] = client.post("/query", json={"question": "tell me about it"}).json()["sections"]
    assert section["status"] == "abstained" and section["hint"] == SPECIFIC_HINT
    assert section["label"] is None


def test_answer_from_a_partial_document_warns_about_coverage():
    docs = [
        {"filename": "big.pdf", "status": "PARTIAL", "pages_done": 20, "pages_total": 312},
        {"filename": "done.pdf", "status": "READY", "pages_done": 5, "pages_total": 5},
    ]
    da = DocAnswer(
        status="answered", message="It was 1.", answer="It was 1.", searched=["big.pdf", "done.pdf"],
        notices=["other.pdf is still processing (1/9 pages). Ask again when it's Ready."],
        number_warning=True, unmatched_numbers=["366"], dropped_citations=["S9"],
    )
    section = _document_section("q?", da, docs)
    assert section.warnings == [
        "other.pdf is still processing (1/9 pages). Ask again when it's Ready.",
        "big.pdf is still processing (20/312 pages): this answer only uses the pages indexed so far "
        "and may be incomplete or wrong. Ask again when the document is Ready.",
        "⚠ number not found verbatim in source: 366",
        "Some citations did not match the retrieved sources and were dropped.",
    ]


# ---------------------------------------------------------------- validation and flags


def test_question_length_limit_gives_422(settings, monkeypatch):
    llm = FakeLLM(settings, general=[PARIS])
    with running_app(settings, monkeypatch, llm) as client:
        too_long = client.post("/query", json={"question": "x" * (settings.query.max_question_chars + 1)})
        assert too_long.status_code == 422 and "too long" in too_long.json()["detail"]
        assert client.post("/query", json={"question": "   "}).status_code == 422
        assert client.post("/query", json={}).status_code == 422
        assert llm.calls == []  # rejected before any LLM call
        at_limit = client.post("/query", json={"question": "x" * settings.query.max_question_chars})
        assert at_limit.status_code == 200


def test_the_limit_comes_from_config(settings, monkeypatch):
    settings.query.max_question_chars = 20
    llm = FakeLLM(settings)
    with running_app(settings, monkeypatch, llm) as client:
        assert client.post("/query", json={"question": "y" * 21}).status_code == 422


@pytest.mark.parametrize("path", ["/debug/retrieve", "/debug/answer_doc"])
def test_debug_endpoints_are_hidden_unless_the_flag_is_on(settings, monkeypatch, path):
    settings.api.debug_endpoints = False
    with running_app(settings, monkeypatch, FakeLLM(settings)) as client:
        assert client.post(path, json={"question": "x"}).status_code == 404
    settings.api.debug_endpoints = True
    with running_app(settings, monkeypatch, FakeLLM(settings)) as client:
        assert client.post(path, json={"question": "x"}).status_code == 200


def test_debug_env_var_turns_the_flag_on(monkeypatch):
    from app.config import load_settings

    monkeypatch.delenv("DOCQA_DEBUG", raising=False)
    assert load_settings().api.debug_endpoints is False
    monkeypatch.setenv("DOCQA_DEBUG", "1")
    assert load_settings().api.debug_endpoints is True
