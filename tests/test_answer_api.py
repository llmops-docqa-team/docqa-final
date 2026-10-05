"""POST /debug/answer_doc through the real app (fake embedder, real Chroma/SQLite, mocked LLM)."""

from __future__ import annotations

import json

from app.llm.client import LLMError, LLMResponse, Usage
from tests.conftest import FakeEmbedder, make_pdf, upload, wait_for

BODY = "Revenue from operations was 12,563 crore in FY25. " * 3


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(content=reply, usage=Usage(50, 10, 60), model="scripted", latency_ms=5.0)


def answer_json(**kw):
    return json.dumps(
        {"answer": "Revenue was 12,563 crore.", "citations": ["S1"], "status": "ANSWERED", **kw}
    )


def test_answer_doc_endpoint(tmp_path, settings, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    settings.retrieval.theta = -2.0
    llm = ScriptedLLM(answer_json())
    pdf = tmp_path / "r.pdf"
    make_pdf(pdf, [BODY, "Other page about something else entirely."])

    with TestClient(create_app(embedder=FakeEmbedder(), llm=llm)) as client:
        doc_id = upload(client, pdf, "r.pdf").json()["doc_id"]
        wait_for(client, doc_id, ("READY",))

        r = client.post("/debug/answer_doc", json={"question": "What was revenue in FY25?"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "answered" and body["answer"] == "Revenue was 12,563 crore."
        [c] = body["citations"]
        assert c["filename"] == "r.pdf" and c["doc_id"] == doc_id and c["display"].startswith("p.")
        assert body["number_check"] == "pass" and body["tokens"]["total"] == 60 and body["llm_calls"] == 1
        assert set(body["timings"]) >= {"embed_ms", "retrieve_ms", "llm_ms", "total_ms"}
        assert '<source id="S1"' in llm.calls[0][1]["content"]


def test_answer_doc_endpoint_degrades_when_llm_is_down(tmp_path, settings, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    settings.retrieval.theta = -2.0
    pdf = tmp_path / "r.pdf"
    make_pdf(pdf, [BODY])
    with TestClient(create_app(embedder=FakeEmbedder(), llm=ScriptedLLM(LLMError("down")))) as client:
        doc_id = upload(client, pdf, "r.pdf").json()["doc_id"]
        wait_for(client, doc_id, ("READY",))
        body = client.post("/debug/answer_doc", json={"question": "revenue?"}).json()
        assert (
            body["status"] == "error" and body["message"] == "The answer service is temporarily unavailable."
        )
        assert body["fallback_passages"]


def test_answer_doc_endpoint_without_documents(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    llm = ScriptedLLM()
    with TestClient(create_app(embedder=FakeEmbedder(), llm=llm)) as client:
        body = client.post("/debug/answer_doc", json={"question": "revenue?"}).json()
        assert body["status"] == "abstained" and body["abstain_reason"] == "no_documents" and llm.calls == []
        assert client.post("/debug/answer_doc", json={"question": ""}).status_code == 422
