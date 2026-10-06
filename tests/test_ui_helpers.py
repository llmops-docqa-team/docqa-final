"""UI helpers: display formatting and the API client (the Streamlit layout itself is not unit-tested)."""

from __future__ import annotations

import json

import pytest
import requests
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeEmbedder, make_pdf, wait_for
from tests.fakes import FakeLLM
from ui import formatting as fmt
from ui.api_client import ApiClient, ApiError, error_message

# ---------------------------------------------------------------- formatting


def doc(status, done=0, total=10, error=None, name="Report.pdf"):
    return {
        "id": "d",
        "filename": name,
        "status": status,
        "pages_done": done,
        "pages_total": total,
        "error": error,
    }


@pytest.mark.parametrize(
    "d, expected",
    [
        (doc("QUEUED"), "⏳ Queued"),
        (doc("PROCESSING", 3, 10), "⚙️ Processing 3/10"),
        (doc("PARTIAL", 20, 312), "🟡 Partial 20/312 · searchable"),
        (doc("READY", 10, 10), "✅ Ready"),
        (doc("READY", 10, 10, error="2 pages could not be read"), "⚠️ Ready (with a warning)"),
        (doc("FAILED", error="no text"), "❌ Failed"),
        (doc("PROCESSING", 0, None), "⚙️ Processing"),
    ],
)
def test_status_label(d, expected):
    assert fmt.status_label(d) == expected


def test_progress_fraction_only_while_in_progress_and_always_in_range():
    assert fmt.progress_fraction(doc("QUEUED")) is None
    assert fmt.progress_fraction(doc("READY", 10, 10)) is None
    assert fmt.progress_fraction(doc("FAILED")) is None
    assert fmt.progress_fraction(doc("PROCESSING", 5, 10)) == 0.5
    assert fmt.progress_fraction(doc("PARTIAL", 20, 20)) == 1.0
    assert fmt.progress_fraction(doc("PROCESSING", 99, 10)) == 1.0  # never above 1 (st.progress would raise)
    assert fmt.progress_fraction(doc("PROCESSING", 0, None)) == 0.0
    assert fmt.progress_fraction(doc("PROCESSING", 0, 0)) == 0.0


def test_status_detail_shows_the_failure_reason_and_the_ready_warning():
    assert fmt.status_detail(doc("FAILED", error="no text could be extracted")) == (
        "Couldn't be processed: no text could be extracted"
    )
    assert fmt.status_detail(doc("READY", error="2 of 86 pages could not be read")) == (
        "Note: 2 of 86 pages could not be read"
    )
    assert fmt.status_detail(doc("READY")) is None
    assert fmt.status_detail(doc("PROCESSING", error="x")) is None
    assert fmt.status_detail(doc("FAILED", error="  ")) is None


def test_is_busy_is_true_while_anything_is_unfinished():
    assert fmt.is_busy([doc("READY"), doc("PROCESSING")])
    assert fmt.is_busy([doc("QUEUED")]) and fmt.is_busy([doc("PARTIAL")])
    assert not fmt.is_busy([doc("READY"), doc("FAILED")])
    assert not fmt.is_busy([])


def test_citation_label_matches_the_design():
    c = {"filename": "Report.pdf", "display": "p.47 (PDF p.53)", "page_label": "47"}
    assert fmt.citation_label(c) == "Report.pdf · p.47 (PDF p.53)"
    assert fmt.citation_label({"filename": "a.pdf", "display": "p.12", "page_label": "12"}) == "a.pdf · p.12"
    assert (
        fmt.citation_label({"filename": "a.pdf", "page_label": "xii"}) == "a.pdf · p.xii"
    )  # no display field


def test_snippet_text_never_empty():
    assert fmt.snippet_text({"snippet": "  Revenue was 12,563 crore.  "}) == "Revenue was 12,563 crore."
    assert fmt.snippet_text({"snippet": ""}) == "(no text available)"
    assert fmt.snippet_text({}) == "(no text available)"


def test_section_titles():
    assert fmt.section_title({"kind": "document"}) == ":material/description: From your documents"
    assert fmt.section_title({"kind": "general"}) == ":material/public: General knowledge"


def test_shorten():
    assert fmt.shorten("short.pdf", 20) == "short.pdf"
    out = fmt.shorten("a" * 60, 20)
    assert len(out) == 20 and out.endswith("…")


def test_escape_markdown_stops_dollar_signs_turning_into_latex():
    assert fmt.escape_markdown("It cost $5 and $10") == r"It cost \$5 and \$10"
    assert fmt.escape_markdown("₹1,25,630 crore") == "₹1,25,630 crore"


def test_feedback_value_maps_the_thumbs_widget_to_the_api_values():
    assert fmt.feedback_value(1) == 1
    assert fmt.feedback_value(0) == -1
    assert fmt.feedback_value(None) is None


def test_route_caption():
    assert fmt.route_caption({"route": "MIXED", "timings": {"total_ms": 2450}}) == "Routed as mixed · 2.5 s"
    assert fmt.route_caption({"route": "GENERAL"}) == "Routed as general"
    assert fmt.route_caption({}) == ""


def test_extra_notes_do_not_repeat_a_coverage_note_already_in_the_abstention_text():
    cov = "X.pdf: searched 20 of 312 pages; the rest is still processing."
    abstained = {"answer": f"I couldn't find this. {cov}", "coverage_note": cov, "warnings": []}
    assert fmt.extra_notes(abstained) == []
    separate = {"answer": "I couldn't find this.", "coverage_note": cov, "warnings": ["w1"]}
    assert fmt.extra_notes(separate) == ["w1", cov]
    assert fmt.extra_notes({"answer": None, "warnings": None}) == []


def test_source_groups_pick_the_right_list_for_each_situation():
    c = [{"filename": "a.pdf", "display": "p.1", "snippet": "s"}]
    assert fmt.source_groups({"citations": c}) == [("Sources", c)]
    assert fmt.source_groups({"closest_pages": c}) == [("Closest pages", c)]
    assert fmt.source_groups({"fallback_passages": c}) == [("Most relevant passages found", c)]
    assert fmt.source_groups({"citations": [], "closest_pages": []}) == []


# ---------------------------------------------------------------- api client (fake HTTP session)


class Resp:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code = status
        self._body, self._bad = body, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


class FakeSession:
    def __init__(self, result):
        self.result = result  # a Resp, or an exception to raise
        self.calls: list[dict] = []

    def request(self, method, url, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def api(result, base="http://api:8000/"):
    s = FakeSession(result)
    return ApiClient(base, s), s


def test_calls_go_to_the_right_url_with_a_timeout():
    c, s = api(Resp(200, [{"id": "d1"}]))
    assert c.list_documents() == [{"id": "d1"}]
    call = s.calls[0]
    assert (call["method"], call["url"]) == ("GET", "http://api:8000/documents") and call["timeout"] > 0


def test_ask_and_feedback_payloads():
    c, s = api(Resp(200, {"trace_id": "t"}))
    c.ask("What was revenue?")
    c.send_feedback("t", -1)
    assert s.calls[0]["json"] == {"question": "What was revenue?"} and s.calls[0]["url"].endswith("/query")
    assert s.calls[1]["json"] == {"trace_id": "t", "value": -1} and s.calls[1]["url"].endswith("/feedback")
    assert s.calls[0]["timeout"] > s.calls[1]["timeout"]  # a question gets longer than a quick call


def test_ask_sends_enhance_off_only_when_it_is_off():
    c, s = api(Resp(200, {"trace_id": "t"}))
    c.ask("q", company="TCS")
    c.ask("q", company="TCS", enhance=False)
    assert s.calls[0]["json"] == {"question": "q", "company": "TCS"}
    assert s.calls[1]["json"] == {"question": "q", "company": "TCS", "enhance": False}


def test_upload_sends_the_file_as_a_pdf():
    c, s = api(Resp(202, {"doc_id": "d", "duplicate": False}))
    assert c.upload("r.pdf", b"%PDF-1.4")["doc_id"] == "d"
    assert s.calls[0]["files"]["file"] == ("r.pdf", b"%PDF-1.4", "application/pdf")


def test_base_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("DOCQA_API_URL", "http://example:9/")
    assert ApiClient().base_url == "http://example:9"


def test_api_down_is_a_friendly_error_not_a_traceback():
    c, _ = api(requests.ConnectionError("Max retries exceeded with url ... NewConnectionError"))
    with pytest.raises(ApiError) as e:
        c.ask("hi")
    assert e.value.unreachable and e.value.status is None
    assert "Can't reach the DocQA service at http://api:8000" in str(e.value)
    assert "Max retries" not in str(e.value) and "NewConnectionError" not in str(e.value)


def test_a_timeout_is_a_friendly_error():
    c, _ = api(requests.Timeout("read timed out"))
    with pytest.raises(ApiError, match="taking too long"):
        c.ask("hi")


def test_any_other_requests_error_is_friendly_too():
    c, _ = api(requests.exceptions.ChunkedEncodingError("boom"))
    with pytest.raises(ApiError, match="Something went wrong"):
        c.list_documents()


def test_a_5xx_never_shows_internals():
    c, _ = api(Resp(500, {"detail": "Traceback (most recent call last): ..."}))
    with pytest.raises(ApiError) as e:
        c.ask("hi")
    assert e.value.status == 500 and "Traceback" not in str(e.value) and "problem" in str(e.value)


def test_a_5xx_with_an_html_body_is_still_handled():
    c, _ = api(Resp(502, bad_json=True))
    with pytest.raises(ApiError, match="problem"):
        c.list_documents()


def test_4xx_shows_the_apis_readable_detail():
    c, _ = api(
        Resp(
            422, {"detail": "The question is too long (600 characters; the limit is 500). Please shorten it."}
        )
    )
    with pytest.raises(ApiError, match="too long") as e:
        c.ask("x")
    assert e.value.status == 422


def test_error_message_for_422_validation_lists_and_unknown_shapes():
    assert "not accepted" in error_message(422, {"detail": [{"loc": ["body"], "msg": "field required"}]})
    assert "not accepted" in error_message(422, None)
    assert error_message(404, {}) == "That item was not found."
    assert "HTTP 409" in error_message(409, {})


def test_health():
    assert api(Resp(200, {"status": "ok"}))[0].health() is True
    assert api(requests.ConnectionError("x"))[0].health() is False
    assert api(Resp(500, {}))[0].health() is False


# ---------------------------------------------------------------- api client against the real app


@pytest.fixture
def live(settings, monkeypatch):
    """The real app (fake embedder, mocked LLM) behind an ApiClient that talks to it in-process."""
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    settings.retrieval.theta = -2.0
    reply = {"answer": "Revenue was 12,563 crore.", "citations": ["S1"], "status": "ANSWERED"}
    llm = FakeLLM(
        settings,
        router=[json.dumps({"route": "DOCUMENT", "document_question": "", "general_question": ""})],
        doc=[json.dumps(reply)],
    )
    with TestClient(create_app(embedder=FakeEmbedder(), llm=llm)) as tc:
        yield ApiClient("http://testserver", tc), tc


def test_upload_list_ask_and_rate_through_the_client(live, tmp_path):
    client, tc = live
    pdf = tmp_path / "r.pdf"
    make_pdf(
        pdf, ["Revenue from operations was 12,563 crore in FY25. " * 3, "Another page, different words."]
    )

    first = client.upload("r.pdf", pdf.read_bytes())
    # The 202 is built after the job is handed to the worker, which may already have started it.
    assert first["status"] in ("QUEUED", "PROCESSING") and first["duplicate"] is False
    wait_for(tc, first["doc_id"], ("READY",))
    assert client.upload("r.pdf", pdf.read_bytes())["duplicate"] is True

    [d] = client.list_documents()
    assert d["status"] == "READY" and fmt.status_label(d) == "✅ Ready" and not fmt.is_busy([d])

    answer = client.ask("What was revenue in FY25?")
    [section] = answer["sections"]
    assert section["status"] == "answered" and fmt.citation_label(section["citations"][0]).startswith(
        "r.pdf · p."
    )
    assert fmt.snippet_text(section["citations"][0]) != "(no text available)"

    client.send_feedback(answer["trace_id"], 1)
    assert tc.app.state.request_store.get(answer["trace_id"])["feedback"] == 1


def test_the_apis_rejections_reach_the_user_as_readable_messages(live, tmp_path):
    client, _ = live
    with pytest.raises(ApiError, match="Not a PDF") as e:
        client.upload("notes.pdf", b"this is plain text, not a pdf")
    assert e.value.status == 415
    with pytest.raises(ApiError, match="too long") as e:
        client.ask("x" * 501)
    assert e.value.status == 422
    with pytest.raises(ApiError, match="empty"):
        client.ask("   ")


def test_processing_caveat_goes_above_an_answer_and_other_notes_below():
    from ui.formatting import split_notes

    partial = "big.pdf is still processing (20/312 pages): this answer only uses the pages indexed so far."
    s = {"status": "answered", "warnings": [partial, "⚠ number not found verbatim in source"]}
    assert split_notes(s) == ([partial], ["⚠ number not found verbatim in source"])
    abstained = {"status": "abstained", "warnings": [], "coverage_note": "x: still processing", "answer": ""}
    assert split_notes(abstained) == ([], ["x: still processing"])
