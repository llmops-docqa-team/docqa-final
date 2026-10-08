"""Headless smoke test of the Streamlit app (`streamlit.testing.AppTest`): the pages render without an
exception for each kind of reply, including the API being down. Layout details are not asserted."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from ui import components
from ui.api_client import ApiClient, ApiError

LABEL = "General knowledge — not from your documents; may be out of date."


def section(kind="document", status="answered", answer="Revenue was 12,563 crore.", **kw):
    base = {
        "kind": kind, "status": status, "answer": answer, "question": "q", "citations": [], "label": None,
        "abstain_reason": None, "hint": None, "closest_pages": [], "fallback_passages": [],
        "coverage_note": None, "warnings": [], "number_check": "na", "top_score": 0.7,
    }  # fmt: skip
    return {**base, **kw}


CITE = {"filename": "Report.pdf", "display": "p.47 (PDF p.53)", "snippet": "Revenue 12,563 crore."}


def response(*sections, route="DOCUMENT", trace_id="t-1"):
    return {"trace_id": trace_id, "route": route, "router_ok": True, "sections": list(sections),
            "timings": {"total_ms": 1500}}  # fmt: skip


class FakeClient:
    def __init__(self, docs=(), reply=None, error=None):
        self.docs, self.reply, self.error = list(docs), reply, error
        self.asked: list[str] = []
        self.feedback: list[tuple[str, int]] = []
        self.uploads: list[str] = []

    def list_documents(self):
        if self.error:
            raise self.error
        return self.docs

    def ask(self, question):
        self.asked.append(question)
        if self.error:
            raise self.error
        return self.reply

    def upload(self, name, data):
        self.uploads.append(name)
        return {"doc_id": "d", "duplicate": False, "status": "QUEUED"}

    def send_feedback(self, trace_id, value):
        self.feedback.append((trace_id, value))


def doc(status, **kw):
    return {"id": "d", "filename": "Report.pdf", "status": status, "pages_done": 0, "pages_total": 10,
            "error": None, **kw}  # fmt: skip


def run(monkeypatch, client) -> AppTest:
    monkeypatch.setattr(components, "get_client", lambda: client)
    at = AppTest.from_file("ui/app.py", default_timeout=30).run()
    assert not at.exception
    return at


def ask(at: AppTest, text="What was revenue?") -> AppTest:
    """Type into the docked composer and send. The page parks the question and reruns twice (draw the
    question, then the answer); AppTest does not follow st.rerun(), so those runs are done here."""
    at.chat_input[0].set_value(text).run()
    for _ in range(2):
        assert not at.exception
        at.run()
    assert not at.exception
    return at


def source_labels(at: AppTest) -> list[str]:
    """Source expanders. An expander with an icon is reported by AppTest as a "status" block."""
    return [e.label for e in [*at.expander, *at.status]]


def texts(elements) -> str:
    return " | ".join(str(e.value) for e in elements)


def test_documents_list_shows_every_status(monkeypatch):
    docs = [
        doc("QUEUED"), doc("PROCESSING", pages_done=3), doc("PARTIAL", pages_done=5), doc("READY"),
        doc("READY", error="2 of 86 pages could not be read"), doc("FAILED", error="no text"),
    ]  # fmt: skip
    at = run(monkeypatch, FakeClient(docs))
    seen = texts(at.sidebar.caption)
    for expected in (
        "Queued",
        "Processing 3/10",
        "Partial 5/10",
        "Ready",
        "Ready (with a warning)",
        "Failed",
    ):
        assert expected in seen
    assert len(at.sidebar.get("progress")) == 2  # PROCESSING and PARTIAL
    assert "Couldn't be processed: no text" in texts(at.sidebar.error)
    assert "2 of 86 pages" in texts(at.sidebar.warning)


def test_empty_state_and_polling_flag(monkeypatch):
    at = run(monkeypatch, FakeClient([]))
    assert "No documents yet" in texts(at.sidebar.caption)
    assert at.session_state["docs_busy"] is False
    at = run(monkeypatch, FakeClient([doc("PROCESSING")]))
    assert at.session_state["docs_busy"] is True  # the fragment now polls every 2 s


def test_document_answer_has_citation_chip_and_number_badge(monkeypatch):
    s = section(
        citations=[CITE], warnings=["⚠ number not found verbatim in source: 366"], number_check="fail"
    )
    at = ask(run(monkeypatch, FakeClient(reply=response(s))))
    assert "Revenue was 12,563 crore." in texts(at.markdown)
    assert source_labels(at) == ["[1] Report.pdf · p.47 (PDF p.53)"]
    assert "number not found verbatim" in texts(at.warning)


def test_general_answer_shows_the_label(monkeypatch):
    s = section("general", answer="Paris.", label=LABEL)
    at = ask(run(monkeypatch, FakeClient(reply=response(s, route="GENERAL"))), "Capital of France?")
    assert LABEL in texts(at.info) and "Paris." in texts(at.markdown)


def test_mixed_has_two_titled_sections(monkeypatch):
    doc_s = section(citations=[CITE])
    gen_s = section("general", answer="Paris.", label=LABEL)
    at = ask(run(monkeypatch, FakeClient(reply=response(doc_s, gen_s, route="MIXED"))))
    assert [s.value for s in at.subheader] == [
        ":material/description: From your documents", ":material/public: General knowledge"
    ]


def test_abstention_shows_reason_closest_pages_and_coverage(monkeypatch):
    cov = "Report.pdf: searched 20 of 312 pages; the rest is still processing."
    s = section(
        status="abstained", answer=f"I couldn't find this in your documents. I won't guess. {cov}",
        abstain_reason="insufficient", closest_pages=[CITE], coverage_note=cov,
        hint="Try asking about something more specific.",
    )  # fmt: skip
    at = ask(run(monkeypatch, FakeClient(reply=response(s))))
    assert "I won't guess" in texts(at.warning)
    assert "Try asking about something more specific" in texts(at.caption)
    assert source_labels(at) == ["[1] Report.pdf · p.47 (PDF p.53)"]
    assert texts(at.warning).count("searched 20 of 312 pages") == 1  # not repeated as a separate note


def test_llm_down_shows_the_fallback_passages(monkeypatch):
    s = section(
        status="error", answer="The answer service is temporarily unavailable.", fallback_passages=[CITE]
    )
    at = ask(run(monkeypatch, FakeClient(reply=response(s))))
    assert "temporarily unavailable" in texts(at.error)
    assert len(source_labels(at)) == 1


def test_not_ready_message(monkeypatch):
    s = section(
        status="not_ready",
        answer="Report.pdf is still processing (140/312 pages). Ask again when it's Ready.",
    )
    at = ask(run(monkeypatch, FakeClient(reply=response(s))))
    assert "still processing (140/312" in texts(at.info)


def test_answer_text_with_dollar_signs_is_not_swallowed_as_latex(monkeypatch):
    s = section("general", answer="It costs $5 and $10.", label=LABEL)
    at = ask(run(monkeypatch, FakeClient(reply=response(s, route="GENERAL"))))
    assert r"\$5 and \$10" in texts(at.markdown)


def test_api_down_gives_friendly_messages_not_a_traceback(monkeypatch):
    down = ApiError("Can't reach the FinChat service at http://x. Is it running?", unreachable=True)
    at = run(monkeypatch, FakeClient(error=down))
    assert "Can't reach the FinChat service" in texts(at.sidebar.warning)
    at = ask(at)
    assert "Can't reach the FinChat service" in texts(at.error)


def test_the_real_client_against_a_dead_port_does_not_crash_the_app(monkeypatch):
    monkeypatch.setattr(components, "get_client", lambda: ApiClient("http://127.0.0.1:9"))
    at = AppTest.from_file("ui/app.py", default_timeout=30).run()
    assert not at.exception and "Can't reach" in texts(at.sidebar.warning)
    ask(at)
    assert "Can't reach" in texts(at.error)


@pytest.mark.parametrize("bad", [{}, {"trace_id": "t"}, {"sections": None, "trace_id": "t"}])
def test_an_unreadable_reply_is_handled(monkeypatch, bad):
    at = ask(run(monkeypatch, FakeClient(reply=bad)))
    assert not at.exception


def test_only_the_current_question_is_sent_to_the_backend(monkeypatch):
    client = FakeClient(reply=response(section()))
    at = ask(run(monkeypatch, client), "first")
    ask(at, "second")
    assert client.asked == ["first", "second"]  # plain strings: no history is passed along


def test_example_question_buttons_ask_and_disappear(monkeypatch):
    client = FakeClient(reply=response(section()))
    at = run(monkeypatch, client)
    assert len([b for b in at.button if (b.key or "").startswith("ex_")]) == 4
    at.button(key="ex_What is working capital?").click().run()
    at.run()
    at.run()
    assert not at.exception
    assert client.asked == ["What is working capital?"]  # asked once, not again on a redraw
    assert not [b for b in at.button if (b.key or "").startswith("ex_")]  # gone once a question was asked


def test_each_question_is_a_bubble_with_its_answer_below(monkeypatch):
    at = ask(run(monkeypatch, FakeClient(reply=response(section()))), "What was revenue?")
    shown = texts(at.markdown)
    assert 'class="dq-you">What was revenue?' in shown and "Revenue was 12,563 crore." in shown
    assert shown.index("What was revenue?") < shown.index("Revenue was 12,563 crore.")


def test_question_text_is_escaped_not_rendered_as_html(monkeypatch):
    at = ask(run(monkeypatch, FakeClient(reply=response(section()))), "<b>bold</b> costs $5?")
    shown = texts(at.markdown)
    assert "&lt;b&gt;bold&lt;/b&gt; costs &#36;5?" in shown and "<b>bold</b>" not in shown


def test_an_empty_question_is_not_sent(monkeypatch):
    client = FakeClient(reply=response(section()))
    at = run(monkeypatch, client)
    at.chat_input[0].set_value("   ").run()
    at.run()
    assert not at.exception and client.asked == []


def test_a_document_can_be_removed_from_the_sidebar(monkeypatch):
    removed: list[str] = []
    client = FakeClient([doc("READY")])
    client.delete_document = removed.append
    at = run(monkeypatch, client)
    at.sidebar.button(key="rm_0_d").click().run()
    assert not at.exception and removed == ["d"]
    assert "Removed" in texts(at.sidebar.info)
