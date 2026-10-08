"""Ask page: a comfy dark chat. Your question sits in a warm bubble on the right, the answer on a soft card
below with numbered sources, and the composer is docked at the bottom. History is shown on screen only;
each question is sent to the API on its own.

Flow: a submitted question (typed or a suggestion) is parked in `pending` and the page reruns, so the
question and its "searching" state are drawn in place, in order, before the answer replaces them.
"""

from __future__ import annotations

import html
from uuid import uuid4

import streamlit as st

from ui import components
from ui.api_client import ApiError

client = components.get_client()
history: list[dict] = st.session_state.setdefault("messages", [])

# A readable column (~65-75 characters a line) for this page only; the Metrics page keeps the full width.
st.markdown(
    "<style>.block-container { max-width: 800px; padding-top: 4.5rem; padding-bottom: 7rem; }</style>",
    unsafe_allow_html=True,
)

# (question, Material icon) - suggestions shown before the first question.
SUGGESTIONS = [
    ("What was revenue from operations in FY2025?", ":material/payments:"),
    ("What was total equity at the end of the year?", ":material/account_balance:"),
    ("What is working capital?", ":material/lightbulb:"),
    ("What is the FY2030 revenue forecast?", ":material/help:"),
]


def ask_later(question: str) -> None:
    st.session_state.pending = question
    st.rerun()


def question_bubble(i: int, text: str) -> None:
    safe = html.escape(text).replace("$", "&#36;")
    with st.container(key=f"you_{i}", width="content"):
        st.markdown(f'<p class="dq-you">{safe}</p>', unsafe_allow_html=True)


def answer_card(i: int, message: dict) -> None:
    with st.container(key=f"answer_{i}"):
        if message.get("error"):
            st.error(message["error"])
        else:
            components.render_response(message["response"], f"answer_{i}")
            components.feedback_widget(client, message)


def ask(question: str, company: str | None, enhance: bool = True) -> dict:
    try:
        options = {**({"company": company} if company else {}), **({} if enhance else {"enhance": False})}
        response = client.ask(question, **options)
        trace_id = response["trace_id"]
        return {"role": "assistant", "response": response, "trace_id": trace_id, "uid": uuid4().hex}
    except ApiError as exc:
        return {"role": "assistant", "error": str(exc)}
    except (KeyError, TypeError, ValueError):  # a reply we cannot read: never show a traceback
        return {"role": "assistant", "error": "The service sent a reply the app could not read."}


companies = components.load_companies(client)
company = components.picked_company(companies)
pending = st.session_state.pop("pending", None)
if pending:
    history.append({"role": "user", "content": pending})

if not history:
    with st.container(key="welcome"):
        st.markdown(
            '<p class="dq-hello">What do you want to know from your reports?</p>', unsafe_allow_html=True
        )
        st.markdown(
            '<p class="dq-promise">Answers come only from the PDFs you upload, with the page each one came '
            "from. If a report doesn't say, DocQA tells you instead of guessing.</p>",
            unsafe_allow_html=True,
        )
    with st.container(key="chips", horizontal=True):
        for label, icon in SUGGESTIONS:
            if st.button(label, key=f"ex_{label}", icon=icon):
                ask_later(label)

# Pair each question with the answer that follows it; a question still waiting has no answer yet.
turns: list[tuple[str, dict | None]] = []
for m in history:
    if m["role"] == "user":
        turns.append((m["content"], None))
    elif turns and turns[-1][1] is None:
        turns[-1] = (turns[-1][0], m)

for i, (question, answer) in enumerate(turns):
    with st.container(key=f"turn_{i}"):
        question_bubble(i, question)
        if answer is not None:
            answer_card(i, answer)
        elif pending and i == len(turns) - 1:
            with st.container(key=f"answer_{i}"):
                with st.spinner("Reading the relevant pages…"):
                    reply = ask(pending, company, components.enhancer_on())
            history.append(reply)
            st.rerun()  # redraw from history, so every answer (and its 👍/👎) is wired the same way

typed = components.composer(companies)
if typed and typed.strip():
    ask_later(typed.strip())
