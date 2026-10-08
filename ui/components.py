"""Streamlit rendering: the sidebar (upload + documents list) and the answer view. No business logic here;
the decisions about what to show live in `ui.formatting` (tested)."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from ui import formatting as fmt
from ui.api_client import ApiClient, ApiError

POLL_SECONDS = 2

# Comfy dark: warm cocoa-brown surfaces instead of cold navy or black, one soft honey accent, a rounded
# typeface (Nunito), roomy padding and soft shadows. Your question sits in a warm bubble on the right, the
# answer on a soft card on the left with numbered sources under it, and the composer is docked at the bottom.
# Only stable Streamlit hooks are targeted (data-testids and the `st-key-<key>` class of keyed containers),
# so an upgrade can at worst drop the styling, never the content.
_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
:root {
  --dq-bg: #1C1A17; --dq-side: #221F1C; --dq-card: #2A2622; --dq-you: #3B3229; --dq-line: #3A352F;
  --dq-text: #EFE8DF; --dq-muted: #B5AA9C; --dq-honey: #E9B872; --dq-honey-soft: rgba(233,184,114,0.16);
  --dq-shadow: 0 6px 24px rgba(0,0,0,0.30);
  --dq-font: 'Nunito', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp { background: var(--dq-bg); }
.stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, h1, h2, h3, h4, button p, label p,
[data-testid="stExpander"] summary p, [data-testid="stChatInputTextArea"], [data-testid="stAlert"] p {
  font-family: var(--dq-font); font-variant-numeric: tabular-nums;
}
.stMarkdown p, .stMarkdown li { line-height: 1.7; }

/* header and sidebar */
[data-testid="stHeader"] { background: rgba(28,26,23,0.92); backdrop-filter: blur(6px); }
[data-testid="stSidebar"] { background: var(--dq-side); border-right: 1px solid var(--dq-line); }
[data-testid="stSidebarHeader"] img, [data-testid="stHeaderLogo"] { height: 2.2rem !important; }
section[data-testid="stSidebar"] h2 { font-size: 1.05rem; font-weight: 800; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px;
  border-color: var(--dq-line); background: var(--dq-card); }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { border-radius: 18px;
  background: var(--dq-card); border: 1.5px dashed var(--dq-line); }
section[data-testid="stSidebar"] [data-testid="stProgress"] > div > div > div { background: var(--dq-honey); }

/* empty state */
.st-key-welcome { align-items: center; padding: 16vh 0 0.5rem; }
.st-key-welcome [data-testid="stElementContainer"] { width: 100%; }
.stMarkdown p.dq-hello { font-size: 2rem; font-weight: 800; letter-spacing: -0.01em; line-height: 1.25;
  text-align: center;
  margin: 0 0 0.5rem; }
.stMarkdown p.dq-promise { color: var(--dq-muted); text-align: center; max-width: 32rem; margin: 0 auto;
  font-size: 1.02rem; }
.st-key-chips { flex-direction: row !important; flex-wrap: wrap; justify-content: center; gap: 0.6rem;
  margin-top: 1.75rem; }
.st-key-chips [data-testid="stElementContainer"] { width: auto !important; }
.st-key-chips button { border-radius: 999px; background: var(--dq-card); border: 1px solid var(--dq-line);
  padding: 0.45rem 1.1rem; min-height: 2.6rem; color: var(--dq-text); box-shadow: var(--dq-shadow); }
.st-key-chips button:hover { background: var(--dq-you); border-color: var(--dq-honey);
  color: var(--dq-text); }
.st-key-chips button p { font-size: 0.95rem; }
.st-key-chips [data-testid="stIconMaterial"] { color: var(--dq-honey); }

/* a turn: your question in a warm bubble on the right, the answer on a soft card on the left */
[class*="st-key-turn_"] { gap: 1.1rem; padding-bottom: 1.6rem; }
[class*="st-key-you_"] { background: var(--dq-you); border-radius: 22px 22px 6px 22px;
  padding: 0.75rem 1.2rem;
  box-shadow: var(--dq-shadow); margin-left: auto; }
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-you_"]) { align-self: flex-end; max-width: 78%; }
.stMarkdown p.dq-you { margin: 0; padding: 0; font-size: 1.05rem; line-height: 1.55; }
[class*="st-key-you_"] [data-testid="stMarkdownContainer"],
[class*="st-key-you_"] [data-testid="stElementContainer"] { margin: 0 !important; overflow: visible; }
[class*="st-key-answer_"] { background: var(--dq-card); border: 1px solid var(--dq-line);
  border-radius: 22px 22px 22px 6px; padding: 1.1rem 1.4rem 0.8rem; gap: 0.6rem;
  box-shadow: var(--dq-shadow); }
[class*="st-key-answer_"] .stMarkdown p { font-size: 1.05rem; }
[class*="st-key-answer_"] h3 { font-size: 0.98rem; font-weight: 800; color: var(--dq-muted);
  padding-top: 0.2rem; }
[class*="st-key-answer_"] [data-testid="stAlert"] { border-radius: 16px; }
[class*="st-key-answer_"] .stCaption p { font-size: 0.82rem; color: var(--dq-muted); }

/* sources: numbered rows that open to the quoted page */
[class*="st-key-answer_"] [data-testid="stExpander"] details { border-radius: 14px;
  border-color: var(--dq-line); background: var(--dq-side); }
[class*="st-key-answer_"] [data-testid="stExpander"] summary { padding: 0.4rem 0.8rem; }
[class*="st-key-answer_"] [data-testid="stExpander"] summary p { font-size: 0.9rem; color: var(--dq-text); }
[class*="st-key-answer_"] [data-testid="stExpander"] summary:hover p { color: var(--dq-honey); }
.dq-enh { border: 1px solid var(--dq-line); border-left: 3px solid var(--dq-honey); border-radius: 12px;
  background: var(--dq-honey-soft); padding: 0.55rem 0.85rem; margin-bottom: 0.8rem; font-size: 0.9rem; }
.dq-enh-h { font-weight: 700; color: var(--dq-honey); margin-bottom: 0.25rem; }
.dq-enh-row { display: flex; gap: 0.6rem; line-height: 1.5; }
.dq-enh-k { color: var(--dq-muted); min-width: 7.5rem; flex-shrink: 0; }
.dq-snippet { font-size: 0.92rem; line-height: 1.6; border-left: 3px solid var(--dq-honey);
  padding: 0.1rem 0 0.1rem 0.85rem; color: var(--dq-text); opacity: 0.9; }

/* composer, docked at the bottom */
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--dq-bg); }
[data-testid="stBottomBlockContainer"] { max-width: 800px; padding-bottom: 1.25rem; }
[data-testid="stChatInput"] { border-radius: 26px; border: 1px solid var(--dq-line);
  background: var(--dq-card);
  box-shadow: var(--dq-shadow); padding: 0.25rem 0.45rem; }
[data-testid="stChatInput"]:focus-within { border-color: var(--dq-honey);
  box-shadow: 0 0 0 4px var(--dq-honey-soft), var(--dq-shadow); }
[data-testid="stChatInput"] div:not([data-testid="stChatInputSubmitButton"]),
[data-testid="stChatInput"] textarea {
  background: transparent !important; border-color: transparent !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--dq-honey) !important; border-radius: 999px; }
[data-testid="stChatInputSubmitButton"] svg { fill: var(--dq-bg); color: var(--dq-bg); }

@media (max-width: 640px) {
  [data-testid="stLayoutWrapper"]:has(> [class*="st-key-you_"]) { max-width: 88%; }
  [class*="st-key-answer_"] { padding: 0.95rem 1rem 0.7rem; }
  .stMarkdown p.dq-hello { font-size: 1.55rem; }
  .st-key-welcome { padding-top: 10vh; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""

LOGO = Path(__file__).resolve().parent / "assets" / "logo.svg"
MARK = Path(__file__).resolve().parent / "assets" / "mark.svg"


def inject_style() -> None:
    st.logo(str(LOGO), size="large", icon_image=str(MARK))
    st.markdown(_STYLE, unsafe_allow_html=True)


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


# ------------------------------------------------------------------ sidebar: upload + documents


def sidebar(client: ApiClient) -> None:
    with st.sidebar:
        st.header("Documents")
        _upload_form(client)
        _documents_panel(client)


def _upload_form(client: ApiClient) -> None:
    for kind, text in st.session_state.pop("upload_notes", []):
        getattr(st, kind)(text)

    n = st.session_state.setdefault("uploader_n", 0)
    files = st.file_uploader(
        "Upload PDF reports", type=["pdf"], accept_multiple_files=True, key=f"uploader_{n}",
        help="Text and scanned PDFs both work. Scanned pages are read with OCR and take longer.",
    )
    if files and st.button("Upload", type="primary", use_container_width=True):
        notes: list[tuple[str, str]] = []
        with st.spinner("Uploading…"):
            for f in files:
                notes.append(_upload_one(client, f.name, f.getvalue()))
        st.session_state.upload_notes = notes
        st.session_state.uploader_n = n + 1  # a new key empties the uploader, so a rerun does not re-post
        st.rerun()


def _upload_one(client: ApiClient, name: str, data: bytes) -> tuple[str, str]:
    try:
        doc = client.upload(name, data)
    except ApiError as exc:
        return "error", f"**{name}**: {exc}"
    if doc.get("duplicate"):
        return "info", f"**{name}** was already uploaded ({fmt.status_label(doc)})."
    return "success", f"**{name}** uploaded. Processing starts now."


def _documents_panel(client: ApiClient) -> None:
    # The fragment re-runs by itself every POLL_SECONDS while anything is in progress. `run_every` is fixed
    # when the fragment is defined, so when "busy" flips we trigger one full rerun to redefine it.
    busy_when_defined = st.session_state.setdefault("docs_busy", False)

    @st.fragment(run_every=POLL_SECONDS if busy_when_defined else None)
    def panel() -> None:
        try:
            docs = client.list_documents()
        except ApiError as exc:
            st.warning(str(exc))
            st.button("Retry", key="docs_retry")  # a click reruns this fragment
            if busy_when_defined:
                st.session_state.docs_busy = False
                st.rerun()
            return
        busy = fmt.is_busy(docs)
        _render_documents(client, docs)
        if busy != busy_when_defined:
            st.session_state.docs_busy = busy
            st.rerun()

    panel()


def _render_documents(client: ApiClient, docs: list[dict]) -> None:
    if not docs:
        st.caption("No documents yet. Upload a PDF to ask questions about it.")
        return
    for i, doc in enumerate(docs):
        with st.container(border=True):
            name_col, btn_col = st.columns([5, 1], vertical_alignment="center")
            name_col.markdown(f"**{fmt.escape_markdown(fmt.shorten(doc['filename'], 30))}**")
            with btn_col.popover("", icon=":material/more_horiz:", help="Document options"):
                st.caption(fmt.escape_markdown(doc["filename"]))
                _catalog_form(client, doc, i)
                if st.button(
                    "Remove document", key=f"rm_{i}_{doc['id']}", type="primary", icon=":material/delete:"
                ):
                    _remove(client, doc)
            tag = fmt.catalog_tag(doc)
            if tag:
                st.caption(tag)
            st.caption(fmt.status_label(doc))
            frac = fmt.progress_fraction(doc)
            if frac is not None:
                st.progress(frac)
            detail = fmt.status_detail(doc)
            if detail:
                (st.error if doc.get("status") == "FAILED" else st.warning)(detail)


def _catalog_form(client: ApiClient, doc: dict, i: int) -> None:
    """Fix what the file name said: company, report type, period. Blank = back to the file-name guess."""
    with st.form(key=f"cat_{i}_{doc['id']}", border=False):
        company = st.text_input("Company", value=doc.get("company") or "")
        report_type = st.selectbox(
            "Report type", fmt.REPORT_TYPES,
            index=fmt.REPORT_TYPES.index(doc["report_type"])
            if doc.get("report_type") in fmt.REPORT_TYPES
            else 0,
        )
        period = st.text_input("Period", value=doc.get("period") or "", placeholder="FY26 or Q1 FY26")
        if st.form_submit_button("Save", icon=":material/save:"):
            try:
                client.update_document(doc["id"], company=company, report_type=report_type, period=period)
                st.session_state.upload_notes = [("success", f"Updated **{doc['filename']}**.")]
            except ApiError as exc:
                st.session_state.upload_notes = [("error", f"Couldn't update **{doc['filename']}**: {exc}")]
            st.rerun()


def load_companies(client: ApiClient) -> list[dict]:
    """The catalog's companies (GET /catalog). Read fresh on every page run, so a document uploaded in the
    sidebar shows up as soon as the page reruns (after the upload, and again when it is Ready)."""
    try:
        return client.catalog().get("companies") or []
    except (ApiError, AttributeError):
        return []


def picked_company(companies: list[dict]) -> str | None:
    """The company questions are pinned to: the only one loaded, or the one picked above the chat box.

    Kept in `st.session_state.company`, not in the selectbox's own key: Streamlit drops a widget's value
    when a run ends before the widget is drawn, and the Ask page reruns mid-script after every answer."""
    names = [c["name"] for c in companies]
    if len(names) == 1:
        return names[0]
    pick = st.session_state.get("company")
    return pick if pick in names else None


def _remember_pick() -> None:
    st.session_state.company = st.session_state.get("company_widget")


def enhancer_on() -> bool:
    """Whether questions go through the query enhancer (the toggle by the chat box; on by default)."""
    return bool(st.session_state.get("enhance", True))


def _remember_enhance() -> None:
    st.session_state.enhance = bool(st.session_state.get("enhance_widget", True))


def composer(companies: list[dict]) -> str | None:
    """Docked at the bottom: the company picker and "what's ingested", right above the chat box.

    With several companies loaded the chat box stays disabled until one is picked, so a question can never
    be answered from the wrong company's reports. Returns the typed question, if any."""
    names = [c["name"] for c in companies]
    company = picked_company(companies)
    st.session_state.company_widget = company  # redraw the widget with the remembered pick
    with st.bottom:
        if companies:
            with st.container(key="composer_bar", horizontal=True, vertical_alignment="center"):
                st.selectbox(
                    "Company", names, key="company_widget", on_change=_remember_pick,
                    placeholder="Choose a company…", label_visibility="collapsed", disabled=len(names) == 1,
                    help="Answers come only from this company's reports.",
                )
                _coverage_popover(companies, company)
                st.session_state.enhance_widget = enhancer_on()  # redraw with the remembered choice
                st.toggle(
                    "✨ Enhance", key="enhance_widget", on_change=_remember_enhance,
                    help="On: fixes spelling, spells out short forms and narrows the search to the right "
                    "reports before answering (one extra small model call). Off: your question is searched "
                    "exactly as typed.",
                )
        blocked = len(names) > 1 and company is None
        if blocked:
            placeholder = "Choose a company above to ask a question"
        elif company:
            placeholder = f"Ask about {company}'s reports"
        else:
            placeholder = "Ask about your reports"
        return st.chat_input(placeholder, max_chars=fmt.MAX_QUESTION_CHARS, disabled=blocked)


def _coverage_popover(companies: list[dict], company: str | None) -> None:
    n_docs = sum(c["n_documents"] for c in companies)
    label = f"{len(companies)} compan{'y' if len(companies) == 1 else 'ies'} · {n_docs} reports"
    shown = [c for c in companies if c["name"] == company] if company else companies
    tip = "What has been ingested, and what you can ask"
    with st.popover(label, icon=":material/library_books:", help=tip):
        for c in shown:
            st.markdown(f"**{fmt.escape_markdown(c['name'])}** · {c['n_documents']} report(s)")
            for line in fmt.coverage_lines(c):
                st.markdown(f"- {line}")
            busy = c["n_documents"] - c["n_ready"]
            if busy:
                st.caption(f"{busy} still processing: answers from them may be incomplete.")
            example = fmt.example_question(c)
            if example and st.button(example, key=f"cov_ex_{c['name']}", icon=":material/lightbulb:"):
                if len(companies) > 1:
                    st.session_state.company = c["name"]
                st.session_state.pending = example
                st.rerun()


def _remove(client: ApiClient, doc: dict) -> None:
    try:
        client.delete_document(doc["id"])
        st.session_state.upload_notes = [("info", f"Removed **{doc['filename']}**. You can upload it again.")]
    except ApiError as exc:
        st.session_state.upload_notes = [("error", f"Couldn't remove **{doc['filename']}**: {exc}")]
    st.rerun()


# ------------------------------------------------------------------ answers


def render_response(response: dict, key_prefix: str = "") -> None:
    enh = response.get("enhancer")
    if isinstance(enh, dict) and enh.get("enabled") is False:
        st.caption(":material/auto_fix_off: Query enhancer off — searched exactly as typed.")
    rows = fmt.enhancer_rows(enh)
    if rows:
        body = "".join(
            f'<div class="dq-enh-row"><span class="dq-enh-k">{html.escape(k)}</span>'
            f"<span>{html.escape(v).replace('$', '&#36;')}</span></div>"
            for k, v in rows
        )
        st.markdown(
            f'<div class="dq-enh"><div class="dq-enh-h">✨ Query enhancer</div>{body}</div>',
            unsafe_allow_html=True,
        )
    sections = response.get("sections") or []
    many = len(sections) > 1
    for i, section in enumerate(sections):
        if many:
            if i:
                st.divider()
            st.subheader(fmt.section_title(section))
        _render_section(section, f"{key_prefix}_s{i}")
    caption = fmt.route_caption(response)
    if caption:
        st.caption(caption)


def _render_section(section: dict, key_prefix: str = "") -> None:
    status = section.get("status")
    text = fmt.escape_markdown(section.get("answer") or "")

    if section.get("label"):  # general knowledge: say so before the answer
        st.info(section["label"], icon=":material/public:")

    before, after = fmt.split_notes(section)
    for note in before:  # "still processing" caveats go above the answer, where they are read first
        st.warning(fmt.escape_markdown(note), icon=":material/hourglass_top:")

    if status == "answered":
        st.markdown(text)
    elif status == "error":
        st.error(text)
    elif status in ("not_ready", "needs_company", "unclear"):
        st.info(text)
    else:  # abstained
        st.warning(text)
    if section.get("hint"):
        st.caption(f":material/lightbulb: {section['hint']}")

    for heading, items in fmt.source_groups(section):
        st.caption(heading)
        for n, c in enumerate(items, start=1):
            with st.expander(f"[{n}] {fmt.citation_label(c)}", expanded=bool(c.get("primary"))):
                st.markdown(f'<div class="dq-snippet">{_snippet_html(c)}</div>', unsafe_allow_html=True)
                _show_in_pdf(c, f"{key_prefix}_{heading}_{n}")

    for note in after:
        st.warning(fmt.escape_markdown(note))
    if status == "answered" and section.get("computed_numbers"):
        st.caption(fmt.computed_caption(section))


@st.cache_data(show_spinner=False, max_entries=64)
def _highlight_image(doc_id: str, pdf_page: int, terms: tuple[str, ...], context: str) -> tuple[bytes, int]:
    return get_client().page_highlight(doc_id, pdf_page, list(terms), context)


def _show_in_pdf(c: dict, key: str) -> None:
    """'Show in PDF': the cited page, cropped, with the answer's row highlighted. The picture is fetched
    (and rendered by the API) only after the click, then kept in the session so reruns do not drop it."""
    if not c.get("doc_id") or not c.get("pdf_page"):
        return
    shown = f"pdf_{key}"
    if st.button("Show in PDF", key=f"btn_{shown}", icon=":material/picture_as_pdf:"):
        st.session_state[shown] = True
    if not st.session_state.get(shown):
        return
    try:
        png, matches = _highlight_image(
            c["doc_id"], int(c["pdf_page"]), tuple(c.get("highlight_terms") or ()), fmt.snippet_text(c)
        )
    except ApiError as exc:
        st.caption(f"Couldn't load the page: {exc}")
        return
    st.image(png, caption=fmt.highlight_caption(matches, c.get("source_kind")), use_container_width=True)


def _snippet_html(c: dict) -> str:
    """Snippet text as inert HTML: escaped, `$` kept literal (no LaTeX), line breaks without blank lines
    (a blank line would end the HTML block and let Markdown loose on the text)."""
    text = html.escape(fmt.snippet_text(c)).replace("$", "&#36;")
    return "<br>".join(line for line in text.splitlines() if line.strip())


def feedback_widget(client: ApiClient, message: dict) -> None:
    """👍/👎 under an answer. Once rated it becomes a one-line note (the rating is kept on the message)."""
    rated = message.get("feedback")
    if rated:
        st.caption("Thanks, you rated this answer 👍" if rated == 1 else "Thanks, you rated this answer 👎")
        return
    trace_id = message["trace_id"]
    key = f"fb_{message['uid']}"  # unique per message, even if two trace ids were equal

    def on_rate() -> None:
        value = fmt.feedback_value(st.session_state.get(key))
        if value is None:  # the same thumb clicked again (deselected): nothing to store
            return
        try:
            client.send_feedback(trace_id, value)
            message["feedback"] = value
        except ApiError:
            message["feedback_error"] = True
            st.session_state[key] = None  # let them click again

    st.feedback("thumbs", key=key, on_change=on_rate)
    if message.pop("feedback_error", None):
        st.caption("Couldn't save your feedback just now. Please try again.")
