"""Pure display helpers (no Streamlit imports), so they can be unit-tested."""

from __future__ import annotations

import os


# Mirrors `query.max_question_chars` (the API enforces its own limit and the UI shows its message on a 422).
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


MAX_QUESTION_CHARS = _env_int("DOCQA_MAX_QUESTION_CHARS", 500)

DOCUMENT_TITLE = ":material/description: From your documents"
GENERAL_TITLE = ":material/public: General knowledge"

_BUSY = ("QUEUED", "PROCESSING", "PARTIAL")
# Same names as app.catalog (the UI does not import the API package).
REPORT_TYPES = [
    "Annual report", "Quarterly results", "Investor presentation", "Earnings call transcript", "Other"
]


def catalog_tag(doc: dict) -> str:
    """'TCS · Quarterly results · Q1 FY26' for a document card ('' when the API sent no catalog fields)."""
    keys = ("company", "report_type", "period")
    return " · ".join(escape_markdown(str(doc[k])) for k in keys if doc.get(k))


def is_busy(docs: list[dict]) -> bool:
    """True while any document is still being ingested (the UI keeps polling)."""
    return any(d.get("status") in _BUSY for d in docs)


def _pages(doc: dict) -> tuple[int, int | None]:
    total = doc.get("pages_total")
    done = doc.get("pages_done") or 0
    return (min(done, total) if total else done), total


def status_label(doc: dict) -> str:
    """One line for the documents list: icon + state (+ page progress)."""
    status = doc.get("status")
    done, total = _pages(doc)
    progress = f" {done}/{total}" if total else ""
    if status == "QUEUED":
        return "⏳ Queued"
    if status == "PROCESSING":
        return f"⚙️ Processing{progress}"
    if status == "PARTIAL":
        return f"🟡 Partial{progress} · searchable"
    if status == "READY":
        return "⚠️ Ready (with a warning)" if doc.get("error") else "✅ Ready"
    if status == "FAILED":
        return "❌ Failed"
    return str(status or "Unknown")


def progress_fraction(doc: dict) -> float | None:
    """0..1 while the document is in progress, None when no bar should be shown."""
    if doc.get("status") not in ("PROCESSING", "PARTIAL"):
        return None
    done, total = _pages(doc)
    if not total:
        return 0.0
    return max(0.0, min(1.0, done / total))


def status_detail(doc: dict) -> str | None:
    """The reason to show under a FAILED document, or the warning on a READY one."""
    err = (doc.get("error") or "").strip()
    if not err:
        return None
    if doc.get("status") == "FAILED":
        return f"Couldn't be processed: {shorten(err, 200)}"
    if doc.get("status") == "READY":
        return f"Note: {shorten(err, 200)}"
    return None


def shorten(text: str, limit: int = 40) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def citation_label(c: dict) -> str:
    """'Report.pdf · p.47 (PDF p.53)'. `display` already folds label vs PDF page."""
    page = c.get("display") or f"p.{c.get('page_label')}"
    return f"{c.get('filename', 'document')} · {page}"


def snippet_text(c: dict) -> str:
    return (c.get("snippet") or "").strip() or "(no text available)"


def section_title(section: dict) -> str:
    return DOCUMENT_TITLE if section.get("kind") == "document" else GENERAL_TITLE


def extra_notes(section: dict) -> list[str]:
    """Notes to show under the answer. An abstention's `answer` already includes the coverage note and the
    closest sections as prose, so the coverage note is only added when it is not already in the text."""
    notes = list(section.get("warnings") or [])
    cov = section.get("coverage_note")
    if cov and cov not in (section.get("answer") or ""):
        notes.append(cov)
    return notes


def computed_caption(section: dict) -> str:
    """One line under an answer whose figures were derived (a difference, a share) rather than printed."""
    return "Computed from cited figures: " + "; ".join(section.get("computed_numbers") or [])


def split_notes(section: dict) -> tuple[list[str], list[str]]:
    """(notes to show above the answer, notes to show below it). A "still processing" caveat on an answered
    section goes above, so nobody reads a possibly incomplete answer before the warning."""
    notes = extra_notes(section)
    if section.get("status") != "answered":
        return [], notes
    before = [n for n in notes if "still processing" in n]
    return before, [n for n in notes if n not in before]


def source_groups(section: dict) -> list[tuple[str, list[dict]]]:
    """The expandable source lists to render: (heading, items). Citations for an answer, the closest pages
    for an abstention, the retrieved passages when the answer service was down."""
    groups: list[tuple[str, list[dict]]] = []
    if section.get("citations"):
        groups.append(("Sources", section["citations"]))
    if section.get("closest_pages"):
        groups.append(("Closest pages", section["closest_pages"]))
    if section.get("fallback_passages"):
        groups.append(("Most relevant passages found", section["fallback_passages"]))
    return groups


def escape_markdown(text: str) -> str:
    """Model output goes through st.markdown; a bare `$` would start LaTeX ("$5 and $10" turns into maths)."""
    return text.replace("$", r"\$")


def feedback_value(picked: int | None) -> int | None:
    """st.feedback("thumbs") gives 1 (up), 0 (down) or None (deselected); the API wants 1 / -1."""
    if picked is None:
        return None
    return 1 if picked == 1 else -1


def route_caption(response: dict) -> str:
    """'Routed as document · 2.4 s'."""
    route = str(response.get("route") or "").lower()
    total_ms = (response.get("timings") or {}).get("total_ms")
    secs = f" · {total_ms / 1000:.1f} s" if isinstance(total_ms, int | float) else ""
    return f"Routed as {route}{secs}" if route else secs.lstrip(" ·")


def enhancer_caption(enhancer: dict | None) -> str:
    """What the query enhancer did: 'searched TCS, Q1 FY26 (3 reports) · PAT -> profit after tax'."""
    if not isinstance(enhancer, dict):
        return ""
    focus = [*(enhancer.get("companies") or []), *(enhancer.get("periods") or [])]
    parts = []
    n = enhancer.get("n_documents")
    if focus or n:
        where = ", ".join(focus) or "selected reports"
        count = f" ({n} report{'s' if n != 1 else ''})" if isinstance(n, int) else ""
        parts.append(f"searched {where}{count}")
    parts += [e.replace("->", "→") for e in enhancer.get("expansions") or []]
    return " · ".join(parts)


def enhancer_rows(enhancer: dict | None) -> list[tuple[str, str]]:
    """The query-enhancer box above an answer: what was typed, how it was understood, what was searched."""
    if not isinstance(enhancer, dict) or enhancer.get("enabled") is False:
        return []
    original = str(enhancer.get("original") or "")
    rows = [("You asked", original)] if original else []
    if enhancer.get("unclear"):
        rows.append(("Understood as", "no question found — please rephrase"))
    elif enhancer.get("rewritten") and enhancer.get("question"):
        rows.append(("Understood as", str(enhancer["question"])))
    elif enhancer.get("rewrite_error"):
        rows.append(("Understood as", "as typed (the rewrite was not available)"))
    else:
        rows.append(("Understood as", "as typed (no changes needed)"))
    if enhancer.get("needs_company"):
        rows.append(("Company", "none picked — choose one above the chat box"))
    scope = enhancer_caption(enhancer).removeprefix("searched ")
    if scope:
        rows.append(("Searched", scope[0].upper() + scope[1:]))
    return rows


def coverage_lines(company: dict) -> list[str]:
    """One line per report type: 'Quarterly results: Q1 FY26, Q2 FY26'."""
    lines = []
    for rt in company.get("report_types") or []:
        periods = ", ".join(rt.get("periods") or []) or "period not known"
        lines.append(f"**{escape_markdown(rt['type'])}**: {escape_markdown(periods)}")
    return lines


def example_question(company: dict) -> str | None:
    """A question the catalog says can be answered: 'What was TCS's revenue from operations in Q1 FY26?'."""
    periods = [p for rt in company.get("report_types") or [] for p in rt.get("periods") or []]
    if not periods:
        return None
    return f"What was {company['name']}'s revenue from operations in {periods[-1]}?"
