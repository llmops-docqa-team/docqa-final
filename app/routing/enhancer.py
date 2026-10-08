"""Query enhancer, rule-based half: decides which reports a question is about and what to search for.
No LLM call here; the optional LLM rewrite (spelling, short forms) is in `app.routing.rewriter`.

1. Company: a company from the catalog named in the question (full name, file-name tag such as "EIG", or
   initials), or the one picked in the UI, narrows the search to that company's documents.
2. Period: "q1fy2026", "Q1 FY26", "FY 2024-25" are normalised. If documents for that period exist, the
   search is narrowed to them (a full year also keeps that year's quarterly documents). If none exist, the
   search is not narrowed and a note says the period is not covered.
3. Wording: common finance abbreviations get their long form added (PAT -> profit after tax), and the
   period is added in the spellings the reports use, so keyword (BM25) search matches either.

These rules only change the search text and which documents are searched. When the LLM rewrite is on, the
router and the answer model are asked the rewritten question (the UI shows the user's original next to it).
With the enhancer switched off (`plain=True`), the question is searched as typed, limited only to the
company the user picked.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.catalog import CatalogCompany, build_catalog, find_periods, fiscal_year, period_sort_key
from app.llm.client import Usage

ABBREVIATIONS: dict[str, str] = {
    "pat": "profit after tax",
    "pbt": "profit before tax",
    "ebitda": "earnings before interest tax depreciation amortisation",
    "ebit": "earnings before interest and tax",
    "eps": "earnings per share",
    "capex": "capital expenditure",
    "rev": "revenue",
    "revenue": "revenue from operations",
    "topline": "revenue from operations",
    "opm": "operating profit margin",
    "npm": "net profit margin",
    "roe": "return on equity",
    "roce": "return on capital employed",
    "fcf": "free cash flow",
    "cfo": "cash flow from operations",
    "dps": "dividend per share",
    "yoy": "year on year",
    "qoq": "quarter on quarter",
    "d/e": "debt to equity",
    "nwc": "net working capital",
    "cogs": "cost of goods sold",
    "headcount": "employees",
    "attrition": "attrition rate",
}

# Phrases the reports use for the same line item (Indian annual reports vs IFRS results vs presentations).
# Every added word also dilutes a keyword query, so an entry stays only if it did not lower retrieval
# (eval.retrieval_eval --enhance). Measured and left out: "borrowings" -> debt; loans (main set MRR -0.05) and
# "dividend" -> dividend declared; ... (T009 ranked lower).
SYNONYMS: dict[str, str] = {
    "profit after tax": "net profit; profit for the period",
    "net profit": "profit after tax; profit for the period",
    "revenue from operations": "total revenue; income from operations",
    "earnings per share": "EPS basic diluted",
    "operating margin": "EBIT margin; operating profit",
    "total equity": "shareholders' funds; net worth",
    "cash flow from operations": "net cash generated from operating activities",
    "finance cost": "finance costs; interest expense; borrowing costs",
    "other income": "interest income; treasury income",
    "net worth": "total equity; shareholders' funds",
    "capital expenditure": "additions to property, plant and equipment",
    "employee cost": "employee benefits expense; staff cost",
    "tax expense": "income tax; current tax; deferred tax",
}

_WORD = re.compile(r"[a-z0-9/&]+")


@dataclass
class Enhancement:
    search_query: str  # what retrieval searches for
    question: str = ""  # what the answer model is asked: the rewrite when there is one, else the user's words
    original: str = ""  # the user's own (sub-)question
    rewritten: bool = False  # the LLM rewrite changed the wording
    rewrite_error: str | None = None  # llm_unavailable | bad_json | rejected (the user's words were kept)
    needs_company: bool = False  # several companies loaded and none named or picked: ask which one
    unclear: bool = False  # the rewrite found no question in the text: ask the user to rephrase
    enabled: bool = True  # False: the user switched it off; only the picked company limits the search
    available_companies: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)  # tokens of the rewrite call
    rewrite_ms: float = 0.0
    companies: list[str] = field(default_factory=list)  # catalog companies the search is limited to
    periods: list[str] = field(default_factory=list)  # canonical periods found in the question
    doc_ids: list[str] | None = None  # None = search every document
    expansions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # shown to the user (e.g. period not covered)
    company_source: str | None = None  # question | selected

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_documents"] = None if self.doc_ids is None else len(self.doc_ids)
        for k in ("doc_ids", "usage", "rewrite_ms"):
            d.pop(k)
        return d


def example_question(docs: list[dict[str, Any]], company: str | None) -> str | None:
    """A question the reports can answer: the picked company (else the first) and its latest period."""
    catalog = build_catalog(docs)
    picked = [c for c in catalog if company and c.name.lower() == company.lower()]
    c = (picked or catalog or [None])[0]
    if c is None:
        return None
    periods = sorted({d.period for d in c.docs if d.period}, key=period_sort_key)
    return f"What was {c.name}'s revenue from operations in {periods[-1]}?" if periods else None


def company_periods(docs: list[dict[str, Any]], company: str | None) -> list[str]:
    """Periods covered for `company` (every company when None), oldest first: context for the rewrite."""
    ps = {
        d.period
        for c in build_catalog(docs)
        if company is None or c.name.lower() == company.lower()
        for d in c.docs
        if d.period
    }
    return sorted(ps, key=period_sort_key)


def _mentions(question: str, company: CatalogCompany) -> bool:
    q = f" {question.lower()} "
    for term in company.terms:
        if len(term) < 2:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?:'s)?(?![a-z0-9])", q):
            return True
    return False


def _period_spellings(period: str) -> list[str]:
    """'Q1 FY26' -> ['Q1 FY26', 'Q1FY26', 'Q1 FY2026']; 'FY26' -> ['FY26', 'FY2026', 'FY 2025-26']."""
    fy = fiscal_year(period) or period
    yy = int(fy[2:])
    if period.startswith(("Q", "H")):
        part = period.split()[0]
        return [period, f"{part}{fy}", f"{part} FY20{yy:02d}"]
    return [fy, f"FY20{yy:02d}", f"FY 20{yy - 1:02d}-{yy:02d}"]


def _matches_period(doc_period: str | None, wanted: str) -> bool:
    if not doc_period:
        return False
    if doc_period == wanted:
        return True
    # A full-year question can also use that year's quarterly documents (Q4 reports carry FY totals).
    return " " not in wanted and fiscal_year(doc_period) == wanted


def enhance(
    question: str,
    docs: list[dict[str, Any]],
    selected_company: str | None = None,
    *,
    plain: bool = False,
) -> Enhancement:
    """`plain`: the enhancer is off. The question is searched exactly as typed, limited only to the company
    the user picked: no company or period detection, no added words."""
    catalog = build_catalog(docs)
    by_name = {c.name.lower(): c for c in catalog}
    if plain:
        picked = by_name.get((selected_company or "").lower())
        return Enhancement(
            search_query=question,
            question=question,
            original=question,
            available_companies=[c.name for c in catalog],
            companies=[picked.name] if picked else [],
            doc_ids=sorted(d.id for d in picked.docs) if picked and len(by_name) > 1 else None,
            company_source="selected" if picked else None,
            enabled=False,
        )

    source = None
    chosen: list[CatalogCompany] = []
    if selected_company and selected_company.lower() in by_name:
        chosen, source = [by_name[selected_company.lower()]], "selected"
    else:
        chosen = [c for c in catalog if _mentions(question, c)]
        source = "question" if chosen else None

    periods = find_periods(question)
    notes: list[str] = []
    pool = [d for c in (chosen or catalog) for d in c.docs]
    scoped = pool
    if periods:
        matched = [d for d in pool if any(_matches_period(d.period, p) for p in periods)]
        missing = [p for p in periods if not any(_matches_period(d.period, p) for d in pool)]
        if matched:
            scoped = matched
        where = " for " + ", ".join(c.name for c in chosen) if chosen else ""
        # Only claim a period is missing when the documents' periods are known at all.
        if missing and any(d.period for d in pool):
            notes.append(f"No report{where} covers {', '.join(missing)}; searched the other reports instead.")

    all_ids = {d["id"] for d in docs}
    scoped_ids = sorted({d.id for d in scoped})
    doc_ids = None if not scoped_ids or set(scoped_ids) >= all_ids else scoped_ids

    extras: list[str] = []
    low = question.lower()
    words = set(_WORD.findall(low))
    expansions = []
    for abbr, full in ABBREVIATIONS.items():
        if abbr in words and full not in low:
            expansions.append(f"{abbr} -> {full}")
            extras.append(full)
    said = low + " " + " ".join(extras)
    for phrase, alts in SYNONYMS.items():
        if phrase in said:
            extras.append(alts)
    for p in periods:
        extras += [s for s in _period_spellings(p) if s.lower() not in low]
    for c in chosen:
        if c.name.lower() not in low:
            extras.append(c.name)

    search_query = question if not extras else f"{question} ({'; '.join(dict.fromkeys(extras))})"
    return Enhancement(
        search_query=search_query,
        question=question,
        original=question,
        available_companies=[c.name for c in catalog],
        companies=[c.name for c in chosen],
        periods=periods,
        doc_ids=doc_ids,
        expansions=expansions,
        notes=notes,
        company_source=source,
    )
