"""The catalog: which companies, report types and periods the uploaded documents cover.

Every document gets a company, a report type and a period. They come from the file name (e.g.
"TCS Result Q1FY26.pdf" -> TCS / Quarterly results / Q1 FY26) when the upload is accepted, and can
be corrected later (PATCH /documents/{id}); a value stored on the row wins over the file-name guess.
No LLM is involved, so it costs nothing and shows up the moment a document is uploaded.

`build_catalog` groups the documents by company for the UI dropdown and the query enhancer.
Pure functions only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.storage import documents as st

ANNUAL, QUARTERLY, PRESENTATION, TRANSCRIPT, OTHER = (
    "Annual report",
    "Quarterly results",
    "Investor presentation",
    "Earnings call transcript",
    "Other",
)

# File-name words (lower case) that name a report type. The first one found ends the company name.
_TYPE_WORDS: dict[str, str] = {
    "ar": ANNUAL, "annual": ANNUAL, "annualreport": ANNUAL,
    "result": QUARTERLY, "results": QUARTERLY, "qr": QUARTERLY, "quarterly": QUARTERLY,
    "ip": PRESENTATION, "presentation": PRESENTATION, "investor": PRESENTATION,
    "ect": TRANSCRIPT, "transcript": TRANSCRIPT, "concall": TRANSCRIPT, "call": TRANSCRIPT,
    "earnings": TRANSCRIPT,
}

# Q1FY26, Q1 FY2026, Q1-FY26, H1FY26 | FY25, FY 2025, FY2024-25
_PERIOD = re.compile(
    r"\b(?:(?P<part>[QH][1-4])[\s_-]*)?FY[\s_-]*(?P<y1>(?:19|20)?\d{2})(?:[\s_-]*[-/][\s_-]*(?P<y2>\d{2,4}))?\b",
    re.IGNORECASE,
)
_COPY_SUFFIX = re.compile(r"\s*[-_(]\s*copy(?:\s*\(?\d+\)?)?\)?\s*$", re.IGNORECASE)
_NOISE = re.compile(r"[_\s]+")


def _fy(y1: str, y2: str | None) -> str:
    """Two-digit financial year: 'FY25' from '25', '2025' or '2024-25' (the year the FY ends)."""
    year = y2 or y1
    return year[-2:]


def normalize_period(text: str) -> str | None:
    """'q1fy2026' -> 'Q1 FY26'; 'FY 2024-25' -> 'FY25'. None when the text holds no period."""
    m = _PERIOD.search(text or "")
    if not m:
        return None
    fy = f"FY{_fy(m.group('y1'), m.group('y2'))}"
    part = m.group("part")
    return f"{part.upper()} {fy}" if part else fy


def find_periods(text: str) -> list[str]:
    """Every distinct canonical period mentioned in `text`, in order of appearance."""
    out: list[str] = []
    for m in _PERIOD.finditer(text or ""):
        fy = f"FY{_fy(m.group('y1'), m.group('y2'))}"
        p = f"{m.group('part').upper()} {fy}" if m.group("part") else fy
        if p not in out:
            out.append(p)
    return out


def fiscal_year(period: str | None) -> str | None:
    """'Q1 FY26' -> 'FY26'; 'FY26' -> 'FY26'."""
    if not period:
        return None
    m = re.search(r"FY\d{2}", period)
    return m.group(0) if m else None


def period_sort_key(period: str) -> tuple[int, int]:
    """Oldest first: FY25 < Q1 FY26 < Q4 FY26 < FY26 (a full year sorts after its own quarters)."""
    m = re.search(r"FY(\d{2})", period or "")
    year = int(m.group(1)) if m else 0
    q = re.match(r"[QH](\d)", period or "")
    return (year, int(q.group(1)) if q else 9)


@dataclass(frozen=True)
class DocMeta:
    company: str
    company_tag: str  # the file-name prefix ("EIG"): kept even if the display name is corrected
    report_type: str
    period: str | None


def parse_filename(filename: str) -> DocMeta:
    stem = _COPY_SUFFIX.sub("", Path(filename or "").stem)
    stem = _NOISE.sub(" ", stem).strip()
    period = normalize_period(stem)
    head = _PERIOD.sub(" ", stem)
    words = head.split()
    report_type = OTHER
    company_words: list[str] = []
    seen_type = False
    for w in words:
        t = _TYPE_WORDS.get(re.sub(r"\W", "", w).lower())
        if t and not seen_type:
            report_type, seen_type = t, True
            continue
        if not seen_type:
            company_words.append(w)
    company = " ".join(company_words).strip(" -_.") or " ".join(words).strip(" -_.") or "Unknown company"
    return DocMeta(company=company, company_tag=company, report_type=report_type, period=period)


def doc_meta(doc: dict[str, Any]) -> DocMeta:
    """The row's stored values, falling back to what the file name says."""
    guess = parse_filename(doc.get("filename") or "")
    return DocMeta(
        company=(doc.get("company") or "").strip() or guess.company,
        company_tag=guess.company_tag,
        report_type=(doc.get("report_type") or "").strip() or guess.report_type,
        period=(doc.get("period") or "").strip() or guess.period,
    )


def acronym(name: str) -> str:
    skip = {"ltd", "limited", "inc", "plc", "pvt", "the", "of"}
    words = [w for w in re.split(r"\W+", name) if w and w.lower() not in skip]
    return "".join(w[0] for w in words).upper() if len(words) > 1 else ""


@dataclass
class CatalogDoc:
    id: str
    filename: str
    status: str
    report_type: str
    period: str | None
    queryable: bool


@dataclass
class CatalogCompany:
    name: str
    terms: list[str]  # lower-case words/phrases that mean this company in a question
    docs: list[CatalogDoc] = field(default_factory=list)

    def report_types(self) -> dict[str, list[str]]:
        """report type -> periods, oldest first (documents without a period count under '')."""
        out: dict[str, set[str]] = {}
        for d in self.docs:
            out.setdefault(d.report_type, set()).add(d.period or "")
        return {t: sorted(ps, key=period_sort_key) if ps != {""} else [] for t, ps in sorted(out.items())}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "documents": [d.__dict__ for d in self.docs],
            "report_types": [
                {"type": t, "periods": [p for p in ps if p]} for t, ps in self.report_types().items()
            ],
            "n_documents": len(self.docs),
            "n_ready": sum(1 for d in self.docs if d.status == st.READY),
        }


def _key(name: str) -> str:
    return re.sub(r"\W+", "", name).lower()


def build_catalog(docs: list[dict[str, Any]]) -> list[CatalogCompany]:
    """Companies, A-Z, each with its documents oldest period first. Failed documents are left out."""
    companies: dict[str, CatalogCompany] = {}
    for d in docs:
        if d.get("status") == st.FAILED:
            continue
        m = doc_meta(d)
        c = companies.setdefault(_key(m.company), CatalogCompany(m.company, []))
        for term in (m.company, m.company_tag, acronym(m.company)):
            t = term.strip().lower()
            if t and t not in c.terms:
                c.terms.append(t)
        c.docs.append(
            CatalogDoc(
                d["id"], d["filename"], d["status"], m.report_type, m.period, d["status"] in st.QUERYABLE
            )
        )
    for c in companies.values():
        c.docs.sort(key=lambda x: (period_sort_key(x.period or ""), x.report_type, x.filename))
    return sorted(companies.values(), key=lambda c: c.name.lower())


def catalog_response(docs: list[dict[str, Any]]) -> dict:
    companies = build_catalog(docs)
    return {"companies": [c.to_dict() for c in companies], "n_companies": len(companies)}
