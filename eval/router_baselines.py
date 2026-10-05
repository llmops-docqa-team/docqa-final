"""The two router baselines the LLM router is benchmarked against (design §9).

(a) `keyword_route`: hand-written rules on the question text. Free, instant, transparent.
(b) `score_route` (in answer_metrics.py): "is the best retrieved chunk similar enough to the question?".
    Free byproduct of retrieval; it cannot split a MIXED question.

They are small on purpose and were written from the design's description of each option, not tuned on the
eval set. Rules cannot be perfect, but they should not be strawmen: they know fiscal-year patterns, document
names, report vocabulary, definition phrasing and how a two-part question is usually joined.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# A question that names a period or the document itself is about the document.
_STRONG = re.compile(
    r"\b(?:fy\s?-?\d{2,4}|(?:19|20)\d{2}|annual report|the report|this report|the document|the pdf|"
    r"uploaded|according to|as per|per the|in the (?:report|document|pdf|filing)|the company|"
    r"the company's|page\s+\d+)\b",
    re.IGNORECASE,
)
# Report vocabulary: weaker evidence, because a definition question uses the same words.
_WEAK = re.compile(
    r"\b(?:revenue|turnover|profit|loss|margin|ebitda|ebit|assets|liabilit\w+|equity|dividend|cash flow|"
    r"capex|borrowings?|employees?|headcount|directors?|board|committee|auditors?|shareholders?|"
    r"subsidiar\w+|segment|plants?|capacity|shares?|eps|tax|depreciation|expenses?|income|sales)\b",
    re.IGNORECASE,
)
# "What is X?" style: asks for a meaning, not a fact about a document.
_DEFINITION = re.compile(
    r"^\s*(?:what\s+(?:is|are|does|do)\b|what's|define|explain|meaning of|"
    r"who (?:is|was|wrote|invented|discovered)|where is|when (?:was|did)|how (?:does|do) \w+ work)",
    re.IGNORECASE,
)
# Where a two-part question is usually joined: "X?  Y?", "X; Y", "X, and what ...", "X and who ...".
_SPLIT = re.compile(
    r"\?\s+(?=\S)|;\s*|,\s*and\s+(?=\w+\s+(?:is|are|was|were|did|does|do|has|have)\b|"
    r"(?:what|who|when|where|which|how|why)\b)|\s+and\s+(?=(?:what|who|when|where|which|how|why)\b)",
    re.IGNORECASE,
)
_STOP = {"pdf", "report", "annual", "ar", "the", "and", "of", "for", "final", "copy", "doc", "document"}


def doc_terms(filenames: Sequence[str]) -> set[str]:
    """Distinctive words in the uploaded file names ("EIG AR FY25.pdf" -> {"eig"}): a question that uses one
    is about that document."""
    terms: set[str] = set()
    for name in filenames:
        for word in re.split(r"[^A-Za-z0-9]+", name.lower()):
            if len(word) >= 3 and word not in _STOP and not re.fullmatch(r"fy\d*|\d+", word):
                terms.add(word)
    return terms


def _clauses(question: str) -> list[str]:
    return [c.strip() for c in _SPLIT.split(question) if c and c.strip(" ?.")]


def _is_document_like(clause: str, terms: set[str]) -> bool:
    words = set(re.findall(r"[a-z0-9]+", clause.lower()))
    if _STRONG.search(clause) or words & terms:
        return True
    if _DEFINITION.match(clause):
        return False
    return bool(_WEAK.search(clause))


def keyword_route(question: str, filenames: Sequence[str] = ()) -> str:
    """DOCUMENT | GENERAL | MIXED from the words alone.

    Two or more clauses with at least one document-like and one general clause -> MIXED. Otherwise a period,
    the document's name or the report's vocabulary -> DOCUMENT, unless it is phrased as a definition."""
    terms = doc_terms(filenames)
    clauses = _clauses(question)
    if len(clauses) >= 2:
        kinds = [_is_document_like(c, terms) for c in clauses]
        if any(kinds) and not all(kinds):
            return "MIXED"
    return "DOCUMENT" if _is_document_like(question, terms) else "GENERAL"
