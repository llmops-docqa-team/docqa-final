"""Pure metric functions for retrieval: hit rank, Recall@k, MRR, with strict and +-1 page matching.

A retrieved chunk is a hit if it is in the right document and on a gold page. "Strict" means the same
page; "+-1" also accepts the neighbouring pages (an answer that straddles a page break, or a gold page
that is off by one). Pages are compared by 1-based PDF index; if a gold page has no `pdf_page`, the
printed label is compared instead.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

STRICT, NEAR = 0, 1   # page tolerance


@dataclass(frozen=True)
class GoldPage:
    doc: str                 # eval doc key (e.g. "report_a")
    pdf_page: int | None
    label: str


@dataclass(frozen=True)
class HitPage:
    doc: str
    pdf_page: int
    label: str


def _label_int(label: str) -> int | None:
    try:
        return int(str(label).strip())
    except ValueError:
        return None


def page_matches(hit: HitPage, gold: GoldPage, tolerance: int = STRICT) -> bool:
    if hit.doc != gold.doc:
        return False
    if gold.pdf_page is not None:
        return abs(hit.pdf_page - gold.pdf_page) <= tolerance
    if str(hit.label).strip() == str(gold.label).strip():
        return True
    if tolerance:
        a, b = _label_int(hit.label), _label_int(gold.label)
        return a is not None and b is not None and abs(a - b) <= tolerance
    return False


def first_hit_rank(hits: Sequence[HitPage], golds: Sequence[GoldPage], tolerance: int = STRICT) -> int | None:
    """1-based rank of the first retrieved chunk on a gold page, or None if there is none."""
    for rank, hit in enumerate(hits, start=1):
        if any(page_matches(hit, g, tolerance) for g in golds):
            return rank
    return None


def recall_at_k(ranks: Iterable[int | None], k: int) -> float:
    """Share of questions whose first hit is within the top k. `ranks` has one entry per question."""
    ranks = list(ranks)
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: Iterable[int | None]) -> float:
    ranks = list(ranks)
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


@dataclass
class QuestionResult:
    id: str
    slice: str
    strict_rank: int | None
    near_rank: int | None


def summarize(results: Sequence[QuestionResult], ks: Sequence[int] = (1, 3, 5, 8)) -> dict:
    """One dict: n, recall_at_{k} (strict), recall_at_5_pm1, mrr, mrr_pm1."""
    strict = [r.strict_rank for r in results]
    near = [r.near_rank for r in results]
    out: dict = {"n": len(results)}
    for k in sorted(set(ks) | {5}):
        out[f"recall_at_{k}"] = round(recall_at_k(strict, k), 4)
    out["recall_at_5_pm1"] = round(recall_at_k(near, 5), 4)
    out["mrr"] = round(mrr(strict), 4)
    out["mrr_pm1"] = round(mrr(near), 4)
    return out


def summarize_by_slice(results: Sequence[QuestionResult], ks: Sequence[int] = (1, 3, 5, 8)) -> dict:
    """{"overall": {...}, "by_slice": {"text": {...}, "table": {...}, ...}}"""
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        groups[r.slice].append(r)
    return {
        "overall": summarize(results, ks),
        "by_slice": {s: summarize(rs, ks) for s, rs in sorted(groups.items())},
    }
