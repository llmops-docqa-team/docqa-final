"""Pure metric functions for the full eval: number match, citation match, rates, confusion matrices, kappa.

No I/O and no LLM calls, so every one of these is unit-tested on hand-made data.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence

from app.answering.numbers import Quantity, extract_numbers
from eval.retrieval_metrics import NEAR, STRICT, GoldPage, HitPage, page_matches

ROUTES = ("DOCUMENT", "GENERAL", "MIXED")


# ---- rates ---------------------------------------------------------------------------------------------
def rate(k: int, n: int) -> dict:
    """{"k": hits, "n": denominator, "value": k/n}. The value is None, not 0, when n is 0."""
    return {"k": k, "n": n, "value": round(k / n, 4) if n else None}


def pct(r: dict | None) -> str:
    """'81.9% (59/72)' / 'n/a'."""
    if not r or r["value"] is None:
        return "n/a"
    return f"{r['value'] * 100:.1f}% ({r['k']}/{r['n']})"


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile, q in [0, 1]. None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def latency_stats(values: Sequence[float]) -> dict:
    return {
        "n": len(values),
        **{
            name: None if (p := percentile(values, q)) is None else round(p, 1)
            for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
        },
    }


# ---- numeric answers -----------------------------------------------------------------------------------
def _quantity_matches(gold: Quantity, got: Quantity) -> bool:
    """Same figure. Exact on Decimals, signs and the percent sign ignored.

    Digits equal: a match unless both sides name a scale and the scales differ ("12 crore" is not "12
    million"). Scaled values equal: a match whatever the words ("12,563 crore" is "125.63 billion")."""
    if gold.value == got.value:
        return True
    if gold.mantissa == got.mantissa:
        return gold.exponent == got.exponent or 0 in (gold.exponent, got.exponent)
    return False


def gold_figures(gold_answer: str | None) -> list[Quantity]:
    """The checkable figures in a gold answer (years, FY codes, dates and single digits are skipped)."""
    return extract_numbers(gold_answer or "", for_answer=True)


def numeric_match(gold_answer: str | None, answer: str | None) -> bool | None:
    """True when every checkable figure in the gold answer appears in the answer.

    None when the gold answer has no checkable figure, so the caller must fall back to the judge. An answer
    that lists extra figures (say standalone and consolidated) still matches if the gold one is in it."""
    golds = gold_figures(gold_answer)
    if not golds:
        return None
    got = extract_numbers(answer or "", for_answer=True)
    return all(any(_quantity_matches(g, a) for a in got) for g in golds)


# ---- citations -----------------------------------------------------------------------------------------
def citation_hits(cited: Sequence[HitPage], golds: Sequence[GoldPage]) -> dict[str, bool]:
    """How the cited pages line up with the gold pages.

    any_strict / any_pm1: at least one cited page is a gold page (same page / within one page).
    all_strict: every cited page is a gold page (no stray citations). All False when nothing is cited."""
    if not cited or not golds:
        return {"any_strict": False, "any_pm1": False, "all_strict": False}
    on = [any(page_matches(c, g, STRICT) for g in golds) for c in cited]
    near = [any(page_matches(c, g, NEAR) for g in golds) for c in cited]
    return {"any_strict": any(on), "any_pm1": any(near), "all_strict": all(on)}


# ---- routing -------------------------------------------------------------------------------------------
def confusion(pairs: Sequence[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """3x3 matrix: matrix[true][predicted]. `pairs` = (true route, predicted route)."""
    m = {t: {p: 0 for p in ROUTES} for t in ROUTES}
    for true, pred in pairs:
        if true in m and pred in m[true]:
            m[true][pred] += 1
    return m


def route_summary(pairs: Sequence[tuple[str, str]]) -> dict:
    """Accuracy, per-class recall (DOCUMENT recall is the one that matters: a document question sent to
    the general path gets an unsourced answer), accuracy on DOCUMENT+GENERAL rows only (so a baseline that
    cannot split MIXED questions is still compared fairly on the rest), and the matrix."""
    m = confusion(pairs)
    correct = sum(m[r][r] for r in ROUTES)
    n = sum(sum(row.values()) for row in m.values())
    recall = {r: rate(m[r][r], sum(m[r].values())) for r in ROUTES}
    dg_n = sum(sum(m[r].values()) for r in ("DOCUMENT", "GENERAL"))
    dg_k = m["DOCUMENT"]["DOCUMENT"] + m["GENERAL"]["GENERAL"]
    return {
        "accuracy": rate(correct, n),
        "accuracy_document_general": rate(dg_k, dg_n),
        "recall": recall,
        "document_recall": recall["DOCUMENT"],
        "confusion": m,
    }


def best_threshold(scores: Sequence[float | None], labels: Sequence[str]) -> tuple[float, int]:
    """The score cut-off tau that routes best when "score >= tau -> DOCUMENT, else GENERAL".

    Returns (tau, number correct). A row with no score (nothing retrieved) is always GENERAL. It looks at
    the labels, so the result is optimistic: that is deliberate, it gives the baseline its best case."""
    cands = sorted({s for s in scores if s is not None})
    if not cands:
        return 0.0, sum(1 for lab in labels if lab == "GENERAL")
    mids = [(a + b) / 2 for a, b in zip(cands, cands[1:], strict=False)]
    taus = [cands[0] - 1e-6, *mids, cands[-1] + 1e-6]
    best_tau, best_k = taus[0], -1
    for tau in taus:
        k = sum(1 for s, lab in zip(scores, labels, strict=True) if score_route(s, tau) == lab)
        if k > best_k:
            best_tau, best_k = tau, k
    return best_tau, best_k


def score_route(top_score: float | None, tau: float) -> str:
    return "DOCUMENT" if top_score is not None and top_score >= tau else "GENERAL"


# ---- agreement -----------------------------------------------------------------------------------------
def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Cohen's kappa for two binary raters. None when it is undefined (no items, or both raters used one
    label throughout, so chance agreement is already 1)."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return None
    return round((po - pe) / (1 - pe), 4)


def agreement(a: Sequence[int], b: Sequence[int]) -> dict:
    n = len(a)
    k = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    return {"n": n, "agree": rate(k, n), "kappa": cohen_kappa(a, b)}


# ---- pacing --------------------------------------------------------------------------------------------
class TokenPacer:
    """Keeps tokens-per-minute under a budget, per model, using only calls that really hit the API.

    After every question the runner reports what it spent (`observe`); before the next one `wait` sleeps
    just long enough that the question's expected cost (the largest single-question spend seen so far for
    that model) fits in the rolling window. Cached replays spend nothing, so a fully cached run never
    waits."""

    def __init__(
        self,
        budget: int,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.budget, self.window = budget, window
        self._clock, self._sleep = clock, sleep
        self._events: dict[str, deque[tuple[float, int]]] = {}
        self._expected: dict[str, int] = {}

    def observe(self, spent: dict[str, int]) -> None:
        now = self._clock()
        for model, tokens in spent.items():
            if tokens <= 0:
                continue
            self._events.setdefault(model, deque()).append((now, tokens))
            self._expected[model] = max(self._expected.get(model, 0), tokens)

    def seconds_to_wait(self) -> float:
        now = self._clock()
        worst = 0.0
        for model, events in self._events.items():
            while events and events[0][0] <= now - self.window:
                events.popleft()
            # An expected cost above the whole budget could never fit: wait for an empty window instead.
            need_free = min(self._expected.get(model, 0), self.budget)
            used = sum(t for _, t in events)
            for ts, tokens in events:
                if used + need_free <= self.budget:
                    break
                worst = max(worst, ts + self.window - now)
                used -= tokens
        return round(worst, 3)

    def wait(self) -> float:
        secs = self.seconds_to_wait()
        if secs > 0:
            self._sleep(secs)
        return secs
