"""Pure metric functions of the full eval: number match, citations, rates, routing, kappa, pacing."""

from __future__ import annotations

import pytest

from eval.answer_metrics import (
    TokenPacer,
    agreement,
    best_threshold,
    citation_hits,
    cohen_kappa,
    confusion,
    latency_stats,
    numeric_match,
    pct,
    percentile,
    rate,
    route_summary,
    score_route,
)
from eval.retrieval_metrics import GoldPage, HitPage

# ---------------------------------------------------------------- rates


def test_rate_is_none_not_zero_when_there_is_nothing_to_divide_by():
    assert rate(3, 4) == {"k": 3, "n": 4, "value": 0.75}
    assert rate(0, 0)["value"] is None
    assert pct(rate(59, 72)) == "81.9% (59/72)"
    assert pct(rate(0, 0)) == "n/a" and pct(None) == "n/a"


def test_percentile_interpolates_and_handles_no_data():
    assert percentile([], 0.5) is None
    assert percentile([5], 0.99) == 5
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10, 20], 0.95) == pytest.approx(19.5)
    stats = latency_stats([100, 200, 300])
    assert stats == {"n": 3, "p50": 200.0, "p95": 290.0, "p99": 298.0}
    assert latency_stats([])["p50"] is None


# ---------------------------------------------------------------- numeric match

GOLD = "Rs 3,124.83 million (standalone, FY2025)"


@pytest.mark.parametrize(
    "answer",
    [
        "Revenue from operations in FY2025 was 3,124.83 million rupees.",
        "₹3,124.83 million",
        "3124.83 million",  # grouping is not required to match
        "It was `3,124.83 million` (standalone).",  # narrow no-break space from the PDF
        "Standalone: 3,124.83 million; consolidated: 3,500.10 million.",  # extra figures do not hurt
        "Revenue was 3.12483 billion.",  # same value, different scale word
    ],
)
def test_numeric_match_accepts_the_same_figure(answer):
    assert numeric_match(GOLD, answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "Revenue was 3,124.84 million.",  # off by a hundredth: exact, no tolerance
        "Revenue was 3,124.83 crore.",  # same digits, a different scale named on both sides
        "I couldn't find this in your documents.",
        "",
        None,
    ],
)
def test_numeric_match_rejects_a_different_figure(answer):
    assert numeric_match(GOLD, answer) is False


def test_numeric_match_digits_only_answer_matches_when_one_side_has_no_scale():
    assert numeric_match("Rs 832.89 million (standalone, FY2025)", "Profit after tax was 832.89.") is True
    assert numeric_match("832.89", "Profit was 832.89 million.") is True


def test_numeric_match_needs_every_gold_figure_and_ignores_years_and_signs():
    gold = "Rs 100.50 million in FY2025 and Rs 80.25 million in FY2024"
    assert numeric_match(gold, "FY2025: 100.50 million; FY2024: 80.25 million") is True
    assert numeric_match(gold, "FY2025: 100.50 million") is False
    assert numeric_match("Rs (692.21) million (standalone, FY2025)", "A loss of 692.21 million") is True


def test_numeric_match_is_none_when_the_gold_answer_has_no_checkable_figure():
    assert numeric_match("Yes. The Audit Committee was a board committee in FY2025.", "Yes.") is None
    assert numeric_match(None, "anything") is None
    assert numeric_match("", "anything") is None


def test_numeric_match_percent_sign_is_not_required():
    assert numeric_match("35.12%", "The EBITDA margin was 35.12 per cent") is True
    assert numeric_match("35.12%", "The margin was 35.12") is True


# ---------------------------------------------------------------- citations


def hit(page, doc="a", label=None):
    return HitPage(doc, page, label or str(page))


def gold(page, doc="a"):
    return GoldPage(doc, page, str(page))


def test_citation_hits_strict_pm1_and_all():
    golds = [gold(56)]
    assert citation_hits([hit(56)], golds) == {"any_strict": True, "any_pm1": True, "all_strict": True}
    assert citation_hits([hit(57)], golds) == {"any_strict": False, "any_pm1": True, "all_strict": False}
    assert citation_hits([hit(58)], golds) == {"any_strict": False, "any_pm1": False, "all_strict": False}
    # one right page plus a stray one: found, but not clean
    assert citation_hits([hit(56), hit(12)], golds) == {
        "any_strict": True,
        "any_pm1": True,
        "all_strict": False,
    }


def test_citation_hits_respects_the_document_and_empty_inputs():
    assert citation_hits([hit(56, doc="b")], [gold(56, doc="a")])["any_strict"] is False
    assert citation_hits([], [gold(1)]) == {"any_strict": False, "any_pm1": False, "all_strict": False}
    assert citation_hits([hit(1)], [])["any_strict"] is False
    # a gold page without a pdf index falls back to the printed label
    labelled = GoldPage("a", None, "xii")
    assert citation_hits([HitPage("a", 14, "xii")], [labelled])["any_strict"] is True


# ---------------------------------------------------------------- routing


def test_confusion_matrix_is_true_by_predicted():
    m = confusion(
        [("DOCUMENT", "DOCUMENT"), ("DOCUMENT", "GENERAL"), ("MIXED", "DOCUMENT"), ("GENERAL", "GENERAL")]
    )
    assert m["DOCUMENT"] == {"DOCUMENT": 1, "GENERAL": 1, "MIXED": 0}
    assert m["MIXED"] == {"DOCUMENT": 1, "GENERAL": 0, "MIXED": 0}
    assert m["GENERAL"]["GENERAL"] == 1
    assert confusion([("DOCUMENT", "NONSENSE")])["DOCUMENT"] == {"DOCUMENT": 0, "GENERAL": 0, "MIXED": 0}


def test_route_summary_accuracy_recall_and_the_document_general_view():
    pairs = (
        [("DOCUMENT", "DOCUMENT")] * 8
        + [("DOCUMENT", "GENERAL")] * 2
        + [("GENERAL", "GENERAL")] * 5
        + [("MIXED", "DOCUMENT")] * 3
        + [("MIXED", "MIXED")] * 2
    )
    s = route_summary(pairs)
    assert s["accuracy"] == rate(15, 20)
    assert s["document_recall"] == rate(8, 10)
    assert s["recall"]["GENERAL"] == rate(5, 5) and s["recall"]["MIXED"] == rate(2, 5)
    assert s["accuracy_document_general"] == rate(13, 15)  # MIXED rows left out
    assert route_summary([])["accuracy"]["value"] is None


def test_best_threshold_finds_the_best_cut_and_treats_missing_scores_as_general():
    scores = [0.9, 0.8, 0.5, 0.4, None]
    labels = ["DOCUMENT", "DOCUMENT", "GENERAL", "GENERAL", "GENERAL"]
    tau, correct = best_threshold(scores, labels)
    assert correct == 5 and 0.5 < tau <= 0.8
    assert score_route(0.9, tau) == "DOCUMENT" and score_route(0.4, tau) == "GENERAL"
    assert score_route(None, 0.0) == "GENERAL"
    # when scores do not separate the classes, the best cut is just the best available
    tau, correct = best_threshold([0.7, 0.7, 0.7, 0.7], ["DOCUMENT", "GENERAL", "DOCUMENT", "DOCUMENT"])
    assert correct == 3
    assert best_threshold([], []) == (0.0, 0)
    assert best_threshold([None], ["GENERAL"]) == (0.0, 1)


def test_a_score_baseline_can_never_predict_mixed():
    tau, _ = best_threshold([0.9, 0.2], ["MIXED", "GENERAL"])
    assert {score_route(s, tau) for s in (0.9, 0.2, None)} <= {"DOCUMENT", "GENERAL"}


# ---------------------------------------------------------------- agreement


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0
    # 8 agree of 10; each rater says 1 six times: po 0.8, pe 0.52 -> kappa 0.5833
    a = [1] * 6 + [0] * 4
    b = [1] * 5 + [0] + [1] + [0] * 3
    assert cohen_kappa(a, b) == pytest.approx(0.5833, abs=1e-3)


def test_cohen_kappa_is_none_when_undefined():
    assert cohen_kappa([], []) is None
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None  # one label throughout: chance agreement is 1
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 0])


def test_agreement_bundles_percent_and_kappa():
    r = agreement([1, 0, 1, 1], [1, 0, 0, 1])
    assert r["n"] == 4 and r["agree"] == rate(3, 4) and r["kappa"] == pytest.approx(0.5)


# ---------------------------------------------------------------- pacing


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, secs):
        self.slept.append(secs)
        self.t += secs


def pacer(clock, budget=6500):
    return TokenPacer(budget, clock=clock, sleep=clock.sleep)


def test_pacer_does_not_wait_before_it_knows_what_a_question_costs():
    c = Clock()
    assert pacer(c).wait() == 0.0 and c.slept == []


def test_pacer_waits_for_the_oldest_spend_to_leave_the_window():
    c = Clock()
    p = pacer(c)  # budget 6,500 tokens per 60 s
    p.observe({"m": 2500})  # t=1000
    c.t += 10
    p.observe({"m": 2500})  # t=1010: 5,000 used; one more question (2,500) would reach 7,500
    c.t += 5  # t=1015
    # Needs 1,000 tokens freed: the t=1000 spend leaves the window at t=1060, i.e. in 45 s.
    assert p.seconds_to_wait() == pytest.approx(45.0)
    assert p.wait() == pytest.approx(45.0) and c.slept == [pytest.approx(45.0)]
    assert p.seconds_to_wait() == 0.0  # now 5,000 -> 2,500 in the window: room again


def test_pacer_budgets_each_model_separately():
    c = Clock()
    p = pacer(c)
    for _ in range(2):
        p.observe({"small": 1100, "big": 2500})
        c.t += 1
    # "big" has used 5,000 of 6,500 and its next question costs 2,500, so it must wait; "small" has not.
    assert p.seconds_to_wait() > 0
    only_small = pacer(Clock())
    for _ in range(2):
        only_small.observe({"small": 1100})
    assert only_small.seconds_to_wait() == 0.0  # 2,200 used + 1,100 expected is far below the budget


def test_pacer_ignores_cached_replays_and_never_waits_forever():
    c = Clock()
    p = pacer(c, budget=1000)
    p.observe({"m": 0})  # a cached replay spends nothing
    assert p.seconds_to_wait() == 0.0
    p.observe({"m": 5000})  # one question costs more than the whole budget
    c.t += 20
    assert p.seconds_to_wait() == pytest.approx(40.0)  # wait for an empty window, not for the impossible
    c.t += 41
    assert p.seconds_to_wait() == 0.0
