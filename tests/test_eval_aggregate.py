"""The aggregator on hand-made records: every number below was worked out by hand from the scenario."""

from __future__ import annotations

import pytest

from eval import aggregate as ag
from eval.answer_metrics import rate
from tests.evalrecs import rec

META = {"run": "t", "git": "abc", "eval_set_hash": "h", "answer_model": "a", "router_model": "r"}
NARRATIVE = "Yes. The Audit Committee was a board committee in FY2025."


def scenario() -> list[dict]:
    return [
        # table slice, numeric: one right, one wrong figure (and a stray citation), one abstained
        rec("N1", judge={"document": (True, True)}),
        rec(
            "N2",
            answer="Profit was 99.00 million.",
            cited_pages=(12,),
            number_check="fail",
            judge={"document": (False, False)},
        ),
        rec("N3", status="abstained", abstain_reason="insufficient", keyword="GENERAL"),
        # text slice, narrative: judged / not judged yet (and one live call came from the cache)
        rec(
            "T1",
            slice="text",
            type="narrative",
            gold_answer=NARRATIVE,
            pages=(37,),
            cited_pages=(37,),
            answer="Yes, there was an Audit Committee.",
            judge={"document": (True, True)},
        ),
        rec(
            "T2",
            slice="text",
            type="narrative",
            gold_answer=NARRATIVE,
            pages=(37,),
            cited_pages=(37,),
            answer="Yes.",
            cached=1,
        ),
        # unanswerable: one correctly declined, one answered anyway
        rec(
            "U1",
            slice="text",
            type="near_miss",
            answerable=False,
            gold_answer=None,
            pages=(),
            status="abstained",
            abstain_reason="low_score",
        ),
        rec(
            "U2",
            slice="table",
            type="near_miss",
            answerable=False,
            gold_answer=None,
            pages=(),
            answer="Revenue forecast is 1,000 million.",
        ),
        # general: right route and answer / sent to the document path instead
        rec(
            "G1",
            route="GENERAL",
            slice="general",
            answerable=None,
            gold_answer="Paris.",
            pages=(),
            status=None,
            general_status="answered",
            top_score=0.3,
            judge={"general": (True, None)},
        ),
        rec(
            "G2",
            route="GENERAL",
            slice="general",
            answerable=None,
            gold_answer="Paris.",
            pages=(),
            llm_route="DOCUMENT",
            status="abstained",
            abstain_reason="insufficient",
            top_score=0.3,
        ),
        # mixed: both halves right
        rec(
            "M1",
            route="MIXED",
            slice="mixed",
            type="other",
            general_answer="Paris.",
            keyword="DOCUMENT",
            general_status="answered",
            judge={"document": (True, True), "general": (True, None)},
        ),
        # an LLM outage says nothing about the system
        rec("E1", status="error", abstain_reason="llm_unavailable", infra=True),
    ]


@pytest.fixture(scope="module")
def report():
    return ag.build_report(scenario(), META)


# ---------------------------------------------------------------- bookkeeping


def test_outage_rows_are_counted_and_left_out_of_every_rate(report):
    assert report["errors"] == ["E1"] and report["meta"]["n_errors"] == 1
    assert report["meta"]["n_records"] == 11 and report["meta"]["n_full"] == 10
    assert report["router"]["n"] == 10
    assert report["tokens"]["n"] == 10


# ---------------------------------------------------------------- router


def test_router_benchmark_three_variants(report):
    v = report["router"]["variants"]
    llm = v["llm_router"]
    assert llm["accuracy"] == rate(9, 10)  # only G2 was misrouted
    assert llm["document_recall"] == rate(7, 7)
    assert llm["recall"]["GENERAL"] == rate(1, 2) and llm["recall"]["MIXED"] == rate(1, 1)
    assert llm["accuracy_document_general"] == rate(8, 9)
    assert llm["confusion"]["GENERAL"] == {"DOCUMENT": 1, "GENERAL": 1, "MIXED": 0}
    assert llm["fallbacks"] == 0
    kw = v["keyword_rules"]
    assert kw["accuracy"] == rate(8, 10)  # N3 -> GENERAL, M1 -> DOCUMENT
    assert kw["confusion"]["MIXED"]["DOCUMENT"] == 1 and kw["confusion"]["DOCUMENT"]["GENERAL"] == 1
    sc = v["score_threshold"]
    assert sc["accuracy"] == rate(9, 10)  # MIXED can never be predicted
    assert sc["recall"]["MIXED"] == rate(0, 1) and 0.3 < sc["tau"] <= 0.7
    assert "optimistic" in sc["note"]
    assert report["router"]["llm_misrouted"] == [{"id": "G2", "gold": "GENERAL", "predicted": "DOCUMENT"}]


def test_a_router_fallback_is_counted(report):
    r = rec("N9", llm_route="DOCUMENT")
    r["router"].update(ok=False, fallback_reason="bad_json")
    out = ag.build_report([r], META)
    assert out["router"]["variants"]["llm_router"]["fallbacks"] == 1
    assert out["router"]["variants"]["llm_router"]["accuracy"] == rate(
        1, 1
    )  # a fallback to DOCUMENT is a prediction


# ---------------------------------------------------------------- answers


def test_answers_by_slice_table(report):
    t = report["answers"]["by_slice"]["table"]
    assert t["n"] == 3 and t["answered"] == rate(2, 3)
    assert t["answer_accuracy"] == rate(1, 3)  # N1 only: N2 wrong figure, N3 abstained
    assert t["accuracy_when_answered"] == rate(1, 2)
    assert t["groundedness"] == rate(1, 2)  # shadow judge verdicts on numeric rows
    assert t["citation_strict"] == rate(1, 2) and t["citation_pm1"] == rate(1, 2)
    assert t["citation_all_strict"] == rate(1, 2)
    assert t["number_check_fail"] == rate(1, 2)
    assert t["unjudged"] == 0


def test_answers_by_slice_text_counts_the_row_the_judge_has_not_seen(report):
    t = report["answers"]["by_slice"]["text"]
    assert t["n"] == 2 and t["answered"] == rate(2, 2)
    assert t["answer_accuracy"] == rate(1, 1)  # T2 has no verdict yet: left out, not wrong
    assert t["unjudged"] == 1
    assert t["groundedness"] == rate(1, 1)
    assert t["citation_strict"] == rate(2, 2)


def test_overall_answers_and_missing_slices(report):
    o = report["answers"]["overall"]
    assert o["n"] == 5 and o["answered"] == rate(4, 5)
    assert o["answer_accuracy"] == rate(2, 4)  # N1 and T1 of the four that can be told
    assert o["accuracy_when_answered"] == rate(2, 3)
    assert o["unjudged"] == 1
    assert set(report["answers"]["by_slice"]) == {"table", "text"}  # no scanned rows: no scanned entry


def test_judge_agrees_with_the_number_match_on_numeric_rows(report):
    j = report["judge"]
    assert j["model"] == "judge-m" and j["prompt_version"] == "v1"
    assert j["n_verdicts"] == 6 and j["tokens"] == 600
    assert j["vs_numeric_match"]["n"] == 3 and j["vs_numeric_match"]["agree"] == rate(3, 3)  # N1, N2, M1


def test_numeric_rows_never_need_the_judge_for_correctness():
    r = rec("N1")  # no judge verdicts at all
    assert ag.doc_correct(r) is True
    assert ag.doc_correct(rec("N2", answer="Profit was 99.00 million.")) is False
    assert ag.doc_correct(rec("N3", status="abstained")) is False


def test_a_numeric_row_whose_gold_has_no_figure_falls_back_to_the_judge():
    r = rec("N1", gold_answer="Yes.", type="numeric")
    assert ag.doc_correct(r) is None
    r2 = rec("N1", gold_answer="Yes.", type="numeric", judge={"document": (False, True)})
    assert ag.doc_correct(r2) is False


def test_a_narrative_gold_with_incidental_figures_is_still_judged_not_matched():
    r = rec("T1", type="narrative", gold_answer="Three plants, with 1,200 employees.", answer="Three plants.")
    assert ag.uses_numeric_match(r["gold"]) is False
    assert ag.doc_correct(r) is None


# ---------------------------------------------------------------- abstention


def test_abstention_overall(report):
    a = report["abstention"]["overall"]
    assert (a["n_unanswerable"], a["n_answerable"]) == (2, 5)
    assert a["abstain_recall"] == rate(1, 2) and a["false_answer_rate"] == rate(1, 2)
    assert a["abstain_precision"] == rate(1, 2)  # abstained on U1 (right) and N3 (wrong)
    assert a["wrong_abstention_rate"] == rate(1, 5)
    assert a["other_unanswerable"] == 0 and a["other_answerable"] == 0
    assert a["abstain_reasons"] == {"insufficient": 1, "low_score": 1}


def test_abstention_by_slice(report):
    text, table = report["abstention"]["by_slice"]["text"], report["abstention"]["by_slice"]["table"]
    assert text["abstain_recall"] == rate(1, 1) and text["false_answer_rate"] == rate(0, 1)
    assert table["abstain_recall"] == rate(0, 1) and table["false_answer_rate"] == rate(1, 1)


def test_an_unanswerable_question_misrouted_away_is_other_not_a_false_answer():
    r = rec(
        "U9",
        type="near_miss",
        answerable=False,
        gold_answer=None,
        pages=(),
        llm_route="GENERAL",
        status=None,
        general_status="answered",
    )
    a = ag.build_report([r], META)["abstention"]["overall"]
    assert a["other_unanswerable"] == 1
    assert a["abstain_recall"] == rate(0, 1) and a["false_answer_rate"] == rate(0, 1)


# ---------------------------------------------------------------- general and mixed


def test_general_block_counts_a_misrouted_question_as_wrong_not_missing(report):
    g = report["general"]
    assert g["n"] == 2 and g["routed_general"] == rate(1, 2) and g["answered"] == rate(1, 2)
    assert g["answer_accuracy"] == rate(1, 2) and g["unjudged"] == 0


def test_mixed_block(report):
    m = report["mixed"]
    assert m["n"] == 1 and m["routed_mixed"] == rate(1, 1)
    assert m["document_part_correct"] == rate(1, 1) and m["general_part_correct"] == rate(1, 1)
    assert m["both_correct"] == rate(1, 1) and m["unjudged"] == 0


def test_mixed_with_an_unjudged_general_half_is_unjudged():
    r = rec("M1", route="MIXED", slice="mixed", type="other", general_status="answered")
    m = ag.build_report([r], META)["mixed"]
    assert m["document_part_correct"] == rate(1, 1) and m["general_part_correct"] == rate(0, 0)
    assert m["both_correct"] == rate(0, 0) and m["unjudged"] == 1


# ---------------------------------------------------------------- retrieval, latency, tokens


def test_retrieval_uses_the_ranks_of_answerable_document_rows(report):
    o = report["retrieval"]["overall"]
    assert o["n"] == 5 and o["recall_at_1"] == 1.0 and o["mrr"] == 1.0
    assert set(report["retrieval"]["by_slice"]) == {"table", "text"}


def test_retrieval_ranks_that_missed_pull_the_numbers_down():
    rows = [rec("A", strict_rank=1), rec("B", strict_rank=4), rec("C", strict_rank=None)]
    o = ag.build_report(rows, META)["retrieval"]["overall"]
    assert o["recall_at_1"] == pytest.approx(1 / 3, abs=1e-4) and o["recall_at_5"] == pytest.approx(
        2 / 3, abs=1e-4
    )
    assert o["mrr"] == pytest.approx((1 + 0.25) / 3, abs=1e-4)


def test_latency_leaves_out_questions_with_a_cached_call_and_stages_that_did_not_run(report):
    lat = report["latency"]
    assert lat["n_live"] == 9 and lat["n_cached_excluded"] == 1  # T2 had a cached call
    assert lat["stages"]["router_ms"]["n"] == 9 and lat["stages"]["router_ms"]["p50"] == 600.0
    assert lat["stages"]["general_ms"]["n"] == 0  # never ran: not a zero
    assert lat["stages"]["total_ms"]["p99"] == 3000.0


def test_an_all_cached_run_has_no_latency_numbers():
    rows = [rec("A", cached=2, live=0), rec("B", cached=1, live=1)]
    lat = ag.build_report(rows, META)["latency"]
    assert lat["n_live"] == 0 and lat["n_cached_excluded"] == 2
    assert "none: every question was replayed" in ag.format_report(ag.build_report(rows, META))


def test_tokens_and_cost(report):
    t = report["tokens"]
    assert t["sum"]["total"]["total"] == 30000 and t["mean_total_per_question"] == 3000.0
    assert t["cost_usd_equiv"] == pytest.approx(0.005) and t["cost_usd_equiv_per_question"] == pytest.approx(
        0.0005
    )


def test_failures_list_the_answerable_questions_that_were_not_right(report):
    f = {x["id"]: x for x in report["failures"]}
    assert set(f) == {"N2", "N3"}  # T2 is unjudged, not a failure
    assert f["N2"]["outcome"] == "answered" and f["N3"]["outcome"] == "abstained"
    assert f["N3"]["reason"] == "insufficient" and f["N2"]["answer"] == "Profit was 99.00 million."


# ---------------------------------------------------------------- router-only records, empties, output


def test_router_only_records_feed_the_router_and_retrieval_but_not_the_answers():
    rows = [rec("A", mode="router", status=None), rec("B", llm_route="GENERAL")]
    r = ag.build_report(rows, META)
    assert r["router"]["n"] == 2 and r["meta"]["n_full"] == 1
    assert r["answers"]["overall"]["n"] == 1
    assert r["retrieval"]["overall"]["n"] == 2
    text = ag.format_report(ag.build_report([rec("A", mode="router", status=None)], META))
    assert "router-only run" in text and "ANSWERS" not in text


def test_an_empty_run_builds_a_report_instead_of_crashing():
    r = ag.build_report([], META)
    assert r["router"]["n"] == 0 and r["answers"]["overall"]["n"] == 0
    assert "router-only run" in ag.format_report(r)
    assert "Router" not in ag.format_markdown(r)


def test_flatten_metrics_gives_mlflow_plain_floats(report):
    m = ag.flatten_metrics(report)
    assert m["answers.overall.answer_accuracy"] == 0.5 and m["answers.overall.answer_accuracy.n"] == 4
    assert m["router.variants.llm_router.accuracy"] == 0.9
    assert m["router.variants.llm_router.recall.MIXED"] == 1.0
    assert m["abstention.overall.false_answer_rate"] == 0.5
    assert m["latency.stages.router_ms.p50"] == 600.0 and m["tokens.cost_usd_equiv"] == pytest.approx(0.005)
    assert all(isinstance(v, float) for v in m.values())
    assert not any("confusion" in k or k.startswith("meta") for k in m)
    assert "answers.by_slice.scanned.answer_accuracy" not in m


def test_the_console_report_names_the_numbers_a_reader_looks_for(report):
    text = ag.format_report(report)
    for needle in (
        "ROUTER",
        "keyword_rules",
        "score_threshold",
        "llm_router",
        "DOCUMENT recall",
        "confusion, llm_router",
        "RETRIEVAL",
        "ANSWERS",
        "table",
        "text",
        "scanned",
        "no READY document for this slice",
        "ABSTENTION",
        "false-answer rate",
        "GENERAL (2)",
        "MIXED (1)",
        "LATENCY",
        "TOKENS",
        "JUDGE",
        "NOT CORRECT",
        "1 question(s) hit an LLM outage",
        "50.0% (2/4)",
    ):
        assert needle in text, needle
    assert "* = fewer than 10 questions" in text


def test_the_markdown_has_the_three_readme_tables(report):
    md = ag.format_markdown(report)
    assert "| LLM router |" in md and "| Keyword rules |" in md and "| Retrieval-score threshold |" in md
    assert "| table | 3 |" in md and "| scanned | 0 | n/a |" in md
    assert "**Abstention**" in md and "**Latency**" in md


def test_a_baseline_that_is_already_good_is_called_out_only_with_enough_questions():
    rows = [rec(f"D{i}") for i in range(12)] + [
        rec(
            f"G{i}",
            route="GENERAL",
            slice="general",
            answerable=None,
            gold_answer="x",
            pages=(),
            status=None,
            general_status="answered",
            top_score=0.3,
        )
        for i in range(3)
    ]
    text = ag.format_report(ag.build_report(rows, META))
    assert "NOTE: the keyword_rules baseline already reaches 100.0%" in text
    small = ag.format_report(ag.build_report(rows[:3], META))
    assert "NOTE: the keyword_rules baseline" not in small


def test_abstain_precision_counts_every_abstention_not_just_the_unanswerable_ones():
    rows = [
        rec(
            "U1",
            type="near_miss",
            answerable=False,
            gold_answer=None,
            pages=(),
            status="abstained",
            abstain_reason="insufficient",
        )
    ]
    rows += [rec(f"A{i}", status="abstained", abstain_reason="insufficient") for i in range(3)]
    a = ag.build_report(rows, META)["abstention"]["overall"]
    assert a["abstain_recall"] == rate(1, 1) and a["false_answer_rate"] == rate(0, 1)
    assert a["abstain_precision"] == rate(1, 4)  # 4 abstentions, 1 of them right
    assert a["wrong_abstention_rate"] == rate(3, 3)
