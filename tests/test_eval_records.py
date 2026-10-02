"""The run file, the resume rules, and what gets judged."""

from __future__ import annotations

import json

import pytest

from app.config import load_settings
from eval.records import (
    RunFile,
    fingerprint_diff,
    infra_error,
    judge_needed,
    judge_targets,
    pipeline_fingerprint,
    plan,
    row_hash,
    verdict_of,
)
from tests.evalrecs import rec, row

# ---------------------------------------------------------------- the run file


def test_run_file_round_trips_and_the_last_line_for_an_id_wins(tmp_path):
    f = RunFile(tmp_path / "runs" / "r.jsonl")
    assert f.load() == (None, {})
    f.write_header({"pipeline": {"theta": 0.5}})
    f.append({"id": "A", "v": 1})
    f.append({"id": "B", "v": 1})
    f.append({"id": "A", "v": 2})  # a later line replaces the earlier record (how a verdict is added)
    header, records = f.load()
    assert header == {"pipeline": {"theta": 0.5}}
    assert records == {"A": {"id": "A", "v": 2}, "B": {"id": "B", "v": 1}}


def test_a_torn_last_line_is_ignored_and_does_not_corrupt_the_next_record(tmp_path):
    f = RunFile(tmp_path / "r.jsonl")
    f.write_header({"x": 1})
    f.append({"id": "A", "v": 1})
    with f.path.open("ab") as fh:
        fh.write(b'{"id": "B", "v": ')  # the process died mid-write: no closing brace, no newline
    f.append({"id": "C", "v": 3})
    _, records = f.load()
    assert set(records) == {"A", "C"} and records["C"] == {"id": "C", "v": 3}


def test_numpy_style_scalars_and_odd_values_are_written_not_raised(tmp_path):
    class Scalar:
        def item(self):
            return 0.5

    f = RunFile(tmp_path / "r.jsonl")
    f.append({"id": "A", "score": Scalar(), "path": tmp_path})
    _, records = f.load()
    assert records["A"]["score"] == 0.5 and isinstance(records["A"]["path"], str)


def test_a_header_can_be_replaced_and_delete_removes_the_file(tmp_path):
    f = RunFile(tmp_path / "r.jsonl")
    f.write_header({"n": 1})
    f.write_header({"n": 2})
    assert f.load()[0] == {"n": 2}
    f.delete()
    f.delete()  # twice is fine
    assert not f.path.exists()


def test_garbage_lines_are_skipped(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('not json\n[1, 2]\n{"id": "A"}\n\n{"no_id": 1}\n', encoding="utf-8")
    assert RunFile(p).load() == (None, {"A": {"id": "A"}})


# ---------------------------------------------------------------- fingerprints


def test_fingerprint_changes_with_anything_that_changes_what_a_result_means():
    s = load_settings()
    base = pipeline_fingerprint(s, {"report_a": "d1"}, "emb-1")
    assert fingerprint_diff(base, pipeline_fingerprint(s, {"report_a": "d1"}, "emb-1")) == []
    assert fingerprint_diff(base, pipeline_fingerprint(s, {"report_a": "d1"}, "emb-2")) == ["embedding_model"]
    assert fingerprint_diff(base, pipeline_fingerprint(s, {"report_a": "d1", "report_b": "d2"}, "emb-1")) == [
        "docs"
    ]
    s.retrieval.theta = 0.7
    s.llm.answer_model = "other"
    s.prompts.answer_doc = "answer_doc_v2"
    assert fingerprint_diff(base, pipeline_fingerprint(s, {"report_a": "d1"}, "emb-1")) == [
        "answer_model",
        "prompts",
        "theta",
    ]


def test_fingerprint_holds_no_secret():
    s = load_settings()
    s.groq_api_key, s.gemini_api_key = "SECRET-1", "SECRET-2"
    assert "SECRET" not in json.dumps(pipeline_fingerprint(s, {}, "e"))


def test_row_hash_changes_when_the_row_changes():
    a = row("D1")
    assert row_hash(a) == row_hash(row("D1"))
    assert row_hash(a) != row_hash(row("D1", question="What was revenue?"))
    assert row_hash(a) != row_hash(row("D1", gold_answer="Rs 1.00 million"))


# ---------------------------------------------------------------- what to judge


def test_an_answered_answerable_document_question_is_judged_on_its_document_answer():
    [t] = judge_targets(rec(judge=None))
    assert t.part == "document" and t.reference.startswith("Rs 100.50") and t.answer.startswith("Profit")
    assert [s.label for s in t.sources] == ["p.56"] and t.sources[0].text == "source text of page 56"


@pytest.mark.parametrize(
    "r",
    [
        rec(status="abstained", abstain_reason="insufficient"),  # nothing was answered
        rec(status="error", abstain_reason="llm_unavailable"),
        rec(answerable=False, gold_answer=None, pages=()),  # unanswerable: nothing to compare with
        rec(status=None),  # misrouted away from the document path
    ],
)
def test_nothing_to_judge(r):
    assert judge_targets(r) == []


def test_general_and_mixed_answers_are_judged_against_their_own_reference():
    g = rec(
        "G1",
        route="GENERAL",
        slice="general",
        answerable=None,
        gold_answer="Paris.",
        pages=(),
        status=None,
        general_status="answered",
        general_text="Paris.",
    )
    [t] = judge_targets(g)
    assert (t.part, t.reference, t.answer, t.sources) == ("general", "Paris.", "Paris.", ())
    m = rec(
        "M1",
        route="MIXED",
        slice="mixed",
        type="other",
        general_answer="Paris.",
        general_status="answered",
        general_text="It is Paris.",
    )
    parts = {t.part: t for t in judge_targets(m)}
    assert set(parts) == {"document", "general"}
    assert parts["document"].reference.startswith("Rs 100.50") and parts["general"].reference == "Paris."


def test_judge_needed_follows_the_verdicts_and_the_judge_in_use():
    fresh = rec()
    assert judge_needed(fresh) is True
    judged = rec(judge={"document": (True, True)})
    assert judge_needed(judged) is False
    assert judge_needed(judged, rejudge=True) is True
    assert judge_needed(judged, current=("judge-m", "v1")) is False
    assert judge_needed(judged, current=("judge-m", "v2")) is True  # a new prompt voids old verdicts
    assert judge_needed(judged, current=("other-model", "v1")) is True
    assert (
        judge_needed(rec(status="abstained")) is False
        and judge_needed(rec(status="abstained"), rejudge=True) is False
    )
    half = rec(
        "M1",
        route="MIXED",
        slice="mixed",
        type="other",
        general_status="answered",
        judge={"document": (True, True)},
    )
    assert judge_needed(half) is True  # the general half has no verdict yet
    assert verdict_of(half, "document")["correct"] is True and verdict_of(half, "general") is None


# ---------------------------------------------------------------- resume


def done(**kw):
    r = rec(**kw)
    r["row_hash"] = row_hash(row("D001"))
    return r


def test_plan_runs_everything_for_a_new_question():
    assert plan(None, row(), mode="full", judge_enabled=True) == (True, True)
    assert plan(None, row(), mode="full", judge_enabled=False) == (True, False)


def test_plan_does_nothing_for_a_finished_judged_question():
    r = done(judge={"document": (True, True)})
    assert plan(r, row(), mode="full", judge_enabled=True) == (False, False)


def test_plan_judges_a_finished_question_that_has_no_verdict_without_rerunning_it():
    assert plan(done(), row(), mode="full", judge_enabled=True) == (False, True)
    assert plan(done(), row(), mode="full", judge_enabled=False) == (False, False)


def test_plan_reruns_an_edited_question():
    r = done(judge={"document": (True, True)})
    edited = row(question="A different question")
    assert plan(r, edited, mode="full", judge_enabled=True) == (True, True)


def test_plan_retries_a_question_that_hit_an_llm_outage():
    r = done(infra=True, status="error", abstain_reason="llm_unavailable")
    assert plan(r, row(), mode="full", judge_enabled=True) == (True, True)


def test_a_router_only_record_is_finished_for_a_router_run_but_not_for_a_full_run():
    r = done(mode="router", status=None)
    assert plan(r, row(), mode="router", judge_enabled=True) == (False, False)
    assert plan(r, row(), mode="full", judge_enabled=True)[0] is True
    assert plan(done(), row(), mode="router", judge_enabled=False)[0] is False  # a full record covers it


def test_plan_rejudges_only_when_asked_or_when_the_judge_changed():
    r = done(judge={"document": (True, True)})
    assert plan(r, row(), mode="full", judge_enabled=True, rejudge=True) == (False, True)
    assert plan(r, row(), mode="full", judge_enabled=True, judge_current=("judge-m", "v9")) == (False, True)


# ---------------------------------------------------------------- infra errors


def test_only_outages_and_crashes_count_as_infra_errors():
    err = {"status": "error", "abstain_reason": "llm_unavailable"}
    assert infra_error([err], None) is True
    assert infra_error([{"status": "error", "abstain_reason": "internal_error"}], None) is True
    assert infra_error([{"status": "answered", "abstain_reason": None}], "llm_unavailable") is True
    # an empty reply from a working model is the model's failure, and is scored as one
    assert infra_error([{"status": "error", "abstain_reason": "empty_reply"}], None) is False
    assert infra_error([{"status": "abstained", "abstain_reason": "insufficient"}], "bad_json") is False
    assert infra_error([], None) is False
