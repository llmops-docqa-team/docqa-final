import json

import pytest
from pydantic import ValidationError

from eval.schema import EvalRow
from eval.validate import DEFAULT_DOCS, DEFAULT_QUESTIONS, main, validate


def doc_row(**kw):
    base = dict(
        id="X1",
        question="fake?",
        route="DOCUMENT",
        slice="text",
        answerable=True,
        gold_answer="a",
        gold_pages=[{"doc": "report_a", "page": 3, "pdf_page": 5}],
        verified_by="p",
    )
    base.update(kw)
    return base


def write(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_valid_rows_of_each_route():
    EvalRow.model_validate(doc_row())
    EvalRow.model_validate(dict(id="G", question="q", route="GENERAL", slice="general"))
    EvalRow.model_validate(
        dict(
            id="M",
            question="q",
            route="MIXED",
            slice="mixed",
            gold_document_question="a",
            gold_general_question="b",
            gold_general_answer="c",
        )
    )


@pytest.mark.parametrize(
    "row",
    [
        doc_row(answerable=None),
        doc_row(slice="general"),
        doc_row(route="GENERAL"),
        doc_row(route="BOGUS"),
        doc_row(extra_field=1),
        doc_row(question="  "),
        doc_row(gold_pages=[{"doc": "report_a", "page": 3, "pdf_page": 0}]),
        dict(id="M", question="q", route="MIXED", slice="mixed", gold_document_question="a"),
    ],
)
def test_invalid_rows(row):
    with pytest.raises(ValidationError):
        EvalRow.model_validate(row)


def test_page_label_may_be_roman_or_int():
    r = EvalRow.model_validate(doc_row(gold_pages=[{"doc": "report_a", "page": "xii"}]))
    assert r.gold_pages[0].page == "xii"


def test_the_shipped_question_set_is_valid_and_has_no_placeholders():
    rep = validate(DEFAULT_QUESTIONS, DEFAULT_DOCS)
    assert rep.ok, rep.errors
    assert len(rep.rows) >= 3
    assert not any("TODO" in w for w in rep.warnings)


def test_duplicate_ids_and_bad_json_reported(tmp_path):
    rep = validate(write(tmp_path, [doc_row(), doc_row(), "{not json"]), DEFAULT_DOCS)
    assert len(rep.rows) == 1
    assert any("duplicate id" in e for e in rep.errors)
    assert any("invalid JSON" in e for e in rep.errors)


def test_schema_error_includes_line_and_field(tmp_path):
    rep = validate(write(tmp_path, [doc_row(answerable=None)]), DEFAULT_DOCS)
    assert any(e.startswith("line 1:") for e in rep.errors)


def test_warnings(tmp_path):
    rows = [
        doc_row(id="A", gold_pages=[]),
        doc_row(id="B", gold_pages=[{"doc": "nope", "page": 1}], verified_by=None),
        doc_row(id="C", answerable=False, gold_answer=None),
    ]
    rep = validate(write(tmp_path, rows), DEFAULT_DOCS)
    assert rep.ok
    text = "\n".join(rep.warnings)
    assert "no gold_pages" in text
    assert "unknown doc key 'nope'" in text
    assert "no pdf_page" in text
    assert "not yet verified" not in text  # verification is informational, not a warning
    assert rep.unverified == 1
    assert "(C)" in text  # unanswerable row that still has gold_pages


def test_unverified_rows_are_reported_but_never_fail_strict(tmp_path, capsys):
    path = write(tmp_path, [doc_row(verified_by=None), doc_row(id="V", verified_by="p")])
    assert main([str(path), "--strict"]) == 0
    assert "info: 1 row(s) with gold pages have no verified_by" in capsys.readouterr().out


def test_mixed_rows_may_carry_gold_pages_for_their_document_part(tmp_path):
    row = {
        "id": "M1", "question": "Revenue in FY25 and the capital of France?", "route": "MIXED",
        "slice": "mixed", "gold_document_question": "Revenue in FY25?",
        "gold_general_question": "Capital of France?",
        "gold_general_answer": "Paris", "gold_answer": "1", "answerable": True,
        "gold_pages": [{"doc": "report_a", "page": 1, "pdf_page": 1}],
    }  # fmt: skip
    rep = validate(write(tmp_path, [row]), DEFAULT_DOCS)
    assert rep.ok and not rep.warnings


def test_clean_row_has_no_warnings(tmp_path):
    rep = validate(write(tmp_path, [doc_row()]), DEFAULT_DOCS)
    assert rep.ok and not rep.warnings


def test_main_exit_codes_and_summary(tmp_path, capsys):
    assert main([str(write(tmp_path, [doc_row()]))]) == 0
    out = capsys.readouterr().out
    assert "DOCUMENT=1" in out and "text=1" in out
    warn_only = write(tmp_path, [doc_row(), doc_row(id="Y", gold_pages=[])])
    assert main([str(warn_only)]) == 0
    assert main([str(warn_only), "--strict"]) == 1
    assert main([str(write(tmp_path, ["{bad"]))]) == 1


def test_trap_questions_validate():
    rep = validate(DEFAULT_QUESTIONS.parent / "questions_traps.jsonl", DEFAULT_DOCS)
    assert rep.ok and not rep.warnings, (rep.errors, rep.warnings)
    assert [r.id for r in rep.rows] == [f"T{i:03d}" for i in range(1, 11)]
