"""Catalog (companies / report types / periods from file names) and the query enhancer."""

from __future__ import annotations

import json

import pytest

from app.catalog import (
    ANNUAL,
    OTHER,
    PRESENTATION,
    QUARTERLY,
    TRANSCRIPT,
    build_catalog,
    catalog_response,
    find_periods,
    normalize_period,
    parse_filename,
)
from app.routing.enhancer import enhance
from tests.fakes import FakeLLM
from tests.test_query_api import add_doc_row, answer_json, route_json, running_app


@pytest.mark.parametrize(
    "name, company, report_type, period",
    [
        ("TCS Result Q1FY26.pdf", "TCS", QUARTERLY, "Q1 FY26"),
        ("EIG AR FY25.pdf", "EIG", ANNUAL, "FY25"),
        ("EIG IP Q2FY26.pdf", "EIG", PRESENTATION, "Q2 FY26"),
        ("EIG ECT Q1FY27.pdf", "EIG", TRANSCRIPT, "Q1 FY27"),
        ("EIG Result Q4FY26 - Copy.pdf", "EIG", QUARTERLY, "Q4 FY26"),
        ("Infosys_Annual_Report_FY2024-25.pdf", "Infosys", ANNUAL, "FY25"),
        ("report.pdf", "report", OTHER, None),
    ],
)
def test_parse_filename(name, company, report_type, period):
    m = parse_filename(name)
    assert (m.company, m.report_type, m.period) == (company, report_type, period)


def test_periods_are_normalised():
    assert normalize_period("q1fy2026") == "Q1 FY26"
    assert normalize_period("FY 2024-25") == "FY25"
    assert normalize_period("revenue") is None
    assert find_periods("Compare Q1FY26 with Q1 FY2025 and FY26") == ["Q1 FY26", "Q1 FY25", "FY26"]


def docs(*names, status="READY"):
    out = []
    for i, n in enumerate(names):
        m = parse_filename(n)
        out.append({"id": f"d{i}", "filename": n, "status": status, "company": m.company,
                    "report_type": m.report_type, "period": m.period})  # fmt: skip
    return out


LIB = docs(
    "TCS Result Q1FY26.pdf", "TCS IP Q1FY26.pdf", "TCS Result Q2FY26.pdf",
    "EIG AR FY25.pdf", "EIG Result Q4FY26.pdf", "EIG AR FY26.pdf",
)  # fmt: skip


def test_catalog_groups_by_company_and_lists_periods():
    cat = catalog_response(LIB)
    assert cat["n_companies"] == 2
    eig, tcs = cat["companies"]
    assert eig["name"] == "EIG" and tcs["name"] == "TCS"
    assert {"type": QUARTERLY, "periods": ["Q1 FY26", "Q2 FY26"]} in tcs["report_types"]
    assert {"type": ANNUAL, "periods": ["FY25", "FY26"]} in eig["report_types"]


def test_failed_documents_are_left_out_and_corrections_win():
    rows = docs("TCS Result Q1FY26.pdf") + docs("X Result Q1FY26.pdf", status="FAILED")
    rows[0]["company"] = "Tata Consultancy Services"
    [only] = build_catalog(rows)
    assert only.name == "Tata Consultancy Services"
    assert {"tata consultancy services", "tcs"} <= set(only.terms)  # the file-name tag still matches


def test_enhancer_scopes_to_company_and_quarter():
    e = enhance("What is the revenue of tcs in Q1FY26?", LIB)
    assert e.companies == ["TCS"] and e.periods == ["Q1 FY26"]
    assert e.doc_ids == ["d0", "d1"]  # TCS Q1 FY26 result + presentation, not Q2, not EIG
    assert "Q1 FY26" in e.search_query and "revenue from operations" in e.search_query


def test_enhancer_full_year_keeps_that_years_quarters():
    e = enhance("EIG PAT in FY26", LIB)
    assert e.doc_ids == ["d4", "d5"]  # Q4 FY26 result + FY26 annual report
    assert "profit after tax" in e.search_query and "pat -> profit after tax" in e.expansions


def test_enhancer_adds_line_item_aliases_only_when_the_phrase_is_asked():
    q = enhance("What was EIG finance cost in FY26?", LIB).search_query
    assert "finance costs; interest expense; borrowing costs" in q
    assert "interest expense" not in enhance("What was EIG revenue in FY26?", LIB).search_query
    # the abbreviation's long form triggers the alias too (capex -> capital expenditure -> additions to PPE)
    assert "additions to property, plant and equipment" in enhance("EIG capex in FY26", LIB).search_query


def test_enhancer_notes_a_missing_period_and_does_not_narrow_to_nothing():
    e = enhance("TCS revenue in Q3FY26", LIB)
    assert e.doc_ids == ["d0", "d1", "d2"]  # all of TCS
    assert e.notes and "Q3 FY26" in e.notes[0]


def test_enhancer_selected_company_wins_and_no_mention_searches_everything():
    assert enhance("What was revenue?", LIB, selected_company="EIG").doc_ids == ["d3", "d4", "d5"]
    e = enhance("What was revenue?", LIB)
    assert e.doc_ids is None and e.companies == []


def test_catalog_endpoint_patch_and_scoped_query(settings, monkeypatch):
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        a = add_doc_row(client, "TCS Result Q1FY26.pdf", "READY")
        add_doc_row(client, "EIG AR FY25.pdf", "READY")
        cat = client.get("/catalog").json()
        assert [c["name"] for c in cat["companies"]] == ["EIG", "TCS"]

        fix = {"company": "Tata Consultancy Services", "period": "q1 fy2026"}
        r = client.patch(f"/documents/{a}", json=fix)
        assert r.status_code == 200 and r.json()["company"] == "Tata Consultancy Services"
        assert r.json()["period"] == "Q1 FY26"
        assert client.patch(f"/documents/{a}", json={"period": "soon"}).status_code == 422
        assert client.patch("/documents/nope", json={"company": "X"}).status_code == 404

        body = client.post("/query", json={"question": "What was revenue?", "company": "EIG"}).json()
        assert body["enhancer"]["companies"] == ["EIG"] and body["enhancer"]["company_source"] == "selected"


def rewrite_json(q):
    return json.dumps({"question": q})


def test_several_companies_and_none_named_asks_which_one(settings, monkeypatch):
    settings.query.require_company = True
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "TCS Result Q1FY26.pdf", "READY")
        add_doc_row(client, "EIG AR FY25.pdf", "READY")
        body = client.post("/query", json={"question": "what is the revenue from operations?"}).json()
    [section] = body["sections"]
    assert section["status"] == "needs_company" and "EIG, TCS" in section["answer"]
    assert body["enhancer"]["needs_company"] is True
    assert llm.calls_for("doc") == []  # no answer call, so nothing can come from the wrong company


def test_rewrite_fixes_wording_and_pins_the_only_company(settings, monkeypatch):
    settings.query.rewrite = True
    settings.query.require_company = True
    fixed = "What was EIG's revenue from operations in FY25?"
    llm = FakeLLM(settings, router=[rewrite_json(fixed), route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "EIG AR FY25.pdf", "READY")
        body = client.post("/query", json={"question": "what is rev of operatins in fy25"}).json()
    rewrite_call = llm.calls_for("router")[0]["messages"][-1]["content"]
    assert "Company: EIG | Periods: FY25" in rewrite_call
    e = body["enhancer"]
    assert e["rewritten"] is True and e["question"] == fixed
    assert e["original"] == "what is rev of operatins in fy25"
    assert body["sections"][0]["question"] == fixed and body["tokens"]["enhancer"]["total"] > 0
    # Searched with the user's own words as well as the rewrite (a rewrite can drop the report's phrasing).
    assert e["search_query"].startswith("what is rev of operatins in fy25") and fixed in e["search_query"]


def test_a_bad_rewrite_keeps_the_users_question(settings, monkeypatch):
    settings.query.rewrite = True
    llm = FakeLLM(settings, router=["not json", route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "EIG AR FY25.pdf", "READY")
        body = client.post("/query", json={"question": "revenue in fy25"}).json()
    assert body["enhancer"]["rewritten"] is False and body["enhancer"]["rewrite_error"] == "bad_json"
    assert body["sections"][0]["question"] == "revenue in fy25"


def test_gibberish_is_caught_before_routing(settings, monkeypatch):
    settings.query.rewrite = True
    llm = FakeLLM(settings, router=[json.dumps({"question": "", "clear": False})])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "EIG AR FY25.pdf", "READY")
        body = client.post("/query", json={"question": "kjedubehjbkjnrderd"}).json()
    [section] = body["sections"]
    assert section["status"] == "unclear" and "EIG's revenue from operations in FY25" in section["answer"]
    assert body["enhancer"]["unclear"] is True
    assert len(llm.calls) == 1  # the rewrite only: no router, no answer call


def test_enhancer_off_searches_as_typed_but_keeps_the_picked_company(settings, monkeypatch):
    settings.query.rewrite = True
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "TCS Result Q1FY26.pdf", "READY")
        add_doc_row(client, "EIG AR FY25.pdf", "READY")
        asked = "wat is rev in q1fy26"
        body = client.post("/query", json={"question": asked, "company": "TCS", "enhance": False}).json()
    e = body["enhancer"]
    assert e["enabled"] is False and e["rewritten"] is False
    assert e["search_query"] == asked and e["question"] == asked  # no rewrite, no added words
    assert e["companies"] == ["TCS"] and e["periods"] == [] and e["n_documents"] == 1  # only the pick scopes
    assert body["sections"][0]["question"] == asked
    assert len(llm.calls_for("router")) == 1  # the router only: no rewrite call


def test_enhancer_off_still_requires_a_company_when_several_are_loaded(settings, monkeypatch):
    settings.query.require_company = True
    llm = FakeLLM(settings, router=[route_json("DOCUMENT")])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "TCS Result Q1FY26.pdf", "READY")
        add_doc_row(client, "EIG AR FY25.pdf", "READY")
        body = client.post("/query", json={"question": "revenue?", "enhance": False}).json()
    assert body["sections"][0]["status"] == "needs_company"


@pytest.mark.parametrize(
    ("asked", "rewrite", "kept"),
    [
        ("revenue for the year ended March 31, 2024?", "What was Fortis' revenue for FY25?", False),
        ("revenue for the year ended March 31, 2024?", "What was Fortis' revenue for FY24?", True),
        ("profit for the year 2024-25?", "What was HPCL's profit for FY25?", True),
        ("ebitda margin q2 vs q1 fy2026?", "TCS's EBITDA margin in Q2 FY26 vs Q1 FY26?", True),
        ("equity as on 31 March 2025", "What was HPCL's total equity as on 31 March 2025?", True),
        ("revenue?", "What was Fortis' revenue in FY25?", False),  # a year the user never gave
        ("hospitals above 500 beds", "How many hospitals does Fortis have above 300 beds?", False),
    ],
)
def test_a_rewrite_that_changes_a_year_or_number_is_not_used(asked, rewrite, kept):
    from app.routing.rewriter import acceptable

    assert acceptable(asked, rewrite, 500) is kept


def test_rewrite_that_changes_the_year_falls_back_to_the_users_question(settings, monkeypatch):
    settings.query.rewrite = True
    asked = "What was revenue from operations for the year ended March 31, 2024?"
    llm = FakeLLM(settings, router=[rewrite_json("What was EIG's revenue from operations for FY25?"),
                                    route_json("DOCUMENT")], doc=[answer_json()])
    with running_app(settings, monkeypatch, llm) as client:
        add_doc_row(client, "EIG AR FY25.pdf", "READY")
        body = client.post("/query", json={"question": asked}).json()
    e = body["enhancer"]
    assert e["rewritten"] is False and e["rewrite_error"] == "rejected" and e["question"] == asked