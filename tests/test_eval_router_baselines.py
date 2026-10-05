"""The keyword-rules router baseline on hand-made questions (written without looking at the eval set)."""

from __future__ import annotations

import pytest

from eval.router_baselines import doc_terms, keyword_route

DOCS = ["Acme Annual Report FY25.pdf"]


@pytest.mark.parametrize(
    "question",
    [
        "What was Acme's consolidated revenue from operations in FY2025?",
        "What was the revenue?",  # report vocabulary, no definition phrasing
        "How many employees did the company have at year end?",
        "Per the annual report, who chairs the audit committee?",
        "What does the report say about dividends?",
        "What was profit after tax in 2024?",
        "Which page lists the borrowings?",
    ],
)
def test_questions_about_the_document_are_document(question):
    assert keyword_route(question, DOCS) == "DOCUMENT"


@pytest.mark.parametrize(
    "question",
    [
        "What is EBITDA?",  # report vocabulary, but phrased as a definition
        "Define working capital",
        "Explain how depreciation works",
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "Why is the sky blue?",
        "Tell me a joke",
    ],
)
def test_definitions_and_general_questions_are_general(question):
    assert keyword_route(question, DOCS) == "GENERAL"


@pytest.mark.parametrize(
    "question",
    [
        "What was Acme's profit before tax in FY2025, and what is the capital of France?",
        "Who wrote Pride and Prejudice, and what was Acme's revenue in FY2025?",
        "What was revenue in FY2025? Who is the Prime Minister of India?",
        "What was profit in FY2025; what is inflation?",
    ],
)
def test_a_document_part_joined_to_a_general_part_is_mixed(question):
    assert keyword_route(question, DOCS) == "MIXED"


def test_two_document_parts_are_not_mixed():
    q = "What was revenue in FY2025, and what was profit in FY2025?"
    assert keyword_route(q, DOCS) == "DOCUMENT"


def test_a_question_that_is_one_clause_is_never_mixed():
    assert keyword_route("What was Acme and Sons revenue in FY2025?", DOCS) == "DOCUMENT"


def test_the_document_name_is_a_cue():
    assert keyword_route("Tell me about Acme", DOCS) == "DOCUMENT"
    assert keyword_route("Tell me about Acme", []) == "GENERAL"


def test_a_fictional_company_with_a_period_is_a_known_failure_of_rules():
    # The rules see "FY2025" and cannot know the company is not the uploaded one. The LLM router can.
    assert keyword_route("What was Zenith Foods Limited's revenue in FY2025?", DOCS) == "DOCUMENT"


def test_doc_terms_keep_distinctive_words_only():
    assert doc_terms(["EIG AR FY25.pdf"]) == {"eig"}
    assert doc_terms(["Acme Annual Report FY25.pdf", "Q3_results-2024 final.pdf"]) == {"acme", "results"}
    assert doc_terms([]) == set()
