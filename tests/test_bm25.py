"""BM25 and rank fusion: pure functions, no index or model."""
from __future__ import annotations

import pytest

from app.retrieval.bm25 import BM25Index, rrf, tokenize


def test_tokenize_keeps_numbers_whole_and_drops_stopwords():
    assert tokenize("What was the closing balance of 1,467.25 in FY2025?") == [
        "closing", "balance", "1,467.25", "fy2025",
    ]
    assert tokenize("EPS: 6.36, ₹ (692.21)") == ["eps", "6.36", "692.21"]


def test_exact_words_and_figures_rank_first():
    idx = BM25Index(
        ["c1", "c2", "c3"],
        ["Standalone Balance Sheet Borrowings 985.71", "auditor report on borrowings", "board signatures"],
        ["a", "a", "a"],
    )
    hits = idx.search("current borrowings 985.71 balance sheet", None, 10)
    assert [cid for cid, _ in hits] == ["c1", "c2"]  # c3 shares no word, so it is not returned
    assert hits[0][1] > hits[1][1] > 0


def test_a_rare_word_outweighs_a_common_one():
    idx = BM25Index(
        ["c1", "c2", "c3"], ["company rubidium", "company report", "company annual"], ["a"] * 3
    )
    assert idx.search("company rubidium", None, 3)[0][0] == "c1"


def test_search_is_restricted_to_the_given_documents_and_limited_to_n():
    idx = BM25Index(["a1", "a2", "b1"], ["fuel cost", "fuel price", "fuel tax"], ["a", "a", "b"])
    assert {cid for cid, _ in idx.search("fuel", {"b"}, 5)} == {"b1"}
    assert len(idx.search("fuel", None, 2)) == 2
    assert idx.search("fuel", {"zzz"}, 5) == [] and idx.search("fuel", None, 0) == []


def test_empty_index_and_empty_query():
    assert BM25Index([], [], []).search("anything", None, 5) == []
    assert BM25Index(["a"], ["text"], ["d"]).search("the of", None, 5) == []


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        BM25Index(["a"], [], ["d"])


def test_rrf_fuses_weighted_rankings():
    dense = ["d1", "d2", "d3"]
    lexical = ["l1", "d3", "l2"]
    assert rrf([(1.0, dense), (1.0, lexical)])[0] == "d3"  # on both lists
    equal, heavy = rrf([(1.0, dense), (1.0, lexical)]), rrf([(1.0, dense), (5.0, lexical)])
    assert equal.index("d1") < equal.index("l1")  # tied at rank 1: the first list wins
    assert heavy.index("l1") < heavy.index("d1")  # a heavier lexical list moves its top hit ahead
    assert rrf([(1.0, dense)]) == dense
    assert rrf([]) == []


def test_rrf_ties_keep_first_seen_order():
    assert rrf([(1.0, ["x"]), (1.0, ["y"])]) == ["x", "y"]


def test_ordinals_fold_to_their_number():
    from app.retrieval.bm25 import tokenize

    assert tokenize("as on 31st March, 2025") == tokenize("as on 31 March, 2025")
    assert "1st" not in tokenize("the 1st quarter") and "1" in tokenize("the 1st quarter")
    assert tokenize("worst first") == ["worst", "first"]  # words ending in "st" are untouched
