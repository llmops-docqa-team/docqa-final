"""Retrieval: status filtering, ranking, query prefix, coverage note, the debug endpoint. Offline."""
from __future__ import annotations

import math

import pytest

from app.config import RetrievalConfig, TopicBoostConfig, TopicRule
from app.ingestion.chunking import Chunk
from app.ingestion.embedder import BGE_QUERY_PREFIX, FastEmbedder
from app.ingestion.index import VectorIndex
from app.retrieval.retriever import Retriever, apply_topic_boost, topic_headings
from app.storage import documents as st
from app.storage.db import init_db
from app.storage.documents import DocumentStore
from tests.conftest import FakeEmbedder, make_pdf, upload, wait_for


class StubEmbedder:
    """Queries map to fixed 2-d vectors, so cosine similarity (and so ranking) is known exactly."""

    model_name = "stub"

    def __init__(self, query_vec):
        self.query_vec = query_vec
        self.queries: list[str] = []

    def embed_documents(self, texts):
        raise AssertionError("not used")

    def embed_query(self, text):
        self.queries.append(text)
        return self.query_vec


def _chunk(
    doc_id: str, page: int, i: int, text: str, *, label: str | None = None, kind: str = "text"
) -> Chunk:
    return Chunk(
        id=f"{doc_id}:{page}:{i}", doc_id=doc_id, filename=f"{doc_id}.pdf", page=page,
        page_label=label or str(page), chunk_idx=i, source_kind=kind, text=text,
        embed_text=f"{doc_id} — page {page}\n{text}", char_len=len(text),
    )


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    store = DocumentStore(db)
    index = VectorIndex(tmp_path / "chroma", "stub")
    return store, index


def add_doc(store, index, doc_id, status, chunks_and_vectors, *, pages_done=0, pages_total=10):
    store.insert(doc_id, f"{doc_id}.pdf", f"sha-{doc_id}", pages_total)
    store.update(doc_id, status=status, pages_done=pages_done)
    if chunks_and_vectors:
        chunks, vectors = zip(*chunks_and_vectors, strict=True)
        index.upsert(list(chunks), list(vectors))


def make_retriever(store, index, query_vec=(1.0, 0.0), fetch_k=8, **cfg):
    """Dense mode unless told otherwise, so these tests keep checking the cosine ranking itself."""
    emb = StubEmbedder(list(query_vec))
    cfg = {"mode": "dense", **cfg}
    return Retriever(index, emb, store, RetrievalConfig(fetch_k=fetch_k, **cfg)), emb


def test_ranking_is_by_cosine_similarity_best_first(env):
    store, index = env
    add_doc(store, index, "a", st.READY, [
        (_chunk("a", 1, 0, "far"), [0.0, 1.0]),            # cosine 0.0
        (_chunk("a", 2, 0, "best"), [1.0, 0.0]),           # cosine 1.0
        (_chunk("a", 3, 0, "mid"), [0.6, 0.8]),            # cosine 0.6
    ])
    retriever, _ = make_retriever(store, index)
    out = retriever.retrieve("q")
    assert [c.text for c in out.chunks] == ["best", "mid", "far"]
    assert [c.rank for c in out.chunks] == [1, 2, 3]
    assert [c.score for c in out.chunks] == pytest.approx([1.0, 0.6, 0.0], abs=1e-4)
    assert out.top_score == out.chunks[0].score


def test_chunk_carries_everything_later_steps_need(env):
    store, index = env
    chunk = _chunk("a", 7, 2, "Revenue was 5", label="iv", kind="table")
    add_doc(store, index, "a", st.READY, [(chunk, [1.0, 0.0])])
    c = make_retriever(store, index)[0].retrieve("q").chunks[0]
    assert (c.id, c.doc_id, c.filename, c.page, c.page_label, c.source_kind, c.text) == (
        "a:7:2", "a", "a.pdf", 7, "iv", "table", "Revenue was 5"
    )


def test_only_partial_and_ready_documents_are_searched(env):
    store, index = env
    statuses = [("r", st.READY), ("p", st.PARTIAL), ("q", st.QUEUED), ("g", st.PROCESSING), ("f", st.FAILED)]
    for doc_id, status in statuses:
        # Every doc holds a perfect match, so a leak shows up as an extra hit.
        add_doc(store, index, doc_id, status, [(_chunk(doc_id, 1, 0, f"text of {doc_id}"), [1.0, 0.0])])
    out = make_retriever(store, index)[0].retrieve("q")
    assert {c.doc_id for c in out.chunks} == {"r", "p"}
    assert {d.doc_id for d in out.searched} == {"r", "p"}
    assert {d.doc_id: d.status for d in out.not_ready} == {"q": st.QUEUED, "g": st.PROCESSING, "f": st.FAILED}


def test_no_queryable_documents_returns_empty_without_embedding(env):
    store, index = env
    add_doc(store, index, "f", st.FAILED, [(_chunk("f", 1, 0, "x"), [1.0, 0.0])])
    retriever, emb = make_retriever(store, index)
    out = retriever.retrieve("anything")
    assert out.chunks == [] and out.top_score is None
    assert emb.queries == []  # no point embedding the question
    assert [d.doc_id for d in out.not_ready] == ["f"]


def test_empty_index_and_store(env):
    store, index = env
    assert make_retriever(store, index)[0].retrieve("q").chunks == []


def test_doc_ids_narrows_but_cannot_resurrect_a_failed_doc(env):
    store, index = env
    add_doc(store, index, "a", st.READY, [(_chunk("a", 1, 0, "in a"), [1.0, 0.0])])
    add_doc(store, index, "b", st.READY, [(_chunk("b", 1, 0, "in b"), [1.0, 0.0])])
    add_doc(store, index, "f", st.FAILED, [(_chunk("f", 1, 0, "in f"), [1.0, 0.0])])
    retriever, _ = make_retriever(store, index)
    assert {c.doc_id for c in retriever.retrieve("q", doc_ids=["a"]).chunks} == {"a"}
    assert {c.doc_id for c in retriever.retrieve("q", doc_ids=["a", "b"]).chunks} == {"a", "b"}
    assert retriever.retrieve("q", doc_ids=["f"]).chunks == []
    assert retriever.retrieve("q", doc_ids=[]).chunks == []


def test_top_k_defaults_to_fetch_k_and_can_be_overridden(env):
    store, index = env
    add_doc(store, index, "a", st.READY, [(_chunk("a", p, 0, f"t{p}"), [1.0, 0.1 * p]) for p in range(1, 7)])
    retriever, _ = make_retriever(store, index, fetch_k=4)
    assert len(retriever.retrieve("q").chunks) == 4
    assert len(retriever.retrieve("q", top_k=2).chunks) == 2
    assert len(retriever.retrieve("q", top_k=50).chunks) == 6  # fewer chunks than asked for is fine
    with pytest.raises(ValueError):
        retriever.retrieve("q", top_k=0)


def test_partial_document_gives_a_coverage_note(env):
    store, index = env
    add_doc(
        store, index, "p", st.PARTIAL, [(_chunk("p", 1, 0, "x"), [1.0, 0.0])], pages_done=20, pages_total=312
    )
    add_doc(store, index, "r", st.READY, [(_chunk("r", 1, 0, "y"), [1.0, 0.0])], pages_done=5, pages_total=5)
    out = make_retriever(store, index)[0].retrieve("q")
    assert [d.doc_id for d in out.partial] == ["p"]
    note = out.coverage_note()
    assert "p.pdf" in note and "20 of 312 pages" in note and "still processing" in note
    assert "r.pdf" not in note


def test_no_coverage_note_when_everything_is_ready(env):
    store, index = env
    add_doc(store, index, "r", st.READY, [(_chunk("r", 1, 0, "y"), [1.0, 0.0])])
    assert make_retriever(store, index)[0].retrieve("q").coverage_note() is None


# ---- query prefix --------------------------------------------------------------------------------------
class _FakeModel:
    def __init__(self):
        self.seen: list[list[str]] = []

    def embed(self, texts, batch_size=32):
        self.seen.append(list(texts))
        for _ in texts:
            yield _Vec()

    def query_embed(self, text):  # fastembed's version adds no prefix for bge; must not be relied on
        raise AssertionError("embed_query must not go through query_embed")


class _Vec:
    def tolist(self):
        return [0.0, 1.0]


def test_embed_query_applies_the_bge_instruction_prefix_but_passages_get_none():
    emb = FastEmbedder("BAAI/bge-small-en-v1.5")
    emb._model = _FakeModel()
    emb.embed_query("What was EBITDA?")
    emb.embed_documents(["a passage"])
    assert emb._model.seen == [[BGE_QUERY_PREFIX + "What was EBITDA?"], ["a passage"]]


def test_query_prefix_is_configurable_and_can_be_disabled():
    emb = FastEmbedder("some-other-model", query_prefix="")
    emb._model = _FakeModel()
    emb.embed_query("plain")
    assert emb._model.seen == [["plain"]]


# ---- debug endpoint ------------------------------------------------------------------------------------
def test_debug_retrieve_endpoint(make_client, tmp_path):
    pdf = tmp_path / "r.pdf"
    make_pdf(pdf, 3, label="annual")
    with make_client(FakeEmbedder()) as client:
        empty = client.post("/debug/retrieve", json={"question": "revenue"})
        assert empty.status_code == 200 and empty.json()["chunks"] == []

        doc_id = upload(client, pdf, "Annual Report.pdf").json()["doc_id"]
        wait_for(client, doc_id, (st.READY,))

        body = client.post("/debug/retrieve", json={"question": "revenue growth", "top_k": 2}).json()
        assert len(body["chunks"]) == 2
        first = body["chunks"][0]
        for key in ("id", "doc_id", "filename", "page", "page_label", "source_kind", "text", "score", "rank"):
            assert key in first
        assert first["filename"] == "Annual Report.pdf" and first["rank"] == 1
        assert [c["rank"] for c in body["chunks"]] == [1, 2]
        assert math.isclose(body["top_score"], max(c["score"] for c in body["chunks"]))
        assert [d["doc_id"] for d in body["searched"]] == [doc_id]

        assert client.post("/debug/retrieve", json={"question": ""}).status_code == 422
        assert client.post("/debug/retrieve", json={"question": "x", "top_k": 0}).status_code == 422


# ---- hybrid retrieval (dense + BM25) ---------------------------------------------------------------------
def _lexical_env(store, index):
    """Dense vectors say 'a' is the best match; only the table chunk contains the words and the figure."""
    add_doc(store, index, "a", st.READY, [
        (_chunk("a", 1, 0, "signatures of the board of directors"), [1.0, 0.0]),     # cosine 1.0
        (_chunk("a", 2, 0, "auditor report on borrowings policy"), [0.9, 0.436]),    # cosine 0.9
        (_chunk("a", 3, 0, "Standalone Balance Sheet\n| Borrowings | 985.71 |", kind="table"), [0.0, 1.0]),
        (_chunk("a", 4, 0, "chairman message"), [0.8, 0.6]),
    ])


def test_hybrid_finds_the_lexical_match_that_dense_ranks_last(env):
    store, index = env
    _lexical_env(store, index)
    q = "standalone balance sheet borrowings 985.71"
    dense, _ = make_retriever(store, index, mode="dense")
    assert dense.retrieve(q).chunks[-1].id == "a:3:0"
    hybrid, _ = make_retriever(store, index, mode="hybrid")
    out = hybrid.retrieve(q)
    assert out.chunks[0].id == "a:3:0" and [c.rank for c in out.chunks] == [1, 2, 3, 4]
    assert out.chunks[0].score == pytest.approx(0.0, abs=1e-4)  # the score stays the cosine similarity
    assert out.top_score == pytest.approx(1.0, abs=1e-4)  # best cosine among the chunks, not the first


def test_bm25_only_chunk_outside_the_dense_pool_is_scored_by_cosine(env):
    store, index = env
    _lexical_env(store, index)
    out, _ = make_retriever(store, index, mode="hybrid", pool=1)
    got = out.retrieve("balance sheet 985.71").chunks
    assert [c.id for c in got][:2] == ["a:3:0", "a:1:0"]  # dense pool is just a:1:0; BM25 adds a:3:0
    assert got[0].score == pytest.approx(0.0, abs=1e-4)


def test_bm25_mode_drops_chunks_with_no_lexical_match(env):
    store, index = env
    _lexical_env(store, index)
    r, _ = make_retriever(store, index, mode="bm25")
    assert [c.id for c in r.retrieve("balance sheet").chunks] == ["a:3:0"]
    assert r.retrieve("zzz unknown").chunks == []


def test_hybrid_respects_doc_ids_and_only_searches_queryable_documents(env):
    store, index = env
    add_doc(store, index, "a", st.READY, [(_chunk("a", 1, 0, "borrowings were 10"), [1.0, 0.0])])
    add_doc(store, index, "b", st.READY, [(_chunk("b", 1, 0, "borrowings were 20"), [0.0, 1.0])])
    add_doc(store, index, "f", st.FAILED, [(_chunk("f", 1, 0, "borrowings were 30"), [0.5, 0.5])])
    r, _ = make_retriever(store, index, mode="hybrid")
    assert {c.doc_id for c in r.retrieve("borrowings").chunks} == {"a", "b"}
    assert {c.doc_id for c in r.retrieve("borrowings", doc_ids=["b"]).chunks} == {"b"}


def test_bm25_index_is_rebuilt_after_new_chunks_are_indexed(env):
    store, index = env
    add_doc(store, index, "a", st.READY, [(_chunk("a", 1, 0, "alpha"), [1.0, 0.0])])
    r, _ = make_retriever(store, index, mode="bm25")
    assert r.retrieve("omega").chunks == []
    add_doc(store, index, "b", st.READY, [(_chunk("b", 1, 0, "omega"), [0.0, 1.0])])
    assert [c.id for c in r.retrieve("omega").chunks] == ["b:1:0"]
    index.delete_doc("b")
    assert r.retrieve("omega").chunks == []


# ---- keyword search's #1 always reaches the model ------------------------------------------------------
CAPEX_TABLE = "Statement of Cash Flows\n| Purchases of property, plant and equipment | (692.21) |"


def _fusion_loses_the_table(store, index):
    """Prose that both lists find beats the table that only keyword search finds (the EIG capex case)."""
    add_doc(store, index, "a", st.READY, [
        (_chunk("a", 1, 0, "property plant and equipment depreciation policy"), [1.0, 0.0]),
        (_chunk("a", 2, 0, "property plant and equipment useful lives"), [0.95, 0.312]),
        (_chunk("a", 3, 0, CAPEX_TABLE, kind="table"), [0.0, 1.0]),
        *[(_chunk("a", 10 + i, 0, f"chairman letter board meeting {i}"), [0.1, 0.995]) for i in range(5)],
    ])


Q_CAPEX = "purchases of property plant and equipment cash flows 692.21"


def test_without_the_guarantee_fusion_pushes_the_keyword_winner_out(env):
    store, index = env
    _fusion_loses_the_table(store, index)
    r, _ = make_retriever(store, index, mode="hybrid", pool=2, top_k=1, bm25_keep_top=0)
    assert r.retrieve(Q_CAPEX).chunks[0].id != "a:3:0"


def test_keyword_winner_is_kept_inside_top_k(env):
    store, index = env
    _fusion_loses_the_table(store, index)
    r, _ = make_retriever(store, index, mode="hybrid", pool=2, top_k=1)
    got = r.retrieve(Q_CAPEX).chunks
    assert got[0].id == "a:3:0" and got[0].rank == 1
    assert len({c.id for c in got}) == len(got)  # nothing duplicated


def test_keep_lexical_top_leaves_an_order_that_already_has_it():
    from app.retrieval.retriever import _keep_lexical_top

    assert _keep_lexical_top(["x", "y", "z"], ["y"], 1, 2) == ["x", "y", "z"]
    assert _keep_lexical_top(["x", "y", "z"], ["z"], 1, 2) == ["x", "z", "y"]
    assert _keep_lexical_top(["x", "y", "z"], ["z", "q"], 2, 2) == ["z", "q", "x", "y"]
    assert _keep_lexical_top(["x", "y"], ["q"], 1, 2) == ["x", "q", "y"]  # not fused at all: still added
    assert _keep_lexical_top(["x", "y"], ["y"], 0, 1) == ["x", "y"]


# ---- small-to-big: a piece of a split table brings the rest of its table -----------------------

_P1 = "Standalone P&L\n| Particulars | FY25 |\n| --- | --- |\n| Revenue | 100 |"
_P2 = "Standalone P&L\n| Particulars | FY25 |\n| --- | --- |\n| Profit for the year | 7 |"
_OTHER = "Segment table\n| Segment | FY25 |\n| --- | --- |\n| Hospitals | 90 |"


def _tables_env(env):
    store, index = env
    add_doc(store, index, "d", st.READY, [
        (_chunk("d", 5, 0, "Some prose about the year."), [0.0, 1.0]),
        (_chunk("d", 5, 1, _P1, kind="table"), [1.0, 0.0]),
        (_chunk("d", 5, 2, _P2, kind="table"), [0.0, 1.0]),
        (_chunk("d", 5, 3, _OTHER, kind="table"), [0.0, 1.0]),
        (_chunk("d", 6, 0, _P2, kind="table"), [0.0, 1.0]),  # same header, other page: not a sibling
    ])
    return store, index


def test_table_siblings_are_the_other_pieces_of_the_same_table_only(env):
    store, index = _tables_env(env)
    r, _ = make_retriever(store, index, fetch_k=1)
    top = r.retrieve("revenue").chunks[0]
    assert top.id == "d:5:1"
    assert [c.id for c in r.table_siblings(top)] == ["d:5:2"]  # not the segment table, not page 6
    assert r.table_siblings(r._by_id(["d:5:0"], [1.0, 0.0])["d:5:0"]) == []  # prose has no siblings


def test_fit_context_inserts_siblings_after_their_piece_within_the_budget(env):
    from app.retrieval.retriever import fit_context

    store, index = _tables_env(env)
    r, _ = make_retriever(store, index, fetch_k=2)
    chunks = r.retrieve("revenue").chunks
    ids = [c.id for c in fit_context(chunks, 5000, r.table_siblings)]
    assert ids[:2] == ["d:5:1", "d:5:2"] and len(ids) == len(set(ids))
    assert [c.id for c in fit_context(chunks, 5000, None)] == [c.id for c in chunks]  # off
    one = fit_context(chunks, 1, r.table_siblings)  # over budget: the best passage is still kept
    assert [c.id for c in one] == ["d:5:1"]

# ---- topic boost ---------------------------------------------------------------------------------------
HIGHLIGHTS_RULE = TopicRule(when=["finance cost", "ebitda"], headings=["FINANCIAL HIGHLIGHTS"])


def _boost(enabled: bool = True, **kw) -> TopicBoostConfig:
    return TopicBoostConfig(enabled=enabled, bonus=0.3, topics=[HIGHLIGHTS_RULE], **kw)


def _note_beats_highlights(store, index):
    add_doc(store, index, "a", st.READY, [
        (_chunk("a", 1, 0, "ebitda ebitda ebitda ebitda note on ebitda"), [1.0, 0.0]),
        (_chunk("a", 2, 0, "FINANCIAL HIGHLIGHTS ebitda 1162.17", kind="table"), [0.8, 0.6]),
        *[(_chunk("a", 10 + i, 0, f"chairman letter board meeting {i}"), [0.0, 1.0]) for i in range(3)],
    ])


def test_topic_boost_moves_the_chunk_with_the_matching_heading_up(env):
    store, index = env
    _note_beats_highlights(store, index)
    off, _ = make_retriever(store, index, mode="hybrid", bm25_keep_top=0, topic_boost=_boost(enabled=False))
    on, _ = make_retriever(store, index, mode="hybrid", bm25_keep_top=0, topic_boost=_boost())
    assert [c.id for c in off.retrieve("what was ebitda").chunks][:2] == ["a:1:0", "a:2:0"]
    assert [c.id for c in on.retrieve("what was ebitda").chunks][:2] == ["a:2:0", "a:1:0"]
    # a question on another topic is not touched
    assert on.retrieve("board meeting letter").chunks[0].id.startswith("a:1")


def test_topic_headings_match_from_the_start_of_a_word_and_ignore_case():
    cfg = _boost()
    assert topic_headings("Why did Finance Costs fall?", cfg) == ["financial highlights"]
    assert topic_headings("what is the outfinance cost", cfg) == []  # not at a word start
    assert topic_headings("what was revenue", cfg) == []


def test_apply_topic_boost_is_stable_and_limited_to_depth():
    fused = [("x", 1.0), ("y", 0.9), ("z", 0.8)]
    texts = {"x": "a", "y": "highlights", "z": "highlights"}
    cfg = TopicBoostConfig(enabled=True, bonus=0.2, depth=2, topics=[])
    out = apply_topic_boost(fused, texts.get, ["highlights"], cfg)
    assert [c for c, _ in out] == ["y", "x", "z"]  # y gets +0.2; z is beyond depth 2
    assert apply_topic_boost(fused, texts.get, [], cfg) == fused  # no matching topic: unchanged
