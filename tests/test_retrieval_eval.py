"""Retrieval eval: metrics, evaluation plumbing, CI fixture consistency, baseline gate, MLflow."""
from __future__ import annotations

import json
import re
import zlib

import pymupdf
import pytest

from app.config import load_settings
from app.retrieval.retriever import RetrievalResult, RetrievedChunk
from eval import ci_fixture
from eval.retrieval_eval import (
    EvalSetupError,
    baseline_from,
    build_fixture_index,
    build_report,
    check_baseline,
    evaluate,
    format_table,
    load_rows,
    resolve_docs,
)
from eval.retrieval_metrics import (
    NEAR,
    STRICT,
    GoldPage,
    HitPage,
    QuestionResult,
    first_hit_rank,
    mrr,
    page_matches,
    recall_at_k,
    summarize,
    summarize_by_slice,
)
from eval.schema import EvalRow
from eval.tracking import file_hash
from eval.validate import validate


# ---- metric functions on hand-made data ----------------------------------------------------------------
def test_page_matches_strict_and_plus_minus_one():
    gold = GoldPage("a", 10, "8")
    assert page_matches(HitPage("a", 10, "8"), gold, STRICT)
    assert not page_matches(HitPage("a", 11, "9"), gold, STRICT)
    assert page_matches(HitPage("a", 11, "9"), gold, NEAR)
    assert page_matches(HitPage("a", 9, "7"), gold, NEAR)
    assert not page_matches(HitPage("a", 12, "10"), gold, NEAR)
    assert not page_matches(HitPage("b", 10, "8"), gold, NEAR)   # same page number, wrong document


def test_page_matches_falls_back_to_printed_label_without_pdf_page():
    gold = GoldPage("a", None, "47")
    assert page_matches(HitPage("a", 53, "47"), gold, STRICT)
    assert not page_matches(HitPage("a", 47, "46"), gold, STRICT)       # pdf index is not the label
    assert page_matches(HitPage("a", 54, "48"), gold, NEAR)
    roman = GoldPage("a", None, "iv")
    assert page_matches(HitPage("a", 4, "iv"), roman, STRICT)
    assert not page_matches(HitPage("a", 5, "v"), roman, NEAR)          # roman labels: no arithmetic


def test_first_hit_rank_uses_the_best_ranked_gold_page():
    hits = [HitPage("a", 3, "3"), HitPage("a", 8, "8"), HitPage("a", 20, "20")]
    assert first_hit_rank(hits, [GoldPage("a", 8, "8")]) == 2
    assert first_hit_rank(hits, [GoldPage("a", 20, "20"), GoldPage("a", 8, "8")]) == 2
    assert first_hit_rank(hits, [GoldPage("a", 9, "9")], STRICT) is None
    assert first_hit_rank(hits, [GoldPage("a", 9, "9")], NEAR) == 2
    assert first_hit_rank([], [GoldPage("a", 1, "1")]) is None


def test_recall_at_k_and_mrr():
    ranks = [1, 2, 5, 6, None]
    assert recall_at_k(ranks, 1) == pytest.approx(0.2)
    assert recall_at_k(ranks, 5) == pytest.approx(0.6)
    assert recall_at_k(ranks, 8) == pytest.approx(0.8)
    assert mrr(ranks) == pytest.approx((1 + 0.5 + 0.2 + 1 / 6 + 0) / 5)
    assert recall_at_k([], 5) == 0.0 and mrr([]) == 0.0


def test_summarize_by_slice():
    results = [
        QuestionResult("1", "text", 1, 1),
        QuestionResult("2", "text", None, 2),      # only a neighbouring page was retrieved
        QuestionResult("3", "table", 6, 6),        # found, but outside the top 5
        QuestionResult("4", "table", 2, 2),
    ]
    out = summarize_by_slice(results)
    assert out["overall"]["n"] == 4
    assert out["overall"]["recall_at_5"] == 0.5
    assert out["overall"]["recall_at_5_pm1"] == 0.75
    assert out["overall"]["mrr"] == pytest.approx((1 + 0 + 1 / 6 + 0.5) / 4, abs=1e-4)
    assert out["by_slice"]["text"]["recall_at_5"] == 0.5 and out["by_slice"]["text"]["n"] == 2
    assert out["by_slice"]["table"]["recall_at_5"] == 0.5
    assert out["by_slice"]["table"]["recall_at_8"] == 1.0
    assert set(out["by_slice"]) == {"table", "text"}
    assert summarize([])["recall_at_5"] == 0.0


# ---- evaluation plumbing with a scripted retriever -----------------------------------------------------
class ScriptedRetriever:
    """Returns chunks from {question: [(doc_id, page, label, score)]}; records what it was asked."""

    def __init__(self, script):
        self.script = script
        self.calls: list[tuple] = []

    def retrieve(self, question, top_k=None, doc_ids=None):
        self.calls.append((question, top_k, list(doc_ids or [])))
        chunks = [
            RetrievedChunk(f"{d}:{p}:0", d, f"{d}.pdf", p, label, "text", "t", score, i)
            for i, (d, p, label, score) in enumerate(self.script.get(question, []), start=1)
        ]
        return RetrievalResult(chunks=chunks)


def _row(**kw) -> EvalRow:
    base = {"route": "DOCUMENT", "slice": "text", "answerable": True}
    return EvalRow(**{**base, **kw})


def test_evaluate_scores_answerable_rows_and_records_unanswerable_scores():
    rows = [
        _row(id="A", question="qa", gold_pages=[{"doc": "k1", "page": "5", "pdf_page": 7}]),
        _row(id="B", question="qb", slice="table", gold_pages=[{"doc": "k1", "page": 9, "pdf_page": 11}]),
        _row(id="U", question="qu", answerable=False),
        EvalRow(id="G", question="qg", route="GENERAL", slice="general"),
    ]
    script = {
        "qa": [("d1", 3, "1", 0.9), ("d1", 7, "5", 0.8)],               # hit at rank 2
        "qb": [("d1", 12, "10", 0.7)],                                  # neighbouring page only
        "qu": [("d1", 3, "1", 0.55)],
    }
    retriever = ScriptedRetriever(script)
    ev = evaluate(rows, retriever, {"k1": "d1"}, top_k=8)
    assert [(r.id, r.strict_rank, r.near_rank) for r in ev.results] == [("A", 2, 2), ("B", None, 1)]
    assert ev.unanswerable_scores == [0.55]
    assert ev.answerable_scores == [0.9, 0.7]
    assert ev.skipped == []
    # GENERAL rows never touch retrieval; the search is limited to the READY documents we resolved.
    assert [c[0] for c in retriever.calls] == ["qa", "qb", "qu"]
    assert all(c[1] == 8 and c[2] == ["d1"] for c in retriever.calls)


def test_evaluate_skips_questions_whose_documents_are_not_ready():
    rows = [
        _row(id="A", question="qa", gold_pages=[{"doc": "k1", "page": 1, "pdf_page": 1}]),
        _row(id="B", question="qb", gold_pages=[{"doc": "k2", "page": 1, "pdf_page": 1}]),
        _row(id="C", question="qc", gold_pages=[
            {"doc": "k1", "page": 1, "pdf_page": 1}, {"doc": "k2", "page": 2, "pdf_page": 2},
        ]),
        _row(id="D", question="qd"),    # answerable but no gold pages
    ]
    ev = evaluate(rows, ScriptedRetriever({"qa": [("d1", 1, "1", 0.9)]}), {"k1": "d1"}, top_k=5)
    assert [r.id for r in ev.results] == ["A"]
    skipped = {s["id"]: s["reason"] for s in ev.skipped}
    assert set(skipped) == {"B", "C", "D"}
    assert "k2" in skipped["B"] and "k2" in skipped["C"] and "gold_pages" in skipped["D"]


def test_resolve_docs_uses_only_ready_documents_matched_by_filename(tmp_path):
    from app.storage.db import init_db
    from app.storage.documents import DocumentStore

    init_db(tmp_path / "db.sqlite")
    store = DocumentStore(tmp_path / "db.sqlite")
    for doc_id, name, status in [
        ("1", "a.pdf", "READY"), ("2", "b.pdf", "PARTIAL"), ("3", "c.pdf", "FAILED"),
    ]:
        store.insert(doc_id, name, f"sha{doc_id}", 3)
        store.update(doc_id, status=status)
    ids, problems = resolve_docs(store, {"ka": "a.pdf", "kb": "b.pdf", "kc": "c.pdf", "kd": "d.pdf"})
    assert ids == {"ka": "1"}
    assert "PARTIAL" in problems["kb"] and "FAILED" in problems["kc"]
    assert "not been uploaded" in problems["kd"]


def test_load_rows_rejects_a_broken_questions_file(tmp_path):
    bad = tmp_path / "q.jsonl"
    bad.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(EvalSetupError):
        load_rows(bad, ci_fixture.DOCS_PATH)


# ---- the CI fixture ------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    ci_fixture.build_corpus(out)
    return out


def test_fixture_questions_validate_strictly():
    rep = validate(ci_fixture.QUESTIONS_PATH, ci_fixture.DOCS_PATH)
    assert rep.errors == [] and rep.warnings == []
    assert sum(1 for r in rep.rows if r.answerable) >= 40


def test_fixture_gold_answers_are_on_their_gold_pages(corpus):
    """The generated PDFs and ci_questions.jsonl must stay in lockstep: every gold answer is printed
    on its gold page, and the printed label matches the recorded label."""
    rows = [r for r in load_rows(ci_fixture.QUESTIONS_PATH, ci_fixture.DOCS_PATH) if r.answerable]
    docs: dict[str, pymupdf.Document] = {}
    try:
        for row in rows:
            for gp in row.gold_pages:
                doc = docs.setdefault(gp.doc, pymupdf.open(str(corpus / ci_fixture.DOCS[gp.doc])))
                page = doc[gp.pdf_page - 1]
                assert page.get_label() == str(gp.page), f"{row.id}: label mismatch"
                assert _norm(row.gold_answer) in _norm(page.get_text()), (
                    f"{row.id}: {row.gold_answer!r} not found on {gp.doc} pdf page {gp.pdf_page}"
                )
    finally:
        for d in docs.values():
            d.close()


def test_fixture_corpus_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ci_fixture.build_corpus(a)
    ci_fixture.build_corpus(b)
    for name in ci_fixture.DOCS.values():
        with pymupdf.open(str(a / name)) as da, pymupdf.open(str(b / name)) as db:
            assert [p.get_text() for p in da] == [p.get_text() for p in db]


def test_committed_baseline_matches_the_current_question_set():
    baseline = json.loads(ci_fixture.BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["eval_set_hash"] == file_hash(ci_fixture.QUESTIONS_PATH), (
        "ci_questions.jsonl changed: run `python -m eval.retrieval_eval --fixture --write-baseline`"
    )
    assert 0 < baseline["recall_at_5"] <= 1 and baseline["n"] > 0


class BagOfWordsEmbedder:
    """Offline stand-in for bge: hashed word counts. Not a good model, but a real lexical signal, so the
    fixture, ingestion, retrieval and eval can be exercised end to end without downloading anything."""

    model_name = "bow-test"
    DIM = 1024

    def _vec(self, text):
        v = [0.0] * self.DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            # crc32, not hash(): str hashes are salted per process, so which words collide in 1024 buckets
            # changed from run to run and the recall asserted below failed on unlucky runs.
            v[zlib.crc32(w.encode()) % self.DIM] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_eval_end_to_end_on_the_fixture_with_an_offline_embedder(tmp_path):
    settings = load_settings()
    emb = BagOfWordsEmbedder()
    retriever, store, docs_map = build_fixture_index(tmp_path, settings, emb)
    ids, problems = resolve_docs(store, docs_map)
    assert set(ids) == {"ci_fy25", "ci_fy26"} and problems == {}

    rows = load_rows(ci_fixture.QUESTIONS_PATH, ci_fixture.DOCS_PATH)
    ev = evaluate(rows, retriever, ids, top_k=8)
    assert ev.skipped == []
    assert len(ev.results) == sum(1 for r in rows if r.answerable)
    assert len(ev.unanswerable_scores) == sum(1 for r in rows if r.answerable is False)

    report = build_report(
        ev, settings, questions=ci_fixture.QUESTIONS_PATH, mode="fixture", top_k=8, embedder=emb
    )
    assert report["overall"]["n"] == len(ev.results)
    assert set(report["by_slice"]) == {"table", "text"}
    # A lexical embedder should find most pages; this guards the plumbing, not model quality.
    assert report["overall"]["recall_at_5_pm1"] >= 0.7, format_table(report)
    assert "R@5" in format_table(report)
    json.dumps(report)   # serialisable

    # Only READY documents are scored: a half-processed one drops out of resolve_docs.
    store.update("fx-ci_fy26", status="PARTIAL")
    ids2, problems2 = resolve_docs(store, docs_map)
    assert set(ids2) == {"ci_fy25"} and "ci_fy26" in problems2


# ---- baseline gate -------------------------------------------------------------------------------------
def _report(r5=1.0, r5pm1=1.0, mrr_=0.9, h="abc"):
    return {
        "overall": {"recall_at_5": r5, "recall_at_5_pm1": r5pm1, "mrr": mrr_, "n": 50},
        "eval_set": {"hash": h},
        "git": "x",
        "created": "now",
        "config": {"embedding_model": "m", "chunk_size_tokens": 400, "chunk_overlap_tokens": 60, "top_k": 8},
    }


def test_gate_passes_within_tolerance_and_fails_beyond_it():
    base = baseline_from(_report(r5=0.95, mrr_=0.9))
    assert check_baseline(_report(r5=0.95, mrr_=0.9), base)[0]
    assert check_baseline(_report(r5=0.93, mrr_=0.9), base)[0]            # -2 points: allowed
    assert check_baseline(_report(r5=1.0, mrr_=0.95), base)[0]            # better is fine
    ok, msgs = check_baseline(_report(r5=0.91, mrr_=0.9), base)           # -4 points
    assert not ok and any("Recall@5 dropped 4.0" in m for m in msgs)
    ok, _ = check_baseline(_report(r5=0.92, mrr_=0.9), base)              # exactly -3 points: allowed
    assert ok


def test_gate_also_fails_on_an_mrr_drop():
    base = baseline_from(_report(r5=1.0, mrr_=0.9))
    ok, msgs = check_baseline(_report(r5=1.0, mrr_=0.8), base)
    assert not ok and any("MRR dropped" in m for m in msgs)
    assert check_baseline(_report(r5=1.0, mrr_=0.86), base)[0]


def test_gate_tolerance_override_and_changed_question_set_warning():
    base = baseline_from(_report(r5=0.95))
    assert not check_baseline(_report(r5=0.92), base, tolerance=0.01)[0]
    ok, msgs = check_baseline(_report(r5=0.95, h="different"), base)
    assert ok and any("question set changed" in m for m in msgs)


# ---- MLflow (optional dependency) ----------------------------------------------------------------------
def test_mlflow_run_records_params_metrics_and_tags(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from eval.tracking import log_mlflow_run

    artifact = tmp_path / "report.json"
    artifact.write_text("{}", encoding="utf-8")
    run_id = log_mlflow_run(
        experiment="retrieval-test", run_name="t", params={"top_k": 8, "model": "m", "none": None},
        metrics={"recall_at_5": 0.9}, tags={"git_hash": "abc", "eval_set_hash": "def"},
        artifacts=[artifact], tracking_dir=tmp_path / "mlruns",
    )
    import mlflow

    run = mlflow.tracking.MlflowClient(tracking_uri=(tmp_path / "mlruns").resolve().as_uri()).get_run(run_id)
    assert run.data.params["top_k"] == "8" and run.data.params["model"] == "m"
    assert run.data.metrics["recall_at_5"] == 0.9
    assert run.data.tags["git_hash"] == "abc" and run.data.tags["eval_set_hash"] == "def"


def test_file_hash_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x\ny\n")
    b.write_bytes(b"x\r\ny\r\n")
    assert file_hash(a) == file_hash(b)
