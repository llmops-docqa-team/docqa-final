"""Retrieval-only evaluation: Recall@k and MRR per slice. No LLM calls, so it is free and deterministic.

    python -m eval.retrieval_eval                       # your corpus: READY docs in the app's index (data/)
    python -m eval.retrieval_eval --mlflow              # ... and log the run to MLflow
    python -m eval.retrieval_eval --fixture             # CI corpus: synthetic reports built into a temp index
    python -m eval.retrieval_eval --fixture --check     # ... and fail (exit 1) if Recall@5 regressed
    python -m eval.retrieval_eval --fixture --write-baseline

Only answerable DOCUMENT questions are scored, and only against documents that are fully READY (partial
indexes would make results depend on timing). Unanswerable questions are not scored, but their top
retrieval scores are reported, because they are what the score threshold (theta) must separate.
A hit = a retrieved chunk in the right document on a gold page: strict (same page) and +-1 page.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import yaml

from app.config import Settings, load_settings
from app.ingestion.embedder import Embedder, embedder_from_settings
from app.ingestion.index import VectorIndex
from app.ingestion.worker import IngestionWorker, upload_path
from app.retrieval.retriever import Retriever
from app.storage import documents as st
from app.storage.db import init_db
from app.storage.documents import DocumentStore
from eval import ci_fixture
from eval.retrieval_metrics import (
    NEAR,
    STRICT,
    GoldPage,
    HitPage,
    QuestionResult,
    first_hit_rank,
    summarize_by_slice,
)
from eval.schema import EvalRow, Route
from eval.tracking import file_hash, git_hash, log_mlflow_run
from eval.validate import DEFAULT_DOCS, DEFAULT_QUESTIONS, validate

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TOLERANCE = 0.03   # CI fails if Recall@5 is more than 3 points below the baseline
# The CI corpus is small, so Recall@5 saturates near 100% and only gross breakage moves it. MRR (how high
# the right page ranks) is far more sensitive, so it is gated too, with a looser limit.
DEFAULT_MRR_TOLERANCE = 0.05
KS = (1, 3, 5, 8)


class EvalSetupError(RuntimeError):
    """The eval cannot run (bad questions file, documents missing or not READY, ...)."""


@dataclass
class Evaluation:
    results: list[QuestionResult]
    details: list[dict]
    skipped: list[dict]
    unanswerable_scores: list[float]
    answerable_scores: list[float]
    latencies_ms: list[float] = field(default_factory=list)


# ---- loading -------------------------------------------------------------------------------------------
def load_rows(questions: Path, docs: Path) -> list[EvalRow]:
    report = validate(questions, docs)
    if report.errors:
        raise EvalSetupError(f"{questions} has errors:\n  " + "\n  ".join(report.errors[:10]))
    return report.rows


def load_docs_map(docs: Path) -> dict[str, str]:
    data = yaml.safe_load(docs.read_text(encoding="utf-8")) or {}
    return {key: spec["filename"] for key, spec in (data.get("docs") or {}).items()}


def resolve_docs(store: DocumentStore, docs_map: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Map eval doc keys to doc ids of READY documents (matched by filename).

    Returns (doc_id_by_key, problems_by_key); a key with a problem is not evaluated.
    """
    by_name: dict[str, list[dict]] = {}
    for doc in store.list():
        by_name.setdefault(doc["filename"], []).append(doc)
    ids: dict[str, str] = {}
    problems: dict[str, str] = {}
    for key, filename in docs_map.items():
        candidates = by_name.get(filename, [])
        ready = [d for d in candidates if d["status"] == st.READY]
        if ready:
            ids[key] = ready[-1]["id"]
        elif candidates:
            problems[key] = f"{filename} is {candidates[-1]['status']}, not READY"
        else:
            problems[key] = f"{filename} has not been uploaded"
    return ids, problems


# ---- evaluation ----------------------------------------------------------------------------------------
def evaluate(
    rows: list[EvalRow], retriever: Retriever, doc_id_by_key: dict[str, str], *, top_k: int
) -> Evaluation:
    key_by_id = {v: k for k, v in doc_id_by_key.items()}
    ready_ids = list(doc_id_by_key.values())
    ev = Evaluation([], [], [], [], [])

    for row in rows:
        if row.route is not Route.DOCUMENT:
            continue
        if row.answerable:
            missing = sorted({g.doc for g in row.gold_pages if g.doc not in doc_id_by_key})
            if not row.gold_pages:
                ev.skipped.append({"id": row.id, "reason": "no gold_pages"})
                continue
            if missing:
                ev.skipped.append({"id": row.id, "reason": f"document(s) not READY: {', '.join(missing)}"})
                continue
        if not ready_ids:
            ev.skipped.append({"id": row.id, "reason": "no READY documents"})
            continue

        t0 = time.perf_counter()
        res = retriever.retrieve(row.question, top_k, ready_ids)
        ev.latencies_ms.append((time.perf_counter() - t0) * 1000)
        if res.top_score is not None:
            (ev.answerable_scores if row.answerable else ev.unanswerable_scores).append(res.top_score)
        if not row.answerable:
            continue

        hits = [HitPage(key_by_id[c.doc_id], c.page, c.page_label) for c in res.chunks]
        golds = [GoldPage(g.doc, g.pdf_page, str(g.page)) for g in row.gold_pages]
        strict, near = first_hit_rank(hits, golds, STRICT), first_hit_rank(hits, golds, NEAR)
        ev.results.append(QuestionResult(row.id, row.slice.value, strict, near))
        ev.details.append(
            {
                "id": row.id,
                "question": row.question,
                "slice": row.slice.value,
                "gold": [f"{g.doc}:p{g.pdf_page or g.page}" for g in row.gold_pages],
                "strict_rank": strict,
                "pm1_rank": near,
                "retrieved": [
                    f"{h.doc}:p{h.pdf_page}:{c.source_kind}:{c.score:.3f}"
                    for h, c in zip(hits, res.chunks, strict=True)
                ],
            }
        )
    return ev


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _score_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def build_report(
    ev: Evaluation, settings: Settings, *, questions: Path, mode: str, top_k: int, embedder: Embedder
) -> dict:
    summary = summarize_by_slice(ev.results, KS)
    return {
        "mode": mode,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_hash(),
        "eval_set": {"path": str(questions.name), "hash": file_hash(questions)},
        "config": {
            "embedding_model": embedder.model_name,
            "chunk_size_tokens": settings.chunking.size_tokens,
            "chunk_overlap_tokens": settings.chunking.overlap_tokens,
            "top_k": top_k,
        },
        **summary,
        "top_score": {
            "answerable": _score_stats(ev.answerable_scores),
            "unanswerable": _score_stats(ev.unanswerable_scores),
        },
        "latency_ms": {
            "p50": round(_percentile(ev.latencies_ms, 0.5), 1),
            "p95": round(_percentile(ev.latencies_ms, 0.95), 1),
        },
        "n_skipped": len(ev.skipped),
        "skipped": ev.skipped,
        "misses_at_5": [d for d in ev.details if d["strict_rank"] is None or d["strict_rank"] > 5],
        "questions": ev.details,
    }


def format_table(report: dict) -> str:
    cols = [f"recall_at_{k}" for k in KS] + ["recall_at_5_pm1", "mrr", "mrr_pm1"]
    heads = ["R@1", "R@3", "R@5", "R@8", "R@5±1", "MRR", "MRR±1"]
    lines = [
        f"Retrieval eval: {report['mode']} | top_k={report['config']['top_k']} | "
        f"model={report['config']['embedding_model']} | git={report['git']} | "
        f"eval-set={report['eval_set']['hash']}",
        f"{'slice':<10}{'n':>4}  " + "  ".join(f"{h:>6}" for h in heads),
    ]
    groups = [("overall", report["overall"])] + list(report["by_slice"].items())
    for name, m in groups:
        cells = [f"{m[c]:>6.3f}" if "mrr" in c else f"{m[c] * 100:>5.1f}%" for c in cols]
        lines.append(f"{name:<10}{m['n']:>4}  " + "  ".join(cells))
    ts = report["top_score"]
    if ts["answerable"]["n"] and ts["unanswerable"]["n"]:
        lines.append(
            f"top score, mean: answerable {ts['answerable']['mean']:.3f} vs "
            f"unanswerable {ts['unanswerable']['mean']:.3f} (max {ts['unanswerable']['max']:.3f})"
        )
    lat = report["latency_ms"]
    lines.append(f"retrieval latency: p50 {lat['p50']} ms, p95 {lat['p95']} ms")
    if report["n_skipped"]:
        lines.append(f"skipped {report['n_skipped']} question(s): see 'skipped' in the JSON")
    for d in report["misses_at_5"]:
        lines.append(f"  miss@5 {d['id']}: {d['question']}  gold={d['gold']}  got={d['retrieved'][:3]}")
    return "\n".join(lines)


# ---- CI baseline ---------------------------------------------------------------------------------------
def baseline_from(report: dict) -> dict:
    o = report["overall"]
    return {
        "recall_at_5": o["recall_at_5"],
        "recall_at_5_pm1": o["recall_at_5_pm1"],
        "mrr": o["mrr"],
        "n": o["n"],
        "tolerance": DEFAULT_TOLERANCE,
        "mrr_tolerance": DEFAULT_MRR_TOLERANCE,
        "embedding_model": report["config"]["embedding_model"],
        "chunk_size_tokens": report["config"]["chunk_size_tokens"],
        "chunk_overlap_tokens": report["config"]["chunk_overlap_tokens"],
        "top_k": report["config"]["top_k"],
        "eval_set_hash": report["eval_set"]["hash"],
        "created": report["created"],
    }


def check_baseline(report: dict, baseline: dict, tolerance: float | None = None) -> tuple[bool, list[str]]:
    """(ok, messages). Fails if overall Recall@5 is more than `tolerance` below the baseline, or MRR is
    more than the baseline's `mrr_tolerance` below it."""
    tol = baseline.get("tolerance", DEFAULT_TOLERANCE) if tolerance is None else tolerance
    mrr_tol = baseline.get("mrr_tolerance", DEFAULT_MRR_TOLERANCE)
    o = report["overall"]
    msgs = [
        f"Recall@5   {o['recall_at_5'] * 100:.1f}%  (baseline {baseline['recall_at_5'] * 100:.1f}%, "
        f"allowed drop {tol * 100:.1f} pts)",
        f"Recall@5±1 {o['recall_at_5_pm1'] * 100:.1f}%  (baseline {baseline['recall_at_5_pm1'] * 100:.1f}%)",
        f"MRR        {o['mrr']:.3f}  (baseline {baseline['mrr']:.3f}, allowed drop {mrr_tol:.3f})",
    ]
    if report["eval_set"]["hash"] != baseline.get("eval_set_hash"):
        msgs.append(
            "WARNING: the question set changed since the baseline was written; re-run with "
            "--write-baseline if that was intended."
        )
    drop = baseline["recall_at_5"] - o["recall_at_5"]
    mrr_drop = baseline["mrr"] - o["mrr"]
    if drop > tol + 1e-9:
        msgs.append(f"FAIL: Recall@5 dropped {drop * 100:.1f} points (limit {tol * 100:.1f}).")
    if mrr_drop > mrr_tol + 1e-9:
        msgs.append(f"FAIL: MRR dropped {mrr_drop:.3f} (limit {mrr_tol:.3f}).")
    return not any(m.startswith("FAIL") for m in msgs), msgs


# ---- corpora -------------------------------------------------------------------------------------------
def build_fixture_index(
    workdir: Path, settings: Settings, embedder: Embedder
) -> tuple[Retriever, DocumentStore, dict]:
    """Generate the synthetic corpus and ingest it through the real worker into a throwaway index."""
    settings.paths.sqlite_path = str(workdir / "db.sqlite")
    settings.paths.upload_dir = str(workdir / "uploads")
    settings.paths.chroma_dir = str(workdir / "chroma")
    init_db(settings.sqlite_path)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    store = DocumentStore(settings.sqlite_path)
    index = VectorIndex(settings.chroma_dir, embedder.model_name)
    worker = IngestionWorker(settings, store, index, embedder)
    for key, filename in ci_fixture.build_corpus(workdir / "pdfs").items():
        pdf = workdir / "pdfs" / filename
        doc_id = f"fx-{key}"
        pdf_bytes = pdf.read_bytes()
        upload_path(settings, doc_id).write_bytes(pdf_bytes)
        with pymupdf.open(str(pdf)) as d:
            pages = d.page_count
        store.insert(doc_id, filename, f"fixture-{key}", pages)
        worker.process(doc_id)
        doc = store.get(doc_id)
        if doc["status"] != st.READY:
            raise EvalSetupError(
                f"fixture document {filename} did not finish: {doc['status']} {doc['error']}"
            )
    return Retriever(index, embedder, store, settings.retrieval), store, dict(ci_fixture.DOCS)


# ---- CLI -----------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Retrieval-only eval (Recall@k, MRR).")
    ap.add_argument("--questions", type=Path)
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--top-k", type=int, help="how many chunks to fetch (default: retrieval.fetch_k)")
    ap.add_argument("--fixture", action="store_true", help="use the synthetic CI corpus in a temp index")
    ap.add_argument("--check", action="store_true", help="compare with the baseline; exit 1 on regression")
    ap.add_argument("--write-baseline", action="store_true", help="rewrite eval/baselines/ci_retrieval.json")
    ap.add_argument("--tolerance", type=float, help=f"allowed Recall@5 drop (default {DEFAULT_TOLERANCE})")
    ap.add_argument("--mlflow", action="store_true", help="log to MLflow (needs requirements-eval.txt)")
    ap.add_argument("--run-name")
    args = ap.parse_args(argv)
    if (args.check or args.write_baseline) and not args.fixture:
        ap.error("--check/--write-baseline apply to the CI fixture corpus: add --fixture")

    settings = load_settings()
    top_k = args.top_k or settings.retrieval.fetch_k
    embedder = embedder_from_settings(settings)
    mode = "fixture" if args.fixture else "corpus"
    questions = args.questions or (ci_fixture.QUESTIONS_PATH if args.fixture else DEFAULT_QUESTIONS)
    docs = args.docs or (ci_fixture.DOCS_PATH if args.fixture else DEFAULT_DOCS)

    try:
        rows = load_rows(questions, docs)
        if args.fixture:
            workdir = Path(tempfile.mkdtemp(prefix="docqa-ci-eval-"))
            print(f"building fixture index in {workdir} (embedding ~35 pages; first run downloads the model)")
            retriever, store, docs_map = build_fixture_index(workdir, settings, embedder)
        else:
            init_db(settings.sqlite_path)
            store = DocumentStore(settings.sqlite_path)
            index = VectorIndex(settings.chroma_dir, embedder.model_name)
            retriever = Retriever(index, embedder, store, settings.retrieval)
            docs_map = load_docs_map(docs)
        doc_ids, problems = resolve_docs(store, docs_map)
        for key, problem in problems.items():
            print(f"note: {key}: {problem}; its questions are skipped")
        ev = evaluate(rows, retriever, doc_ids, top_k=top_k)
        if not ev.results:
            raise EvalSetupError(
                "no answerable questions could be scored (no READY documents match the question set; "
                "placeholders and un-uploaded documents are skipped)"
            )
    except EvalSetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = build_report(ev, settings, questions=questions, mode=mode, top_k=top_k, embedder=embedder)
    out = args.out or RESULTS_DIR / ("retrieval_ci_latest.json" if args.fixture else "retrieval_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(format_table(report))
    print(f"wrote {out}")

    if args.mlflow:
        _log_mlflow(report, out, args.run_name or f"retrieval-{mode}-{report['git']}")

    if args.write_baseline:
        ci_fixture.BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        baseline_json = json.dumps(baseline_from(report), indent=2) + "\n"
        ci_fixture.BASELINE_PATH.write_text(baseline_json, encoding="utf-8")
        print(f"wrote baseline {ci_fixture.BASELINE_PATH}")
    if args.check:
        if not ci_fixture.BASELINE_PATH.exists():
            print(
                f"error: no baseline at {ci_fixture.BASELINE_PATH}; create it with --write-baseline",
                file=sys.stderr,
            )
            return 2
        baseline = json.loads(ci_fixture.BASELINE_PATH.read_text(encoding="utf-8"))
        ok, msgs = check_baseline(report, baseline, args.tolerance)
        print("\nCI gate")
        print("\n".join("  " + m for m in msgs))
        print("  PASS" if ok else "  FAILED")
        return 0 if ok else 1
    return 0


def _log_mlflow(report: dict, out: Path, run_name: str) -> None:
    metrics = {}
    for name, m in [("overall", report["overall"]), *report["by_slice"].items()]:
        for k, v in m.items():
            if k != "n":
                metrics[f"{name}.{k}"] = v
        metrics[f"{name}.n"] = m["n"]
    metrics["latency_p50_ms"] = report["latency_ms"]["p50"]
    try:
        run_id = log_mlflow_run(
            experiment="retrieval",
            run_name=run_name,
            params={**report["config"], "mode": report["mode"]},
            metrics=metrics,
            tags={"git_hash": report["git"], "eval_set_hash": report["eval_set"]["hash"]},
            artifacts=[out],
        )
    except ImportError:
        print("note: --mlflow needs mlflow (requirements-eval.txt); run not logged", file=sys.stderr)
        return
    print(f"logged MLflow run {run_id} (view: mlflow ui --backend-store-uri ./mlruns)")


if __name__ == "__main__":
    sys.exit(main())
