"""Full evaluation: every question through the same code path as POST /query, scored and logged.

    python -m eval.run --config config.yaml              # everything that is not done yet
    python -m eval.run --slice table --limit 20          # the next 20 unfinished table questions
    python -m eval.run --slice general --slice mixed
    python -m eval.run --no-llm-judge                    # number match, routing, retrieval, latency only
    python -m eval.run --router-only                     # router + baselines, no answer generation
    python -m eval.run --report-only                     # re-print the report from the run file
    python -m eval.run --no-cache                        # live LLM calls: needed for latency numbers

What it does, per question: retrieve (Recall@k / MRR as in step 04), run the real `QueryService` (router ->
document and/or general path), run the two baseline routers, and store one record in
`eval/results/runs/<run>.jsonl`. Then it asks the judge (Gemini) about the answers that need it. The report
(console, `eval/results/latest.json`, an MLflow run, optionally Markdown) is built from the run file, so it
always covers everything done under this run name so far.

Free-tier quotas are the constraint, so: the dev LLM cache is on (a question is only paid for once), calls are
paced to stay under the tokens-per-minute cap, a run can be split over days (`--slice`, `--limit`, just run
it again), a question that hit an LLM outage is retried on the next run, and a judge outage leaves the
verdicts pending instead of failing the run. Only questions whose documents are READY are scored.

Like `eval.retrieval_eval` this opens the app's own `data/` index: stop the API while it runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.answering.document import DocumentAnswerer
from app.answering.general import GeneralAnswerer
from app.answering.query import QueryService
from app.config import Settings, load_settings
from app.ingestion.embedder import Embedder, embedder_from_settings
from app.ingestion.index import VectorIndex
from app.llm.client import DiskCache, LLMClient, LLMResponse, cache_enabled
from app.llm.prompts import load_prompt
from app.observability.request_log import build_record
from app.retrieval.retriever import Retriever
from app.routing.router import Router
from app.storage.db import init_db
from app.storage.documents import DocumentStore
from eval import aggregate
from eval.answer_metrics import TokenPacer
from eval.env import load_env_file
from eval.judge import Judge, JudgeError, JudgeInput, JudgeQuotaExhausted, judge_from_settings
from eval.records import (
    RunFile,
    fingerprint_diff,
    infra_error,
    judge_needed,
    judge_targets,
    pipeline_fingerprint,
    plan,
    row_hash,
)
from eval.retrieval_eval import EvalSetupError, load_docs_map, load_rows, resolve_docs
from eval.retrieval_metrics import NEAR, STRICT, GoldPage, HitPage, first_hit_rank
from eval.router_baselines import keyword_route
from eval.schema import EvalRow, Route
from eval.tracking import file_hash, git_hash, log_mlflow_run
from eval.validate import DEFAULT_DOCS, DEFAULT_QUESTIONS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"
SLICES = ("text", "table", "scanned", "general", "mixed")
JUDGE_GIVE_UP_AFTER = 3  # consecutive judge failures before the judge is left alone for this invocation


# ---- watching the LLM calls ----------------------------------------------------------------------------
@dataclass
class Call:
    model: str
    tokens: int
    cached: bool


class RecordingLLM:
    """Wraps the real client and notes every call (model, tokens, cache hit). The runner reads the notes
    after each question: the cache flag decides which latencies mean anything, and the live token counts
    feed the pacer. Thread-safe, because a MIXED question calls the model from two threads."""

    def __init__(self, inner: Any):
        self.inner = inner
        self._calls: list[Call] = []
        self._lock = threading.Lock()

    def chat(self, messages, **kw) -> LLMResponse:
        resp = self.inner.chat(messages, **kw)
        with self._lock:
            self._calls.append(Call(kw.get("model") or resp.model, resp.usage.total_tokens, resp.cached))
        return resp

    def drain(self) -> list[Call]:
        with self._lock:
            calls, self._calls = self._calls, []
        return calls

    def __getattr__(self, name: str) -> Any:  # anything else (cfg, sdk, ...) is the real client's
        return getattr(self.inner, name)


# ---- wiring --------------------------------------------------------------------------------------------
@dataclass
class Context:
    settings: Settings
    store: DocumentStore
    retriever: Retriever
    router: Router
    service: QueryService
    llm: RecordingLLM
    embedder: Embedder
    doc_id_by_key: dict[str, str]
    judge: Judge | None
    pacer: TokenPacer
    filenames: list[str] = field(default_factory=list)
    judge_failures: int = 0
    judge_budget: int | None = None   # live judge calls this invocation may still make (None = no limit)

    @property
    def key_by_id(self) -> dict[str, str]:
        return {v: k for k, v in self.doc_id_by_key.items()}

    @property
    def ready_ids(self) -> list[str]:
        return list(self.doc_id_by_key.values())


def build_context(
    settings: Settings,
    docs_map: dict[str, str],
    *,
    use_cache: bool = True,
    judge: Judge | None = None,
    embedder: Embedder | None = None,
    llm: Any = None,
    pace: bool = True,
    sleep=time.sleep,
) -> tuple[Context, dict[str, str]]:
    """The same objects the API builds at startup (minus the ingestion worker), plus the eval-only parts.
    Returns the context and the problems (doc key -> why it cannot be scored)."""
    init_db(settings.sqlite_path)
    embedder = embedder or embedder_from_settings(settings)
    index = VectorIndex(settings.chroma_dir, embedder.model_name)
    store = DocumentStore(settings.sqlite_path)
    retriever = Retriever(index, embedder, store, settings.retrieval)
    if llm is None:
        cache = DiskCache(settings.llm_cache_dir) if use_cache and cache_enabled(True) else None
        llm = LLMClient(settings, cache=cache)
    rec_llm = RecordingLLM(llm)
    router = Router(rec_llm, settings)  # type: ignore[arg-type]
    service = QueryService(
        store,
        router,
        DocumentAnswerer(retriever, rec_llm, settings),  # type: ignore[arg-type]
        GeneralAnswerer(rec_llm, settings),  # type: ignore[arg-type]
        settings,
        request_store=None,  # eval traffic must not show up on the app's Metrics page
    )
    doc_ids, problems = resolve_docs(store, docs_map)
    ctx = Context(
        settings,
        store,
        retriever,
        router,
        service,
        rec_llm,
        embedder,
        doc_ids,
        judge,
        TokenPacer(settings.eval.tpm_budget if pace else 10**12, sleep=sleep),
        filenames=[docs_map[k] for k in doc_ids],
    )
    return ctx, problems


# ---- selecting rows ------------------------------------------------------------------------------------
def select_rows(rows: list[EvalRow], slices: list[str] | None) -> list[EvalRow]:
    if not slices:
        return list(rows)
    wanted = set(slices)
    return [r for r in rows if r.slice.value in wanted]


def skip_reason(row: EvalRow, ctx: Context) -> str | None:
    """Why a question cannot be scored right now (None = go). Only READY documents are scored."""
    if not ctx.doc_id_by_key:
        return "no READY documents"
    missing = sorted({g.doc for g in row.gold_pages if g.doc not in ctx.doc_id_by_key})
    if missing:
        return f"document(s) not READY: {', '.join(missing)}"
    if row.route is Route.DOCUMENT and row.answerable and not row.gold_pages:
        return "no gold_pages"
    return None


# ---- one question --------------------------------------------------------------------------------------
def _gold_block(row: EvalRow) -> dict:
    return {
        "route": row.route.value,
        "slice": row.slice.value,
        "answerable": row.answerable,
        "type": row.type.value,
        "answer": row.gold_answer,
        "general_answer": row.gold_general_answer,
        "pages": [{"doc": g.doc, "pdf_page": g.pdf_page, "label": str(g.page)} for g in row.gold_pages],
    }


def _retrieval_info(row: EvalRow, ctx: Context) -> dict:
    res = ctx.retriever.retrieve(row.question, ctx.settings.retrieval.fetch_k, ctx.ready_ids)
    info: dict[str, Any] = {"top_score": res.top_score}
    if row.route is Route.DOCUMENT and row.answerable and row.gold_pages:
        key_by_id = ctx.key_by_id
        hits = [HitPage(key_by_id.get(c.doc_id, c.doc_id), c.page, c.page_label) for c in res.chunks]
        golds = [GoldPage(g.doc, g.pdf_page, str(g.page)) for g in row.gold_pages]
        info["strict_rank"] = first_hit_rank(hits, golds, STRICT)
        info["pm1_rank"] = first_hit_rank(hits, golds, NEAR)
    return info


def _sections_block(response: dict, results: dict, key_by_id: dict[str, str]) -> dict[str, dict]:
    raw_doc = results["document"].answer if "document" in results else None
    out: dict[str, dict] = {}
    for s in response["sections"]:
        entry: dict[str, Any] = {
            "status": s["status"],
            "answer": s["answer"],
            "question": s["question"],
            "abstain_reason": s["abstain_reason"],
        }
        if s["kind"] == "document":
            cited_texts = getattr(raw_doc, "cited_texts", {}) or {}
            entry["number_check"] = s["number_check"]
            entry["top_score"] = s["top_score"]
            entry["citations"] = [
                {
                    "doc": key_by_id.get(c["doc_id"]),
                    "doc_id": c["doc_id"],
                    "pdf_page": c["pdf_page"],
                    "page_label": c["page_label"],
                    "chunk_id": c["chunk_id"],
                }
                for c in s["citations"]
            ]
            entry["sources"] = [
                {
                    "chunk_id": c["chunk_id"],
                    "pdf_page": c["pdf_page"],
                    "page_label": c["page_label"],
                    "text": cited_texts.get(c["chunk_id"], c["snippet"]),
                }
                for c in s["citations"]
            ]
        out[s["kind"]] = entry
    return out


def run_pipeline(ctx: Context, row: EvalRow, *, mode: str, run: str) -> dict:
    """Run one question and return its record. Never raises: a crash becomes an `infra_error` record."""
    ctx.llm.drain()
    rec: dict[str, Any] = {
        "id": row.id,
        "row_hash": row_hash(row),
        "mode": mode,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_hash(),
        "question": row.question,
        "gold": _gold_block(row),
        "infra_error": False,
    }
    try:
        rec["retrieval"] = _retrieval_info(row, ctx)
        keyword = keyword_route(row.question, ctx.filenames)
        if mode == "full":
            response, results = ctx.service.run_detailed(row.question, f"eval-{run}-{row.id}")
            rec["router"] = {
                "llm": response["route"],
                "ok": response["router_ok"],
                "fallback_reason": response["router"]["fallback_reason"],
                "keyword": keyword,
            }
            rec["sections"] = _sections_block(response, results, ctx.key_by_id)
            rec["timings"] = response["timings"]
            rec["tokens"] = response["tokens"]
            rec["cost_usd"] = build_record(
                response,
                question_len=len(row.question),
                doc=results["document"].answer if "document" in results else None,  # type: ignore[arg-type]
                gen=results["general"].answer if "general" in results else None,  # type: ignore[arg-type]
                obs=ctx.settings.observability,
                app_version="eval",
            )["cost_usd_equiv"]
            rec["infra_error"] = infra_error(rec["sections"].values(), rec["router"]["fallback_reason"])
        else:
            outcome = ctx.router.route(row.question, ctx.store.list())
            rec["router"] = {
                "llm": outcome.route,
                "ok": outcome.router_ok,
                "fallback_reason": outcome.fallback_reason,
                "keyword": keyword,
            }
            rec["timings"] = {"router_ms": round(outcome.latency_ms, 1)}
            rec["infra_error"] = outcome.fallback_reason == "llm_unavailable"
    except Exception as exc:  # noqa: BLE001  one broken question must not end a long run
        rec["infra_error"] = True
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        rec.setdefault("retrieval", {})
        rec.setdefault(
            "router", {"llm": "DOCUMENT", "ok": False, "fallback_reason": "crash", "keyword": "DOCUMENT"}
        )
    calls = ctx.llm.drain()
    rec["calls"] = {
        "live": sum(1 for c in calls if not c.cached),
        "cached": sum(1 for c in calls if c.cached),
    }
    spent: dict[str, int] = {}
    for c in calls:
        if not c.cached:
            spent[c.model] = spent.get(c.model, 0) + c.tokens
    ctx.pacer.observe(spent)
    return rec


# ---- the judge -----------------------------------------------------------------------------------------
def run_judge(ctx: Context, rec: dict, *, rejudge: bool = False) -> dict:
    """Add judge verdicts to a record for every answer in it that has none. A judge failure leaves that
    verdict out (it is retried on the next run) and never fails the run."""
    judge = ctx.judge
    assert judge is not None
    stamp = {"model": judge.model, "prompt_version": judge.prompt_version}
    old = rec.get("judge") or {}
    same_judge = (old.get("model"), old.get("prompt_version")) == (stamp["model"], stamp["prompt_version"])
    verdicts = dict(old.get("verdicts") or {}) if same_judge and not rejudge else {}
    errors: dict[str, str] = {}
    for t in judge_targets(rec):
        if t.part in verdicts:
            continue
        if ctx.judge_failures >= JUDGE_GIVE_UP_AFTER:
            errors[t.part] = "judge unavailable (skipped after repeated failures)"
            continue
        if ctx.judge_budget is not None and ctx.judge_budget <= 0:
            continue   # over this run's judge budget: the verdict stays pending, a later run can add it
        try:
            v = judge.judge(JudgeInput(t.question, t.answer, t.reference, t.sources))
        except JudgeQuotaExhausted as exc:
            ctx.judge_failures = JUDGE_GIVE_UP_AFTER   # a daily quota will not clear during this run
            errors[t.part] = str(exc)[:200]
            continue
        except JudgeError as exc:
            ctx.judge_failures += 1
            errors[t.part] = str(exc)[:200]
            continue
        ctx.judge_failures = 0
        if ctx.judge_budget is not None and not v.cached:
            ctx.judge_budget -= 1
        verdicts[t.part] = {
            "correct": v.correct,
            "grounded": v.grounded,
            "reason": v.reason,
            "tokens": v.tokens,
        }
    rec = dict(rec)
    rec["judge"] = {**stamp, "verdicts": verdicts, **({"errors": errors} if errors else {})}
    return rec


# ---- the loop ------------------------------------------------------------------------------------------
@dataclass
class RunStats:
    ran: int = 0
    judged: int = 0
    cached_rows: int = 0
    infra_errors: int = 0
    skipped: list[dict] = field(default_factory=list)
    slept: float = 0.0


def execute(
    ctx: Context,
    rows: list[EvalRow],
    run_file: RunFile,
    existing: dict[str, dict],
    *,
    mode: str,
    run: str,
    limit: int | None = None,
    judge_enabled: bool = True,
    rejudge: bool = False,
    out=print,
) -> RunStats:
    stats = RunStats()
    judge_current = (ctx.judge.model, ctx.judge.prompt_version) if ctx.judge else None
    judge_on = judge_enabled and ctx.judge is not None
    worked = 0
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        why = skip_reason(row, ctx)
        if why:
            stats.skipped.append({"id": row.id, "reason": why})
            continue
        rec = existing.get(row.id)
        run_pipe, run_jdg = plan(
            rec, row, mode=mode, judge_enabled=judge_on, rejudge=rejudge, judge_current=judge_current
        )
        if not (run_pipe or run_jdg):
            continue
        if limit is not None and worked >= limit:
            out(f"--limit {limit} reached; run again to continue")
            break
        worked += 1
        note = []
        if run_pipe:
            stats.slept += ctx.pacer.wait()
            t0 = time.perf_counter()
            rec = run_pipeline(ctx, row, mode=mode, run=run)
            run_file.append(rec)
            existing[row.id] = rec
            stats.ran += 1
            if rec["infra_error"]:
                stats.infra_errors += 1
                note.append(
                    f"LLM OUTAGE ({rec.get('error') or rec['router'].get('fallback_reason') or 'error'})"
                )
            else:
                sec = rec.get("sections") or {}
                shown = ", ".join(f"{k}:{v['status']}" for k, v in sec.items()) or "router only"
                note.append(f"{rec['router']['llm']} [{shown}] {time.perf_counter() - t0:.1f}s")
            if rec["calls"]["cached"] and not rec["calls"]["live"]:
                stats.cached_rows += 1
        re_judge = rejudge and not run_pipe  # a freshly run question has no verdicts to redo
        if judge_on and not rec["infra_error"] and judge_needed(rec, rejudge=re_judge, current=judge_current):
            rec = run_judge(ctx, rec, rejudge=re_judge)
            run_file.append(rec)
            existing[row.id] = rec
            stats.judged += 1
            v = rec["judge"]["verdicts"]
            pending = [t.part for t in judge_targets(rec) if t.part not in v]
            note.append("judged " + (",".join(f"{k}={int(x['correct'])}" for k, x in v.items()) or "-"))
            if pending:
                note.append(f"judge pending: {','.join(pending)}")
        out(f"[{i:>3}/{total}] {row.id:<5} {'; '.join(note)}")
    return stats


# ---- report assembly -----------------------------------------------------------------------------------
def meta_from_header(header: dict, *, run: str, eval_set_hash: str, judge_model: str | None) -> dict:
    fp = header["pipeline"]
    return {
        "run": run,
        "git": git_hash(),
        "run_created": header.get("created"),
        "eval_set_hash": eval_set_hash,
        "answer_model": fp["answer_model"],
        "router_model": fp["router_model"],
        "judge_model": judge_model,
        "prompts": header.get("prompt_versions") or fp["prompts"],
        "theta": fp["theta"],
        "fetch_k": fp["fetch_k"],
        "top_k": fp["top_k"],
        "chunk_size": fp["chunk_size"],
        "chunk_overlap": fp["chunk_overlap"],
        "embedding_model": fp["embedding_model"],
        "docs": fp["docs"],
    }


def current_records(rows: list[EvalRow], existing: dict[str, dict]) -> list[dict]:
    """The records that still match the question file (an edited row's old record is not reported)."""
    out = []
    for row in rows:
        rec = existing.get(row.id)
        if rec and rec.get("row_hash") == row_hash(row):
            out.append(rec)
    return out


def log_to_mlflow(report: dict, latest: Path, run_name: str) -> None:
    m = report["meta"]
    params = {
        k: m.get(k)
        for k in (
            "answer_model",
            "router_model",
            "judge_model",
            "theta",
            "fetch_k",
            "top_k",
            "chunk_size",
            "chunk_overlap",
            "embedding_model",
        )
    }
    params["prompts"] = json.dumps(m.get("prompts"), sort_keys=True)
    params["n_questions"] = m["n_records"]
    tags = {"git_hash": m["git"], "eval_set_hash": m["eval_set_hash"], "run": m["run"]}
    tags.update({f"prompt_{k}": str(v) for k, v in (m.get("prompts") or {}).items()})
    try:
        run_id = log_mlflow_run(
            experiment="full_eval",
            run_name=run_name,
            params=params,
            metrics=aggregate.flatten_metrics(report),
            tags=tags,
            artifacts=[latest],
        )
    except ImportError:
        print(
            "note: MLflow is not installed (requirements-eval.txt); the run was not logged", file=sys.stderr
        )
        return
    print(f"logged MLflow run {run_id} (view: mlflow ui --backend-store-uri ./mlruns)")


# ---- CLI -----------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full eval: router benchmark, answers, abstention, latency.")
    ap.add_argument("--config", type=Path, help="config.yaml (default: the project's)")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    ap.add_argument("--slice", action="append", help=f"only these slices ({', '.join(SLICES)}); repeatable")
    ap.add_argument("--limit", type=int, help="do at most N unfinished questions this time, then stop")
    ap.add_argument("--no-llm-judge", action="store_true", help="skip the judge (verdicts stay pending)")
    ap.add_argument("--run", default="default", help="run name = the file under eval/results/runs/")
    ap.add_argument("--fresh", action="store_true", help="delete this run's file first and start over")
    ap.add_argument(
        "--judge-limit", type=int, metavar="N",
        help="live judge calls allowed this run (0 = no limit; default: eval.judge_max_calls in config.yaml)",
    )
    ap.add_argument("--rejudge", action="store_true", help="ask the judge again for every judged answer")
    ap.add_argument("--router-only", action="store_true", help="router + baselines only (no answers)")
    ap.add_argument("--report-only", action="store_true", help="only rebuild the report from the run file")
    ap.add_argument("--no-cache", action="store_true", help="do not use the dev LLM cache (real latency)")
    ap.add_argument("--no-pace", action="store_true", help="do not wait for the tokens-per-minute budget")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "latest.json")
    ap.add_argument("--markdown", nargs="?", const=str(RESULTS_DIR / "latest.md"), help="also write tables")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)
    args.slices = [s.strip() for item in (args.slice or []) for s in item.split(",") if s.strip()]
    bad = [s for s in args.slices if s not in SLICES]
    if bad:
        ap.error(f"unknown slice {bad}; choose from {', '.join(SLICES)}")
    return args


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    load_env_file()
    settings = load_settings(args.config)
    run_file = RunFile(RUNS_DIR / f"{args.run}.jsonl")
    mode = "router" if args.router_only else "full"
    try:
        rows = load_rows(args.questions, args.docs)
        eval_hash = file_hash(args.questions)
        if args.fresh:
            run_file.delete()
        header, existing = run_file.load()

        if args.report_only:
            if header is None:
                raise EvalSetupError(f"no run file at {run_file.path}; run the eval first")
            return finish(
                rows, existing, header, args, eval_hash, skipped=[], judge_model=settings.llm.judge_model
            )

        use_judge = not args.no_llm_judge
        judge = judge_from_settings(settings, use_cache=not args.no_cache) if use_judge else None
        if use_judge and judge is None:
            print(
                "note: GEMINI_API_KEY is not set, so the judge is off; verdicts stay pending", file=sys.stderr
            )
        ctx, problems = build_context(
            settings,
            load_docs_map(args.docs),
            use_cache=not args.no_cache,
            judge=judge,
            pace=not args.no_pace,
        )
        limit_calls = settings.eval.judge_max_calls if args.judge_limit is None else args.judge_limit
        ctx.judge_budget = limit_calls if limit_calls > 0 else None
        for key, problem in problems.items():
            print(f"note: {key}: {problem}; its questions are skipped")
        if not ctx.doc_id_by_key:
            raise EvalSetupError("no READY documents match docs.yaml: upload them in the app first")

        fp = pipeline_fingerprint(settings, ctx.doc_id_by_key, ctx.embedder.model_name)
        if header is None:
            header = {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "pipeline": fp,
                "prompt_versions": {k: load_prompt(v).version for k, v in fp["prompts"].items()},
            }
            run_file.write_header(header)
        elif diff := fingerprint_diff(header["pipeline"], fp):
            raise EvalSetupError(
                f"run '{args.run}' was made with a different setup ({', '.join(diff)}). Results from "
                "different setups must not be mixed: use --fresh, or another --run name"
            )

        t0 = time.perf_counter()
        stats = execute(
            ctx,
            select_rows(rows, args.slices),
            run_file,
            existing,
            mode=mode,
            run=args.run,
            limit=args.limit,
            judge_enabled=use_judge,
            rejudge=args.rejudge,
        )
        print(
            f"done in {time.perf_counter() - t0:.0f}s: {stats.ran} question(s) run "
            f"({stats.cached_rows} fully from the cache, {stats.infra_errors} LLM outage(s)), "
            f"{stats.judged} judged, {len(stats.skipped)} skipped, {stats.slept:.0f}s spent pacing"
        )
        return finish(
            rows,
            existing,
            header,
            args,
            eval_hash,
            skipped=stats.skipped,
            judge_model=ctx.judge.model if ctx.judge else settings.llm.judge_model,
        )
    except EvalSetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def finish(
    rows: list[EvalRow],
    existing: dict[str, dict],
    header: dict,
    args: argparse.Namespace,
    eval_hash: str,
    *,
    skipped: list[dict],
    judge_model: str | None,
) -> int:
    records = current_records(rows, existing)
    if not records:
        print("error: nothing has been scored yet (no READY documents for these questions?)", file=sys.stderr)
        return 2
    meta = meta_from_header(header, run=args.run, eval_set_hash=eval_hash, judge_model=judge_model)
    meta["n_skipped"] = len(skipped)
    meta["skipped"] = skipped
    report = aggregate.build_report(records, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print()
    print(aggregate.format_report(report))
    print(f"\nwrote {args.out}")
    if args.markdown:
        Path(args.markdown).write_text(aggregate.format_markdown(report), encoding="utf-8")
        print(f"wrote {args.markdown}")
    if not args.no_mlflow:
        log_to_mlflow(report, args.out, f"full-{report['meta']['git']}-{args.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
