"""The eval runner end to end: a real ingested index and the real QueryService, with a scripted LLM and judge.

Covers one pass over every route, resume (nothing is paid for twice), LLM-outage retry, a judge outage, an
edited question, `--limit`, router-only runs, the pacer being fed, and the CLI (files written, setup guard).
"""

from __future__ import annotations

import json

import pymupdf
import pytest
import yaml

from app.llm.client import LLMError, LLMResponse, Usage
from app.storage import documents as st
from eval import run as run_mod
from eval.judge import JudgeError, JudgeQuotaExhausted, Verdict
from eval.records import RunFile, row_hash
from eval.schema import EvalRow
from tests.conftest import FakeEmbedder, make_pdf
from tests.evalrecs import row
from tests.fakes import FakeLLM

BODY = "Revenue from operations was 12,563 crore in FY25. " * 3
PARIS = "Paris is the capital of France."


def route_json(route, doc_q="", gen_q=""):
    return json.dumps({"route": route, "document_question": doc_q, "general_question": gen_q})


def answer_json(text="Revenue was 12,563 crore."):
    return json.dumps({"answer": text, "citations": ["S1"], "status": "ANSWERED"})


INSUFFICIENT = json.dumps({"answer": "", "citations": [], "status": "INSUFFICIENT"})


class FakeJudge:
    model, prompt_version = "judge-m", "v1"

    def __init__(self, verdicts=(), fail=False):
        self.verdicts, self.fail, self.calls = list(verdicts), fail, []

    def judge(self, item):
        self.calls.append(item)
        if self.fail:
            raise JudgeError("quota exhausted")
        if self.verdicts:
            return self.verdicts.pop(0)
        return Verdict(True, True if item.sources else None, "ok", tokens=500)


@pytest.fixture
def corpus(settings, tmp_path):
    """One READY document (page 1 holds the revenue figure) in a throwaway index; returns docs_map."""
    from app.ingestion.index import VectorIndex
    from app.ingestion.worker import IngestionWorker, upload_path
    from app.storage.db import init_db
    from app.storage.documents import DocumentStore

    settings.retrieval.theta = -2.0  # the fake embedder's scores mean nothing
    init_db(settings.sqlite_path)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "r.pdf"
    make_pdf(pdf, [BODY, "Other page about something else entirely."])
    emb = FakeEmbedder()
    store = DocumentStore(settings.sqlite_path)
    worker = IngestionWorker(settings, store, VectorIndex(settings.chroma_dir, emb.model_name), emb)
    upload_path(settings, "doc-1").write_bytes(pdf.read_bytes())
    with pymupdf.open(str(pdf)) as d:
        store.insert("doc-1", "r.pdf", "sha-r", d.page_count)
    worker.process("doc-1")
    assert store.get("doc-1")["status"] == st.READY
    return {"report_a": "r.pdf"}


def rows() -> list[EvalRow]:
    return [
        row("D1", question="What was revenue in FY25?", gold_answer="Rs 12,563 crore", pages=(1,)),
        row(
            "A1",
            slice="text",
            type="near_miss",
            answerable=False,
            gold_answer=None,
            question="What is the FY30 revenue forecast?",
        ),
        row(
            "G1",
            route="GENERAL",
            slice="general",
            answerable=None,
            type="definition",
            question="What is the capital of France?",
            gold_answer="Paris.",
        ),
        row(
            "M1",
            route="MIXED",
            slice="mixed",
            type="other",
            question="Revenue in FY25, and capital of France?",
            gold_answer="Rs 12,563 crore",
            pages=(1,),
            gold_document_question="Revenue in FY25?",
            gold_general_question="Capital of France?",
            gold_general_answer="Paris.",
        ),
    ]


def scripted(settings, **extra):
    """Replies for `rows()` in order: D1 doc answer, A1 abstains, G1 general, M1 split and both answered."""
    base = dict(
        router=[
            route_json("DOCUMENT"),
            route_json("DOCUMENT"),
            route_json("GENERAL"),
            route_json("MIXED", "Revenue in FY25?", "Capital of France?"),
        ],
        doc=[answer_json(), INSUFFICIENT, answer_json()],
        general=[PARIS, PARIS],
    )
    base.update(extra)
    return FakeLLM(settings, **base)


def context(settings, corpus, llm, judge=None):
    ctx, problems = run_mod.build_context(
        settings, corpus, judge=judge, embedder=FakeEmbedder(), llm=llm, pace=False
    )
    assert problems == {}
    return ctx


def execute(ctx, tmp_path, rs=None, *, mode="full", limit=None, judge_enabled=True, rejudge=False, name="r"):
    rf = RunFile(tmp_path / f"{name}.jsonl")
    _, existing = rf.load()
    stats = run_mod.execute(
        ctx,
        rs if rs is not None else rows(),
        rf,
        existing,
        mode=mode,
        run=name,
        limit=limit,
        judge_enabled=judge_enabled,
        rejudge=rejudge,
        out=lambda *_: None,
    )
    return stats, rf.load()[1]


# ---------------------------------------------------------------- one pass


def test_one_pass_runs_every_route_through_the_real_pipeline_and_judges(settings, corpus, tmp_path):
    llm, judge = scripted(settings), FakeJudge()
    stats, recs = execute(context(settings, corpus, llm, judge), tmp_path)
    assert stats.ran == 4 and stats.skipped == [] and stats.infra_errors == 0
    assert set(recs) == {"D1", "A1", "G1", "M1"}

    d1 = recs["D1"]
    assert d1["mode"] == "full" and d1["infra_error"] is False
    assert d1["router"] == {"llm": "DOCUMENT", "ok": True, "fallback_reason": None, "keyword": "DOCUMENT"}
    doc = d1["sections"]["document"]
    assert doc["status"] == "answered" and doc["citations"][0]["doc"] == "report_a"
    assert doc["citations"][0]["pdf_page"] == 1 and doc["sources"][0]["text"].startswith(
        "Revenue from operations"
    )
    assert d1["retrieval"]["strict_rank"] == 1 and d1["retrieval"]["top_score"] is not None
    assert d1["tokens"]["total"]["total"] > 0 and d1["cost_usd"] > 0 and d1["timings"]["total_ms"] > 0
    assert d1["gold"]["pages"] == [{"doc": "report_a", "pdf_page": 1, "label": "1"}]

    assert recs["A1"]["sections"]["document"]["status"] == "abstained"
    assert recs["A1"]["sections"]["document"]["abstain_reason"] == "insufficient"
    assert recs["G1"]["router"]["llm"] == "GENERAL" and list(recs["G1"]["sections"]) == ["general"]
    assert recs["M1"]["router"]["llm"] == "MIXED" and set(recs["M1"]["sections"]) == {"document", "general"}

    # judged: D1 doc (with sources), G1 general, M1 doc + general. A1 was abstained: nothing to judge.
    assert stats.judged == 3 and len(judge.calls) == 4
    assert recs["D1"]["judge"]["verdicts"]["document"]["grounded"] is True
    assert recs["G1"]["judge"]["verdicts"]["general"]["grounded"] is None
    assert set(recs["M1"]["judge"]["verdicts"]) == {"document", "general"}
    assert "judge" not in recs["A1"]
    # the judge saw the reference answer and the cited text, and M1's *split* question
    d1_call = judge.calls[0]
    assert d1_call.reference == "Rs 12,563 crore" and d1_call.sources[0].text.startswith("Revenue")
    assert {c.question for c in judge.calls} >= {"What was revenue in FY25?", "Revenue in FY25?"}


def test_the_eval_does_not_write_to_the_apps_request_log(settings, corpus, tmp_path):
    ctx = context(settings, corpus, scripted(settings), FakeJudge())
    execute(ctx, tmp_path)
    import sqlite3

    conn = sqlite3.connect(settings.sqlite_path)
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    assert ctx.service.request_store is None


def test_report_from_a_pass_has_the_expected_numbers(settings, corpus, tmp_path):
    from eval import aggregate

    ctx = context(settings, corpus, scripted(settings), FakeJudge())
    _, recs = execute(ctx, tmp_path)
    rep = aggregate.build_report(list(recs.values()), {"run": "t"})
    assert rep["router"]["variants"]["llm_router"]["accuracy"]["value"] == 1.0
    assert rep["answers"]["overall"]["answer_accuracy"] == {
        "k": 1,
        "n": 1,
        "value": 1.0,
    }  # D1: 12,563 matched
    assert rep["abstention"]["overall"]["abstain_recall"]["value"] == 1.0  # A1 declined
    assert rep["general"]["answer_accuracy"]["value"] == 1.0 and rep["mixed"]["both_correct"]["value"] == 1.0
    assert rep["answers"]["overall"]["citation_strict"]["value"] == 1.0
    assert rep["retrieval"]["overall"]["recall_at_5"] == 1.0


# ---------------------------------------------------------------- resume


def test_running_again_does_nothing_and_asks_the_llm_and_judge_nothing(settings, corpus, tmp_path):
    llm, judge = scripted(settings), FakeJudge()
    ctx = context(settings, corpus, llm, judge)
    execute(ctx, tmp_path)
    calls_before, judge_before = len(llm.calls), len(judge.calls)
    stats, recs = execute(ctx, tmp_path)  # the FakeLLM queues are empty: any call would fail
    assert stats.ran == 0 and stats.judged == 0
    assert len(llm.calls) == calls_before and len(judge.calls) == judge_before
    assert len(recs) == 4


def test_a_judge_outage_leaves_verdicts_pending_and_a_later_run_fills_them_without_the_pipeline(
    settings, corpus, tmp_path
):
    llm = scripted(settings)
    ctx = context(settings, corpus, llm, FakeJudge(fail=True))
    stats, recs = execute(ctx, tmp_path)
    assert stats.ran == 4 and recs["D1"]["judge"]["verdicts"] == {}
    assert "quota exhausted" in recs["D1"]["judge"]["errors"]["document"]
    # the judge gives up for this invocation after repeated failures instead of hammering the API
    assert ctx.judge_failures >= run_mod.JUDGE_GIVE_UP_AFTER
    n_llm = len(llm.calls)

    ctx2 = context(settings, corpus, llm, FakeJudge())
    stats2, recs2 = execute(ctx2, tmp_path)
    assert stats2.ran == 0 and len(llm.calls) == n_llm  # the pipeline is not run again
    assert stats2.judged == 3 and recs2["D1"]["judge"]["verdicts"]["document"]["correct"] is True


def test_a_daily_judge_quota_stops_all_further_judging_at_once(settings, corpus, tmp_path):
    class Exhausted(FakeJudge):
        def judge(self, item):
            self.calls.append(item)
            raise JudgeQuotaExhausted("the judge's daily quota is used up (resets in about 7h)")

    judge = Exhausted()
    ctx = context(settings, corpus, scripted(settings), judge)
    stats, recs = execute(ctx, tmp_path)
    assert stats.ran == 4 and len(judge.calls) == 1            # one call, then no more hammering the API
    assert "daily quota" in recs["D1"]["judge"]["errors"]["document"]
    assert all(r.get("judge", {}).get("verdicts", {}) == {} for r in recs.values())


def test_the_judge_budget_caps_live_calls_and_a_later_run_fills_the_rest(settings, corpus, tmp_path):
    llm, judge = scripted(settings), FakeJudge()
    ctx = context(settings, corpus, llm, judge)
    ctx.judge_budget = 2
    stats, recs = execute(ctx, tmp_path)
    assert stats.ran == 4 and len(judge.calls) == 2 and ctx.judge_budget == 0
    judged = [p for r in recs.values() for p in (r.get("judge") or {}).get("verdicts", {})]
    assert len(judged) == 2
    assert not any("errors" in (r.get("judge") or {}) for r in recs.values())  # pending, not failed
    assert ctx.judge_failures == 0

    judge2 = FakeJudge()
    ctx2 = context(settings, corpus, llm, judge2)
    ctx2.judge_budget = 5
    stats2, _ = execute(ctx2, tmp_path)
    # 4 answers are judgeable in all: the 2 that already have a verdict are not asked again
    assert stats2.ran == 0 and len(judge2.calls) == 2 and ctx2.judge_budget == 3


def test_a_cached_judge_replay_does_not_use_up_the_budget(settings, corpus, tmp_path):
    replay = Verdict(True, True, "ok", tokens=0, cached=True)
    ctx = context(settings, corpus, scripted(settings), FakeJudge([replay] * 10))
    ctx.judge_budget = 1
    execute(ctx, tmp_path)
    assert ctx.judge_budget == 1


def test_no_llm_judge_runs_the_pipeline_and_leaves_verdicts_for_later(settings, corpus, tmp_path):
    judge = FakeJudge()
    stats, recs = execute(context(settings, corpus, scripted(settings), judge), tmp_path, judge_enabled=False)
    assert stats.ran == 4 and stats.judged == 0 and judge.calls == []
    assert all("judge" not in r for r in recs.values())


def test_an_llm_outage_marks_the_question_and_the_next_run_retries_only_that_one(settings, corpus, tmp_path):
    llm = scripted(settings, doc=[LLMError("down", 503), INSUFFICIENT, answer_json()])
    stats, recs = execute(context(settings, corpus, llm, FakeJudge()), tmp_path)
    assert stats.infra_errors == 1 and recs["D1"]["infra_error"] is True
    assert recs["D1"]["sections"]["document"]["abstain_reason"] == "llm_unavailable"
    assert "judge" not in recs["D1"]  # nothing worth judging

    llm.queues["router"] = [route_json("DOCUMENT")]
    llm.queues["doc"] = [answer_json()]
    stats2, recs2 = execute(context(settings, corpus, llm, FakeJudge()), tmp_path)
    assert stats2.ran == 1  # D1 only; the others were fine
    assert recs2["D1"]["infra_error"] is False and recs2["D1"]["sections"]["document"]["status"] == "answered"


def test_a_crash_in_the_document_path_is_isolated_by_the_service_and_still_retried_later(
    settings, corpus, tmp_path
):
    llm = scripted(settings, doc=[])  # the first document call hits an empty queue
    stats, recs = execute(context(settings, corpus, llm, None), tmp_path, judge_enabled=False)
    assert stats.ran == 4
    d1 = recs["D1"]
    assert d1["sections"]["document"]["abstain_reason"] == "internal_error" and d1["infra_error"] is True
    assert recs["G1"]["infra_error"] is False  # later questions still ran


def test_a_crash_outside_the_paths_is_recorded_and_does_not_end_the_run(settings, corpus, tmp_path):
    llm = scripted(
        settings,
        router=[
            RuntimeError("boom"),
            route_json("DOCUMENT"),
            route_json("GENERAL"),
            route_json("MIXED", "Revenue in FY25?", "Capital of France?"),
        ],
        doc=[INSUFFICIENT, answer_json()],
    )
    stats, recs = execute(context(settings, corpus, llm, None), tmp_path, judge_enabled=False)
    assert stats.ran == 4 and stats.infra_errors == 1
    d1 = recs["D1"]
    assert d1["infra_error"] is True and d1["error"] == "RuntimeError: boom"
    assert d1["router"]["fallback_reason"] == "crash" and "sections" not in d1
    assert all(recs[i]["infra_error"] is False for i in ("A1", "G1", "M1"))


def test_an_edited_question_is_run_again_and_an_untouched_one_is_not(settings, corpus, tmp_path):
    llm = scripted(settings)
    ctx = context(settings, corpus, llm, FakeJudge())
    execute(ctx, tmp_path)
    rs = rows()
    rs[2] = row(
        "G1",
        route="GENERAL",
        slice="general",
        answerable=None,
        type="definition",
        question="What is the capital of Germany?",
        gold_answer="Berlin.",
    )
    llm.queues["router"], llm.queues["general"] = [route_json("GENERAL")], ["Berlin."]
    stats, recs = execute(ctx, tmp_path, rs)
    assert stats.ran == 1 and recs["G1"]["question"] == "What is the capital of Germany?"
    assert recs["G1"]["row_hash"] == row_hash(rs[2])


def test_limit_does_the_next_unfinished_questions_and_a_rerun_continues(settings, corpus, tmp_path):
    llm = scripted(settings)
    ctx = context(settings, corpus, llm, FakeJudge())
    stats, recs = execute(ctx, tmp_path, limit=2)
    assert stats.ran == 2 and set(recs) == {"D1", "A1"}
    stats, recs = execute(ctx, tmp_path, limit=2)
    assert stats.ran == 2 and set(recs) == {"D1", "A1", "G1", "M1"}
    stats, _ = execute(ctx, tmp_path, limit=2)
    assert stats.ran == 0


def test_questions_whose_documents_are_not_ready_are_skipped_with_a_reason(settings, corpus, tmp_path):
    rs = rows() + [
        row("D9", pages=(1,), gold_answer="Rs 1.00 million"),
    ]
    rs[-1].gold_pages[0].doc = "report_b"
    stats, recs = execute(context(settings, corpus, scripted(settings), FakeJudge()), tmp_path, rs)
    assert stats.skipped == [{"id": "D9", "reason": "document(s) not READY: report_b"}] and "D9" not in recs


def test_an_answerable_question_without_gold_pages_is_skipped(settings, corpus, tmp_path):
    bad = row("D8", pages=(), gold_answer="Rs 1.00 million")
    stats, recs = execute(context(settings, corpus, scripted(settings), None), tmp_path, [bad])
    assert stats.skipped == [{"id": "D8", "reason": "no gold_pages"}] and recs == {}


# ---------------------------------------------------------------- router-only


def test_router_only_makes_one_router_call_per_question_and_no_answers(settings, corpus, tmp_path):
    llm = scripted(settings)
    stats, recs = execute(context(settings, corpus, llm, FakeJudge()), tmp_path, mode="router")
    assert stats.ran == 4 and stats.judged == 0
    assert len(llm.calls_for("router")) == 4 and llm.calls_for("doc") == [] and llm.calls_for("general") == []
    assert all(r["mode"] == "router" and "sections" not in r for r in recs.values())
    assert recs["M1"]["router"]["llm"] == "MIXED" and recs["M1"]["router"]["keyword"] in ("MIXED", "DOCUMENT")
    assert recs["D1"]["retrieval"]["strict_rank"] == 1  # retrieval is free, so it is always scored


def test_a_full_run_after_a_router_only_run_completes_the_records(settings, corpus, tmp_path):
    llm = scripted(settings)
    ctx = context(settings, corpus, llm, FakeJudge())
    execute(ctx, tmp_path, mode="router")
    llm.queues["router"] = [
        route_json("DOCUMENT"),
        route_json("DOCUMENT"),
        route_json("GENERAL"),
        route_json("MIXED", "Revenue in FY25?", "Capital of France?"),
    ]
    stats, recs = execute(ctx, tmp_path, mode="full")
    assert stats.ran == 4 and all(r["mode"] == "full" for r in recs.values())
    stats, _ = execute(ctx, tmp_path, mode="router")  # a full record covers a router-only request
    assert stats.ran == 0


def test_rejudge_asks_again_for_every_judged_answer(settings, corpus, tmp_path):
    judge = FakeJudge()
    ctx = context(settings, corpus, scripted(settings), judge)
    execute(ctx, tmp_path)
    n = len(judge.calls)
    judge2 = FakeJudge(verdicts=[Verdict(False, False, "now wrong", 10)])
    ctx2 = context(settings, corpus, scripted(settings), judge2)
    stats, recs = execute(ctx2, tmp_path, rejudge=True)
    assert stats.ran == 0 and stats.judged == 3 and len(judge2.calls) == n
    assert recs["D1"]["judge"]["verdicts"]["document"]["correct"] is False


def test_a_changed_judge_model_voids_the_old_verdicts(settings, corpus, tmp_path):
    ctx = context(settings, corpus, scripted(settings), FakeJudge())
    execute(ctx, tmp_path)
    newer = FakeJudge()
    newer.prompt_version = "v2"
    ctx2 = context(settings, corpus, scripted(settings), newer)
    stats, recs = execute(ctx2, tmp_path)
    assert stats.ran == 0 and stats.judged == 3 and recs["D1"]["judge"]["prompt_version"] == "v2"


# ---------------------------------------------------------------- watching the LLM


def test_the_recording_llm_notes_model_tokens_and_cache_hits_across_threads():
    class Inner:
        def chat(self, messages, **kw):
            return LLMResponse("x", Usage(10, 5, 15), model="resp-model", cached=kw.get("cached", False))

    rec = run_mod.RecordingLLM(Inner())
    rec.chat([], model="m-a")
    rec.chat([], model="m-b", cached=True)
    rec.chat([])
    calls = rec.drain()
    assert [(c.model, c.tokens, c.cached) for c in calls] == [
        ("m-a", 15, False),
        ("m-b", 15, True),
        ("resp-model", 15, False),
    ]
    assert rec.drain() == []


def test_live_token_use_feeds_the_pacer_and_cached_replays_do_not(settings, corpus, tmp_path):
    ctx = context(settings, corpus, scripted(settings), None)
    execute(ctx, tmp_path, rs=rows()[:1], judge_enabled=False)
    spent = ctx.pacer._events
    assert set(spent) == {settings.llm.router_model, settings.llm.answer_model}
    assert sum(t for _, t in spent[settings.llm.answer_model]) == 60

    class Cached:
        def __init__(self, inner):
            self.inner = inner

        def chat(self, *a, **kw):
            r = self.inner.chat(*a, **kw)
            r.cached = True
            return r

    ctx2 = context(settings, corpus, Cached(scripted(settings)), None)
    execute(ctx2, tmp_path, rs=rows()[:1], judge_enabled=False, name="cached")
    assert ctx2.pacer._events == {}


# ---------------------------------------------------------------- the CLI


@pytest.fixture
def cli(settings, corpus, tmp_path, monkeypatch):
    """Everything `main()` builds from disk, pointed at temp files and fakes."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(settings.model_dump(exclude={"groq_api_key", "gemini_api_key"})), encoding="utf-8"
    )
    qs = tmp_path / "questions.jsonl"
    qs.write_text("\n".join(r.model_dump_json(exclude_none=True) for r in rows()) + "\n", encoding="utf-8")
    docs = tmp_path / "docs.yaml"
    docs.write_text(yaml.safe_dump({"docs": {"report_a": {"filename": "r.pdf"}}}), encoding="utf-8")
    state = {"llm": scripted(settings), "judge": FakeJudge()}
    monkeypatch.setattr(run_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_mod, "load_env_file", lambda: [])
    monkeypatch.setattr(run_mod, "embedder_from_settings", lambda s: FakeEmbedder())
    monkeypatch.setattr(run_mod, "LLMClient", lambda s, cache=None: state["llm"])
    monkeypatch.setattr(run_mod, "judge_from_settings", lambda s, use_cache=True: state["judge"])

    def invoke(*extra):
        return run_mod.main(
            [
                "--config",
                str(cfg),
                "--questions",
                str(qs),
                "--docs",
                str(docs),
                "--out",
                str(tmp_path / "latest.json"),
                "--no-mlflow",
                "--no-pace",
                *extra,
            ]
        )

    invoke.state, invoke.tmp, invoke.cfg = state, tmp_path, cfg
    return invoke


def test_cli_runs_writes_the_report_and_a_second_run_is_free(cli, capsys):
    assert cli("--markdown", str(cli.tmp / "latest.md")) == 0
    out = capsys.readouterr().out
    assert "ROUTER (4 questions" in out and "LLM router fell back" in out and "wrote" in out
    report = json.loads((cli.tmp / "latest.json").read_text(encoding="utf-8"))
    assert (
        report["meta"]["n_records"] == 4 and report["answers"]["overall"]["answer_accuracy"]["value"] == 1.0
    )
    assert report["meta"]["run"] == "default" and report["meta"]["answer_model"]
    assert "| LLM router |" in (cli.tmp / "latest.md").read_text(encoding="utf-8")
    assert (cli.tmp / "runs" / "default.jsonl").exists()

    calls = len(cli.state["llm"].calls)
    assert cli() == 0 and len(cli.state["llm"].calls) == calls  # nothing re-run
    assert "0 question(s) run" in capsys.readouterr().out


def test_cli_judge_limit_caps_live_judge_calls(cli, capsys):
    assert cli("--judge-limit", "1") == 0
    assert len(cli.state["judge"].calls) == 1
    assert "judge pending" in capsys.readouterr().out


def test_cli_slice_and_limit_split_a_run_across_invocations(cli, capsys):
    assert cli("--slice", "table,text", "--limit", "1") == 0
    out = capsys.readouterr().out
    assert "--limit 1 reached" in out
    assert cli("--slice", "general", "--slice", "mixed") == 0
    report = json.loads((cli.tmp / "latest.json").read_text(encoding="utf-8"))
    assert report["meta"]["n_records"] == 3  # D1 + G1 + M1: the report spans the run


def test_cli_refuses_to_mix_results_from_different_setups(cli, capsys, settings):
    assert cli() == 0
    capsys.readouterr()
    cfg = yaml.safe_load(cli.cfg.read_text(encoding="utf-8"))
    cfg["retrieval"]["theta"] = 0.9
    cli.cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cli() == 2
    err = capsys.readouterr().err
    assert "different setup (theta)" in err and "--fresh" in err
    cli.state["llm"] = scripted(settings)  # --fresh starts over under the new setup
    assert cli("--fresh") == 0


def test_cli_report_only_needs_no_services_and_no_run_file_is_an_error(cli, capsys):
    assert cli("--report-only") == 2
    assert "no run file" in capsys.readouterr().err
    assert cli() == 0
    capsys.readouterr()
    cli.state["llm"] = None  # any use would crash
    assert cli("--report-only") == 0
    assert "ROUTER (4 questions" in capsys.readouterr().out


def test_cli_without_a_judge_key_still_runs_and_says_so(cli, capsys):
    cli.state["judge"] = None
    assert cli() == 0
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


def test_cli_rejects_an_unknown_slice():
    with pytest.raises(SystemExit):
        run_mod.parse_args(["--slice", "tables"])


def test_cli_errors_clearly_when_nothing_is_ready(cli, capsys):
    docs = cli.tmp / "docs.yaml"
    docs.write_text(yaml.safe_dump({"docs": {"report_a": {"filename": "missing.pdf"}}}), encoding="utf-8")
    assert cli() == 2
    assert "no READY documents" in capsys.readouterr().err


def test_mlflow_gets_params_tags_and_flat_metrics(cli, monkeypatch):
    seen = {}
    monkeypatch.setattr(run_mod, "log_mlflow_run", lambda **kw: seen.update(kw) or "run-1")
    cli_args = [
        "--config",
        str(cli.cfg),
        "--questions",
        str(cli.tmp / "questions.jsonl"),
        "--docs",
        str(cli.tmp / "docs.yaml"),
        "--out",
        str(cli.tmp / "latest.json"),
        "--no-pace",
    ]
    assert run_mod.main(cli_args) == 0
    assert seen["experiment"] == "full_eval"
    assert seen["params"]["answer_model"] and seen["params"]["theta"] == -2.0 and seen["params"]["top_k"] == 8
    assert (
        seen["tags"]["git_hash"] and seen["tags"]["eval_set_hash"] and seen["tags"]["prompt_router"] == "v1"
    )
    assert seen["metrics"]["answers.overall.answer_accuracy"] == 1.0
    assert seen["artifacts"] == [cli.tmp / "latest.json"]
