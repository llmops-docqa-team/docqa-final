"""Turn the per-question records of a run into the report: router benchmark, retrieval, answers by slice,
abstention, number check, latency, tokens. Pure functions over plain dicts (see eval/records.py), so the
numbers can be recomputed from a run file without any service, and tested on hand-made records.

Conventions
- Rows whose LLM was unavailable (`infra_error`) say nothing about the system: they are left out of every
  rate and counted in `errors`. Re-running the same run name retries them.
- A rate is {"k", "n", "value"}; `value` is None (not 0) when n is 0.
- Correctness is end to end: a question the system abstained on, misrouted or errored on is not correct.
  `accuracy_when_answered` is the precision-like view.
- Numeric questions are scored by number match; everything else by the judge. A row that needs the judge and
  has no verdict yet is counted in `unjudged` and left out of the correctness rate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from eval.answer_metrics import (
    ROUTES,
    best_threshold,
    citation_hits,
    gold_figures,
    latency_stats,
    numeric_match,
    pct,
    rate,
    route_summary,
    score_route,
)
from eval.records import verdict_of
from eval.retrieval_metrics import GoldPage, HitPage, QuestionResult, summarize_by_slice

DOC_SLICES = ("text", "table", "scanned")
STAGES = ("router_ms", "embed_ms", "retrieve_ms", "document_llm_ms", "document_ms", "general_ms", "total_ms")
SMALL_N = 10  # below this a per-slice number is an anecdote; the formatters mark it


# ---- helpers on a record -------------------------------------------------------------------------------
def _doc(rec: dict) -> dict | None:
    return (rec.get("sections") or {}).get("document")


def _gen(rec: dict) -> dict | None:
    return (rec.get("sections") or {}).get("general")


def _answered(sec: dict | None) -> bool:
    return bool(sec) and sec["status"] == "answered"


def uses_numeric_match(gold: dict) -> bool:
    """Numeric questions get the number match. A MIXED row's document half does too when its gold answer is a
    figure. A numeric row whose gold answer has no checkable figure falls back to the judge."""
    if not gold_figures(gold.get("answer")):
        return False
    return gold.get("type") == "numeric" or gold["route"] == "MIXED"


def doc_correct(rec: dict) -> bool | None:
    """Is the document answer right? None = cannot tell yet (needs the judge, which has not run)."""
    sec = _doc(rec)
    if not _answered(sec):
        return False
    gold = rec["gold"]
    if uses_numeric_match(gold):
        return bool(numeric_match(gold.get("answer"), sec["answer"]))
    v = verdict_of(rec, "document")
    return None if v is None else bool(v["correct"])


def general_correct(rec: dict) -> bool | None:
    sec = _gen(rec)
    if not _answered(sec):
        return False
    v = verdict_of(rec, "general")
    return None if v is None else bool(v["correct"])


def _gold_pages(rec: dict) -> list[GoldPage]:
    return [GoldPage(p["doc"], p.get("pdf_page"), str(p["label"])) for p in rec["gold"].get("pages") or []]


def _cited_pages(sec: dict) -> list[HitPage]:
    return [
        HitPage(c.get("doc") or c.get("doc_id", ""), c["pdf_page"], str(c["page_label"]))
        for c in sec.get("citations") or []
    ]


def _count(items: Sequence[Any], pred: Callable[[Any], bool]) -> int:
    return sum(1 for x in items if pred(x))


# ---- router --------------------------------------------------------------------------------------------
def router_block(rows: list[dict]) -> dict:
    labels = [r["gold"]["route"] for r in rows]
    scores = [r["retrieval"].get("top_score") for r in rows]
    tau, _ = best_threshold(scores, labels)
    variants = {
        "keyword_rules": route_summary(
            [(lab, r["router"]["keyword"]) for lab, r in zip(labels, rows, strict=True)]
        ),
        "score_threshold": {
            **route_summary([(lab, score_route(s, tau)) for lab, s in zip(labels, scores, strict=True)]),
            "tau": round(tau, 4),
            "note": "tau is the best cut-off found on this same set (optimistic); this baseline cannot "
            "answer MIXED",
        },
        "llm_router": {
            **route_summary([(lab, r["router"]["llm"]) for lab, r in zip(labels, rows, strict=True)]),
            "fallbacks": _count(rows, lambda r: r["router"].get("ok") is False),
        },
    }
    misrouted = [
        {"id": r["id"], "gold": r["gold"]["route"], "predicted": r["router"]["llm"]}
        for r in rows
        if r["router"]["llm"] != r["gold"]["route"]
    ]
    return {"n": len(rows), "variants": variants, "llm_misrouted": misrouted}


# ---- retrieval -----------------------------------------------------------------------------------------
def retrieval_block(rows: list[dict]) -> dict:
    results = [
        QuestionResult(r["id"], r["gold"]["slice"], r["retrieval"]["strict_rank"], r["retrieval"]["pm1_rank"])
        for r in rows
        if r["gold"]["route"] == "DOCUMENT" and r["gold"]["answerable"] and "strict_rank" in r["retrieval"]
    ]
    return summarize_by_slice(results) if results else {"overall": {"n": 0}, "by_slice": {}}


# ---- answers -------------------------------------------------------------------------------------------
def answer_block(rows: list[dict]) -> dict:
    """Metrics for answerable DOCUMENT questions (one slice, or all of them)."""
    n = len(rows)
    answered = [r for r in rows if _answered(_doc(r))]
    verdicts = {r["id"]: doc_correct(r) for r in rows}
    decided = [r for r in rows if verdicts[r["id"]] is not None]
    answered_decided = [r for r in answered if verdicts[r["id"]] is not None]
    grounded = [
        v["grounded"] for r in answered if (v := verdict_of(r, "document")) and v.get("grounded") is not None
    ]
    cites = [citation_hits(_cited_pages(_doc(r)), _gold_pages(r)) for r in answered]
    checks = [_doc(r)["number_check"] for r in answered]
    return {
        "n": n,
        "answered": rate(len(answered), n),
        "answer_accuracy": rate(_count(decided, lambda r: verdicts[r["id"]]), len(decided)),
        "accuracy_when_answered": rate(
            _count(answered_decided, lambda r: verdicts[r["id"]]), len(answered_decided)
        ),
        "groundedness": rate(sum(1 for g in grounded if g), len(grounded)),
        "citation_strict": rate(_count(cites, lambda c: c["any_strict"]), len(cites)),
        "citation_pm1": rate(_count(cites, lambda c: c["any_pm1"]), len(cites)),
        "citation_all_strict": rate(_count(cites, lambda c: c["all_strict"]), len(cites)),
        "number_check_fail": rate(
            _count(checks, lambda c: c == "fail"), _count(checks, lambda c: c in ("pass", "fail"))
        ),
        "unjudged": n - len(decided),
    }


def answers_by_slice(rows: list[dict]) -> dict:
    """`rows`: full-mode records of answerable DOCUMENT questions."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["gold"]["slice"]].append(r)
    return {
        "overall": answer_block(rows),
        "by_slice": {s: answer_block(by[s]) for s in DOC_SLICES if by.get(s)},
    }


# ---- abstention ----------------------------------------------------------------------------------------
def _outcome(rec: dict) -> str:
    """What the document path did: ANSWER, ABSTAIN, or OTHER (misrouted away from it, not ready, error)."""
    sec = _doc(rec)
    if not sec:
        return "OTHER"
    return {"answered": "ANSWER", "abstained": "ABSTAIN"}.get(sec["status"], "OTHER")


def abstention_block(rows: list[dict]) -> dict:
    """`rows`: full-mode DOCUMENT records, answerable and not. Positive class = ABSTAIN.

    A question the router sent away from the document path counts as OTHER, so it lowers recall and the
    false-answer rate alike; `other_*` says how many there were."""
    un = [r for r in rows if not r["gold"]["answerable"]]
    an = [r for r in rows if r["gold"]["answerable"]]
    out_un = [_outcome(r) for r in un]
    out_an = [_outcome(r) for r in an]
    predicted_abstain = out_un.count("ABSTAIN") + out_an.count("ABSTAIN")
    return {
        "n_unanswerable": len(un),
        "n_answerable": len(an),
        "abstain_recall": rate(out_un.count("ABSTAIN"), len(un)),
        "false_answer_rate": rate(out_un.count("ANSWER"), len(un)),
        "abstain_precision": rate(out_un.count("ABSTAIN"), predicted_abstain),
        "wrong_abstention_rate": rate(out_an.count("ABSTAIN"), len(an)),
        "other_unanswerable": out_un.count("OTHER"),
        "other_answerable": out_an.count("OTHER"),
        "abstain_reasons": dict(
            sorted(_tally(_doc(r)["abstain_reason"] for r in rows if _outcome(r) == "ABSTAIN").items())
        ),
    }


def _tally(items) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for i in items:
        out[str(i)] += 1
    return out


def abstention_by_slice(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["gold"]["slice"]].append(r)
    return {
        "overall": abstention_block(rows),
        "by_slice": {s: abstention_block(by[s]) for s in DOC_SLICES if by.get(s)},
    }


# ---- general and mixed ---------------------------------------------------------------------------------
def general_block(rows: list[dict]) -> dict:
    ok = {r["id"]: general_correct(r) for r in rows}
    decided = [r for r in rows if ok[r["id"]] is not None]
    return {
        "n": len(rows),
        "routed_general": rate(_count(rows, lambda r: r["router"]["llm"] == "GENERAL"), len(rows)),
        "answered": rate(_count(rows, lambda r: _answered(_gen(r))), len(rows)),
        "answer_accuracy": rate(_count(decided, lambda r: ok[r["id"]]), len(decided)),
        "unjudged": len(rows) - len(decided),
    }


def mixed_block(rows: list[dict]) -> dict:
    d = {r["id"]: doc_correct(r) for r in rows}
    g = {r["id"]: general_correct(r) for r in rows}
    decided = [r for r in rows if d[r["id"]] is not None and g[r["id"]] is not None]
    return {
        "n": len(rows),
        "routed_mixed": rate(_count(rows, lambda r: r["router"]["llm"] == "MIXED"), len(rows)),
        "document_part_correct": rate(
            _count([r for r in rows if d[r["id"]] is not None], lambda r: d[r["id"]]),
            _count(rows, lambda r: d[r["id"]] is not None),
        ),
        "general_part_correct": rate(
            _count([r for r in rows if g[r["id"]] is not None], lambda r: g[r["id"]]),
            _count(rows, lambda r: g[r["id"]] is not None),
        ),
        "both_correct": rate(_count(decided, lambda r: d[r["id"]] and g[r["id"]]), len(decided)),
        "unjudged": len(rows) - len(decided),
    }


# ---- latency, tokens, judge ----------------------------------------------------------------------------
def latency_block(rows: list[dict]) -> dict:
    """Per-stage percentiles over questions where no LLM call was a cache replay (a replay reports 0 ms).
    A stage that did not run is 0 and is left out."""
    live = [r for r in rows if r.get("timings") and (r.get("calls") or {}).get("cached", 0) == 0]
    out: dict[str, Any] = {"n_live": len(live), "n_cached_excluded": len(rows) - len(live), "stages": {}}
    for stage in STAGES:
        vals = [r["timings"][stage] for r in live if r["timings"].get(stage)]
        out["stages"][stage] = latency_stats(vals)
    return out


def tokens_block(rows: list[dict]) -> dict:
    full = [r for r in rows if r.get("tokens")]
    parts = ("router", "document", "general", "total")
    sums = {
        p: {k: sum(r["tokens"][p][k] for r in full) for k in ("prompt", "completion", "total")} for p in parts
    }
    cost = round(sum(r.get("cost_usd") or 0.0 for r in full), 6)
    n = len(full)
    return {
        "n": n,
        "sum": sums,
        "mean_total_per_question": round(sums["total"]["total"] / n, 1) if n else None,
        "cost_usd_equiv": cost,
        "cost_usd_equiv_per_question": round(cost / n, 6) if n else None,
    }


def judge_block(rows: list[dict]) -> dict:
    verdicts = [v for r in rows if (j := r.get("judge")) for v in (j.get("verdicts") or {}).values()]
    # Free calibration check: on numeric questions the judge ran as well, so compare it with the number match.
    pairs = [
        (bool(numeric_match(r["gold"].get("answer"), _doc(r)["answer"])), bool(v["correct"]))
        for r in rows
        if _answered(_doc(r)) and uses_numeric_match(r["gold"]) and (v := verdict_of(r, "document"))
    ]
    meta = next((r["judge"] for r in rows if r.get("judge")), None) or {}
    return {
        "model": meta.get("model"),
        "prompt_version": meta.get("prompt_version"),
        "n_verdicts": len(verdicts),
        "tokens": sum(v.get("tokens", 0) for v in verdicts),
        "vs_numeric_match": {
            "n": len(pairs),
            "agree": rate(_count(pairs, lambda p: p[0] == p[1]), len(pairs)),
        },
    }


def failures(rows: list[dict], limit: int = 15) -> list[dict]:
    """Answerable DOCUMENT questions the system did not get right, for a quick look."""
    out = []
    for r in rows:
        if r["gold"]["route"] != "DOCUMENT" or not r["gold"]["answerable"]:
            continue
        ok = doc_correct(r)
        if ok is not False:
            continue
        sec = _doc(r)
        out.append(
            {
                "id": r["id"],
                "question": r["question"],
                "outcome": "misrouted" if not sec else sec["status"],
                "reason": (sec or {}).get("abstain_reason") or f"routed {r['router']['llm']}",
                "gold": r["gold"].get("answer"),
                "answer": (sec or {}).get("answer") if _answered(sec) else None,
            }
        )
    return out[:limit]


# ---- the report ----------------------------------------------------------------------------------------
def build_report(records: list[dict], meta: dict[str, Any]) -> dict:
    usable = [r for r in records if not r.get("infra_error")]
    errors = [r["id"] for r in records if r.get("infra_error")]
    full = [r for r in usable if r.get("mode") == "full"]
    doc_rows = [r for r in full if r["gold"]["route"] == "DOCUMENT"]
    answerable = [r for r in doc_rows if r["gold"]["answerable"]]
    return {
        "meta": {**meta, "n_records": len(records), "n_full": len(full), "n_errors": len(errors)},
        "errors": errors,
        "router": router_block(usable) if usable else {"n": 0, "variants": {}, "llm_misrouted": []},
        "retrieval": retrieval_block(usable),
        "answers": answers_by_slice(answerable),
        "abstention": abstention_by_slice(doc_rows),
        "general": general_block([r for r in full if r["gold"]["route"] == "GENERAL"]),
        "mixed": mixed_block([r for r in full if r["gold"]["route"] == "MIXED"]),
        "latency": latency_block(usable),
        "tokens": tokens_block(full),
        "judge": judge_block(full),
        "failures": failures(full),
    }


def flatten_metrics(report: dict) -> dict[str, float]:
    """Every rate / number in the report as `a.b.c -> float` for MLflow (a rate gives its value and `.n`)."""
    out: dict[str, float] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            if set(node) == {"k", "n", "value"}:
                if node["value"] is not None:
                    out[prefix] = float(node["value"])
                    out[f"{prefix}.n"] = float(node["n"])
                return
            for k, v in node.items():
                if k in ("confusion", "meta", "failures", "llm_misrouted", "errors", "note"):
                    continue
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            out[prefix] = float(node)

    walk("", {k: v for k, v in report.items() if k not in ("meta", "failures", "errors")})
    return out


# ---- text output ---------------------------------------------------------------------------------------
def _p(r: dict | None) -> str:
    return pct(r)


def _n_mark(n: int) -> str:
    return "*" if 0 < n < SMALL_N else ""


def _answer_lines(report: dict) -> list[str]:
    ans = report["answers"]
    lines = ["", "ANSWERS (answerable DOCUMENT questions, end to end; * = fewer than 10 questions)"]
    cols = [
        ("answer_accuracy", "correct"),
        ("accuracy_when_answered", "correct|answered"),
        ("groundedness", "grounded"),
        ("citation_strict", "cite strict"),
        ("citation_pm1", "cite ±1"),
        ("number_check_fail", "num-check fail"),
    ]
    lines.append(f"{'slice':<10}{'n':>4}   " + "".join(f"{h:>20}" for _, h in cols) + f"{'unjudged':>10}")
    for name, g in [("overall", ans["overall"])] + [(s, ans["by_slice"].get(s)) for s in DOC_SLICES]:
        if g is None:
            lines.append(f"{name:<10}{'-':>4}   (no questions: no READY document for this slice)")
            continue
        lines.append(
            f"{name:<10}{g['n']:>4}{_n_mark(g['n']):<3}"
            + "".join(f"{_p(g[k]):>20}" for k, _ in cols)
            + f"{g['unjudged']:>10}"
        )
    return lines


def format_report(report: dict) -> str:
    m = report["meta"]
    lines = [
        f"Full eval: run '{m.get('run')}' | git {m.get('git')} | eval-set {m.get('eval_set_hash')} | "
        f"{m['n_records']} questions scored ({m['n_full']} through the full pipeline)",
        f"models: answer {m.get('answer_model')}, router {m.get('router_model')}, "
        f"judge {report['judge']['model'] or m.get('judge_model') or 'none'} | "
        f"theta {m.get('theta')}, top_k {m.get('top_k')} | prompts {m.get('prompts')}",
    ]
    if report["errors"]:
        shown = ", ".join(report["errors"][:8]) + ("..." if len(report["errors"]) > 8 else "")
        lines.append(
            f"!! {len(report['errors'])} question(s) hit an LLM outage and are left out ({shown}). "
            "Run the same command again to retry them."
        )
    if m.get("n_skipped"):
        lines.append(
            f"skipped {m['n_skipped']} question(s) (documents not READY, ...): see 'skipped' in the JSON"
        )

    # router
    rt = report["router"]
    if rt["n"]:
        lines += ["", f"ROUTER ({rt['n']} questions; DOC/GEN = accuracy on DOCUMENT+GENERAL rows only)"]
        lines.append(
            f"{'variant':<18}{'accuracy':>20}{'DOC/GEN only':>20}{'DOCUMENT recall':>20}{'MIXED recall':>20}"
        )
        for name, v in rt["variants"].items():
            lines.append(
                f"{name:<18}{_p(v['accuracy']):>20}{_p(v['accuracy_document_general']):>20}"
                f"{_p(v['document_recall']):>20}{_p(v['recall']['MIXED']):>20}"
            )
        for name, v in rt["variants"].items():
            lines.append(f"  confusion, {name} (rows = true, cols = predicted; DOC GEN MIX)")
            for t in ROUTES:
                lines.append(f"    {t:<9}" + "".join(f"{v['confusion'][t][p]:>5}" for p in ROUTES))
        llm = rt["variants"]["llm_router"]
        lines.append(f"  LLM router fell back to DOCUMENT on {llm['fallbacks']} question(s)")
        sb = rt["variants"]["score_threshold"]
        lines.append(f"  score baseline: {sb['note']} (tau = {sb['tau']})")
        for name, v in rt["variants"].items():
            dg = v["accuracy_document_general"]["value"]
            if name != "llm_router" and dg is not None and dg >= 0.9 and rt["n"] >= SMALL_N:
                lines.append(
                    f"  NOTE: the {name} baseline already reaches {dg * 100:.1f}% on DOCUMENT/GENERAL rows. "
                    "The LLM router's value then rests on MIXED (see the recall column), and a hybrid "
                    "(rules first, LLM only when unsure) is a possible cost saving."
                )
        if rt["llm_misrouted"]:
            lines.append(
                "  LLM misrouted: "
                + ", ".join(
                    f"{x['id']} ({x['gold'][:3]}->{x['predicted'][:3]})" for x in rt["llm_misrouted"][:12]
                )
            )

    # retrieval
    rv = report["retrieval"]
    if rv["overall"]["n"]:
        lines += ["", "RETRIEVAL (answerable DOCUMENT questions; hit = a retrieved chunk on a gold page)"]
        lines.append(
            f"{'slice':<10}{'n':>4}   {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@8':>6} {'R@5±1':>6} {'MRR':>6}"
        )
        for name, g in [("overall", rv["overall"]), *rv["by_slice"].items()]:
            lines.append(
                f"{name:<10}{g['n']:>4}{_n_mark(g['n']):<3}"
                f"{g['recall_at_1'] * 100:>6.1f}% {g['recall_at_3'] * 100:>5.1f}% "
                f"{g['recall_at_5'] * 100:>5.1f}% {g['recall_at_8'] * 100:>5.1f}% "
                f"{g['recall_at_5_pm1'] * 100:>5.1f}% {g['mrr']:>6.3f}"
            )

    if not m["n_full"]:
        lines += [
            "",
            "(router-only run: no answers were generated, so no answer/abstention/latency sections)",
        ]
        return "\n".join(lines)
    lines += _answer_lines(report)

    # abstention
    ab = report["abstention"]["overall"]
    if ab["n_unanswerable"] or ab["n_answerable"]:
        lines += [
            "",
            f"ABSTENTION ({ab['n_unanswerable']} unanswerable, {ab['n_answerable']} answerable questions)",
        ]
        lines.append(f"  abstain recall        {_p(ab['abstain_recall'])}   (unanswerable ones it declined)")
        lines.append(
            f"  false-answer rate     {_p(ab['false_answer_rate'])}   (unanswerable ones it answered)"
        )
        lines.append(f"  abstain precision     {_p(ab['abstain_precision'])}   (declines that were right)")
        lines.append(
            f"  wrong-abstention rate {_p(ab['wrong_abstention_rate'])}   (answerable ones it declined)"
        )
        if ab["other_unanswerable"] or ab["other_answerable"]:
            lines.append(
                "  not on the document path (misrouted / not ready): "
                f"{ab['other_unanswerable']} unanswerable, {ab['other_answerable']} answerable"
            )
        lines.append(f"  abstain reasons: {ab['abstain_reasons']}")

    # general / mixed
    g, mx = report["general"], report["mixed"]
    if g["n"]:
        lines += [
            "",
            f"GENERAL ({g['n']}): routed GENERAL {_p(g['routed_general'])}, answered {_p(g['answered'])}, "
            f"correct {_p(g['answer_accuracy'])}, unjudged {g['unjudged']}",
        ]
    if mx["n"]:
        lines += [
            f"MIXED ({mx['n']}): routed MIXED {_p(mx['routed_mixed'])}, "
            f"document part correct {_p(mx['document_part_correct'])}, "
            f"general part correct {_p(mx['general_part_correct'])}, both {_p(mx['both_correct'])}, "
            f"unjudged {mx['unjudged']}"
        ]

    # latency
    lat = report["latency"]
    lines += [
        "",
        f"LATENCY ms ({lat['n_live']} questions with no cached call; {lat['n_cached_excluded']} left out)",
    ]
    if lat["n_live"]:
        lines.append(f"{'stage':<18}{'n':>5}{'p50':>9}{'p95':>9}{'p99':>9}")
        for stage, s in lat["stages"].items():
            if s["n"]:
                lines.append(f"{stage:<18}{s['n']:>5}{s['p50']:>9}{s['p95']:>9}{s['p99']:>9}")
    else:
        lines.append(
            "  none: every question was replayed from the dev cache. Use --no-cache for a latency run."
        )

    # tokens
    tk = report["tokens"]
    if tk["n"]:
        t = tk["sum"]["total"]
        lines += [
            "",
            f"TOKENS ({tk['n']} questions): {t['total']:,} total ({t['prompt']:,} in, "
            f"{t['completion']:,} out), {tk['mean_total_per_question']:,.0f} per question; "
            f"cost-equivalent ${tk['cost_usd_equiv']:.4f} (${tk['cost_usd_equiv_per_question']:.5f} per "
            "question at list price; we run on a free tier)",
        ]
    jd = report["judge"]
    if jd["n_verdicts"]:
        v = jd["vs_numeric_match"]
        lines.append(
            f"JUDGE {jd['model']} / {jd['prompt_version']}: {jd['n_verdicts']} verdicts, "
            f"{jd['tokens']:,} tokens; agrees with the number match on {_p(v['agree'])} numeric questions"
        )
    if report["failures"]:
        lines += ["", "NOT CORRECT (answerable DOCUMENT questions, first few)"]
        for f in report["failures"]:
            lines.append(
                f"  {f['id']} {f['outcome']}/{f['reason']}: gold '{f['gold']}'"
                + (f" got '{' '.join(f['answer'].split())[:90]}'" if f["answer"] else "")
            )
    return "\n".join(lines)


def format_markdown(report: dict) -> str:
    """README-ready tables."""
    m = report["meta"]
    out = [f"<!-- eval run '{m.get('run')}', git {m.get('git')}, eval-set {m.get('eval_set_hash')} -->", ""]
    rt = report["router"]
    if rt["n"]:
        out += [
            f"**Router** ({rt['n']} labelled questions)",
            "",
            "| Variant | Accuracy | DOCUMENT/GENERAL only | DOCUMENT recall | MIXED recall |",
            "|---|---|---|---|---|",
        ]
        label = {
            "keyword_rules": "Keyword rules",
            "score_threshold": "Retrieval-score threshold",
            "llm_router": "LLM router",
        }
        for name, v in rt["variants"].items():
            out.append(
                f"| {label.get(name, name)} | {_p(v['accuracy'])} | {_p(v['accuracy_document_general'])} | "
                f"{_p(v['document_recall'])} | {_p(v['recall']['MIXED'])} |"
            )
        out.append("")
    ans = report["answers"]
    out += [
        "**Answers** (answerable document questions, end to end)",
        "",
        "| Slice | n | Correct | Grounded | Cited page (strict) | Cited page (±1) | Number check failed |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, g in [("overall", ans["overall"])] + [(s, ans["by_slice"].get(s)) for s in DOC_SLICES]:
        if g is None:
            out.append(f"| {name} | 0 | n/a | n/a | n/a | n/a | n/a |")
            continue
        out.append(
            f"| {name} | {g['n']} | {_p(g['answer_accuracy'])} | {_p(g['groundedness'])} | "
            f"{_p(g['citation_strict'])} | {_p(g['citation_pm1'])} | {_p(g['number_check_fail'])} |"
        )
    ab = report["abstention"]["overall"]
    out += [
        "",
        "**Abstention**",
        "",
        "| Abstain recall | False-answer rate | Abstain precision | Wrong-abstention rate |",
        "|---|---|---|---|",
        f"| {_p(ab['abstain_recall'])} | {_p(ab['false_answer_rate'])} | "
        f"{_p(ab['abstain_precision'])} | {_p(ab['wrong_abstention_rate'])} |",
    ]
    lat = report["latency"]
    if lat["n_live"]:
        out += [
            "",
            f"**Latency** (ms, {lat['n_live']} live questions)",
            "",
            "| Stage | p50 | p95 | p99 |",
            "|---|---|---|---|",
        ]
        for stage, s in lat["stages"].items():
            if s["n"]:
                out.append(f"| {stage} | {s['p50']} | {s['p95']} | {s['p99']} |")
    return "\n".join(out) + "\n"
