"""Load test for POST /query: 1 / 3 / 5 concurrent users send a mix of eval questions (design §14).

    python scripts/load_test.py                        # against http://localhost:8000, defaults below
    python scripts/load_test.py --levels 1,3 --max-requests 4 --token-budget 30000

For each concurrency level it reports requests/min, p50 / p95 latency, the error rate, the 429 count and
whether the level held up, then says where the system first breaks. Results go to
eval/results/load_test.json.

It spends real LLM quota, so it is small on purpose. Groq's free tier allows about 8,000 tokens a minute
and 200,000 a day per model, and one question costs roughly 1.5K-5K tokens. Every level stops at
`--duration` seconds or `--max-requests` requests, whichever comes first, and the whole run stops issuing
requests once `--token-budget` tokens have been spent. Levels are separated by `--cooldown` seconds so each
starts with a fresh per-minute allowance. The questions come from eval/questions.jsonl, so the documents
those questions are about should be uploaded.

A 429 is counted where the API reports it: the Groq backend's 429 responses that the answer went through
(`llm.rate_limited` in the /query reply, retried or not), plus any HTTP 429 the API itself returns.
The API answers 200 even when the LLM was unavailable (the section says `error`), so that case is counted as
"degraded", and the error rate is HTTP failures + timeouts + degraded answers over all requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "eval" / "questions.jsonl"
DEFAULT_OUT = ROOT / "eval" / "results" / "load_test.json"

DEFAULT_WEIGHTS = {"DOCUMENT": 0.5, "GENERAL": 0.3, "MIXED": 0.2}
BROKEN_ERROR_RATE = 0.05  # design §15: error rate above 5% is red
SLOW_P95_MS = 6000  # design §15: p95 above 6 s is red
STEADY_SECONDS = 30  # a level shorter than this is a burst: its per-minute figures are extrapolated
GROQ_FREE_TPM = 8000  # Groq free tier, tokens per minute per model (docs/design.md §12)


# ---------------------------------------------------------------- pure helpers


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (p in 0-100); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def load_questions(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({"question": row["question"], "route": row.get("route", "DOCUMENT")})
    return rows


def parse_weights(text: str) -> dict[str, float]:
    """'document=0.5,general=0.3,mixed=0.2' -> {'DOCUMENT': 0.5, ...}"""
    out: dict[str, float] = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        out[name.strip().upper()] = float(value)
    if not out or any(w < 0 for w in out.values()) or sum(out.values()) <= 0:
        raise ValueError(f"bad --weights {text!r}")
    return out


def build_plan(
    questions: list[dict[str, str]], weights: dict[str, float], n: int, seed: int
) -> list[dict[str, str]]:
    """`n` questions drawn at the route weights, in a fixed order, so every level sees the same sequence."""
    rng = random.Random(seed)
    by_route: dict[str, list[dict[str, str]]] = {}
    for q in questions:
        by_route.setdefault(q["route"], []).append(q)
    routes = [r for r in weights if by_route.get(r)]
    if not routes:
        raise ValueError("none of the question routes in --weights exist in the question file")
    pools = {r: rng.sample(by_route[r], len(by_route[r])) for r in routes}
    plan = []
    for i in range(n):
        route = rng.choices(routes, weights=[weights[r] for r in routes])[0]
        plan.append(pools[route][i % len(pools[route])])
    return plan


def classify(status: int | None, body: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    """One finished request -> its outcome. `error` is 'timeout' / 'connection' when no response came."""
    out: dict[str, Any] = {
        "http_status": status,
        "kind": "ok",
        "rate_limited": 0,
        "fallback_calls": 0,
        "tokens": 0,
        "route": None,
    }
    if error is not None:
        out["kind"] = error
        return out
    if status == 429:
        out["rate_limited"] = 1
    if status is None or status >= 400:
        out["kind"] = "http_error"
        return out
    body = body or {}
    out["route"] = body.get("route")
    llm = body.get("llm") or {}
    out["rate_limited"] += int(llm.get("rate_limited") or 0)
    out["fallback_calls"] = int(llm.get("fallback_calls") or 0)
    out["tokens"] = int(((body.get("tokens") or {}).get("total") or {}).get("total") or 0)
    if any(s.get("status") == "error" for s in body.get("sections") or []):
        out["kind"] = "degraded"  # 200, but an LLM call failed for good
    return out


def summarize_level(
    concurrency: int, outcomes: list[dict[str, Any]], latencies_ms: list[float], elapsed_s: float
) -> dict[str, Any]:
    n = len(outcomes)
    kinds = Counter(o["kind"] for o in outcomes)
    failures = kinds["http_error"] + kinds["timeout"] + kinds["connection"]
    errors = failures + kinds["degraded"]
    rate = errors / n if n else 0.0
    count_429 = sum(o["rate_limited"] for o in outcomes)
    tokens = sum(o["tokens"] for o in outcomes)
    p95 = percentile(latencies_ms, 95)
    if rate >= BROKEN_ERROR_RATE:
        verdict = "broken"
    elif count_429 or (p95 is not None and p95 > SLOW_P95_MS):
        verdict = "strained"
    else:
        verdict = "ok"
    return {
        "concurrency": concurrency,
        "requests": n,
        "duration_s": round(elapsed_s, 1),
        "req_per_min": round(n / elapsed_s * 60, 2) if elapsed_s > 0 else None,
        "ok_per_min": round((n - errors) / elapsed_s * 60, 2) if elapsed_s > 0 else None,
        "p50_ms": _r(percentile(latencies_ms, 50)),
        "p95_ms": _r(p95),
        "max_ms": _r(max(latencies_ms) if latencies_ms else None),
        "error_rate": round(rate, 3),
        "failure_rate": round(failures / n, 3) if n else 0.0,  # HTTP 4xx/5xx + timeouts + connection errors
        "degraded": kinds["degraded"],  # answered 200, but an LLM call failed for good (usually Groq 429s)
        "http_errors": kinds["http_error"],
        "timeouts": kinds["timeout"],
        "connection_errors": kinds["connection"],
        "count_429": count_429,
        "fallback_calls": sum(o["fallback_calls"] for o in outcomes),
        "tokens": tokens,
        "tokens_per_min": round(tokens / elapsed_s * 60) if elapsed_s > 0 else None,
        "extrapolated": elapsed_s < STEADY_SECONDS,  # per-minute figures come from a burst shorter than that
        "routes": dict(Counter(o["route"] or "?" for o in outcomes)),
        "verdict": verdict,
    }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


def find_breaks(levels: list[dict[str, Any]]) -> dict[str, Any]:
    """The first concurrency at which each thing happened (None = it did not, within what was run)."""

    def first(pred: Callable[[dict[str, Any]], bool]) -> int | None:
        return next((lv["concurrency"] for lv in levels if "skipped" not in lv and pred(lv)), None)

    return {
        "first_429_at": first(lambda lv: lv["count_429"] > 0),
        "first_slow_at": first(lambda lv: (lv["p95_ms"] or 0) > SLOW_P95_MS),
        "breaks_at": first(lambda lv: lv["verdict"] == "broken"),
        "rule": f"broken = error rate >= {BROKEN_ERROR_RATE:.0%}; slow = p95 > {SLOW_P95_MS} ms",
    }


# ---------------------------------------------------------------- the run


async def one_request(client: httpx.AsyncClient, question: str, timeout: float) -> tuple[dict, float]:
    t0 = time.perf_counter()
    status: int | None = None
    body: dict[str, Any] | None = None
    error: str | None = None
    try:
        resp = await client.post("/query", json={"question": question}, timeout=timeout)
        status = resp.status_code
        if status == 200:
            body = resp.json()
    except httpx.TimeoutException:
        error = "timeout"
    except httpx.TransportError:
        error = "connection"
    except ValueError:  # 200 with a body that is not JSON
        status = None
    return classify(status, body, error), (time.perf_counter() - t0) * 1000


async def run_level(
    client: httpx.AsyncClient,
    concurrency: int,
    plan: list[dict[str, str]],
    *,
    duration: float,
    max_requests: int,
    timeout: float,
    tokens_left: Callable[[], int | None],
) -> dict[str, Any]:
    """`concurrency` users, each sending one question after another until the time or request cap is hit."""
    outcomes: list[dict[str, Any]] = []
    latencies: list[float] = []
    issued = 0
    t0 = time.monotonic()
    deadline = t0 + duration

    async def user() -> None:
        nonlocal issued
        while time.monotonic() < deadline and issued < max_requests:
            left = tokens_left()  # what the levels before this one left; add this level's own spend
            if left is not None and left - sum(o["tokens"] for o in outcomes) <= 0:
                return  # (requests already in flight can overshoot the budget by up to `concurrency`)
            q = plan[issued % len(plan)]
            issued += 1
            outcome, ms = await one_request(client, q["question"], timeout)
            outcomes.append(outcome)
            latencies.append(ms)

    await asyncio.gather(*(user() for _ in range(concurrency)))
    return summarize_level(concurrency, outcomes, latencies, time.monotonic() - t0)


async def run_load_test(
    client: httpx.AsyncClient,
    questions: list[dict[str, str]],
    *,
    levels: list[int],
    duration: float,
    max_requests: int,
    token_budget: int | None,
    cooldown: float,
    timeout: float,
    seed: int,
    weights: dict[str, float],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    plan = build_plan(questions, weights, max(max_requests, 1), seed)
    spent = 0
    results: list[dict[str, Any]] = []

    def tokens_left() -> int | None:
        return None if token_budget is None else token_budget - spent

    for i, concurrency in enumerate(levels):
        if token_budget is not None and spent >= token_budget:
            results.append({"concurrency": concurrency, "skipped": "token budget reached"})
            log(f"{concurrency} user(s): skipped (token budget {token_budget} reached)")
            continue
        if i and cooldown > 0:
            log(f"  cooling down {cooldown:.0f}s so the per-minute quota refills ...")
            await sleep(cooldown)
        log(f"running {concurrency} concurrent user(s) ...")
        level = await run_level(
            client,
            concurrency,
            plan,
            duration=duration,
            max_requests=max_requests,
            timeout=timeout,
            tokens_left=tokens_left,
        )
        spent += level["tokens"]
        results.append(level)
        log("  " + describe(level))
    return {
        "meta": {
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_url": str(client.base_url),
            "levels": levels,
            "duration_s_per_level": duration,
            "max_requests_per_level": max_requests,
            "token_budget": token_budget,
            "tokens_spent": spent,
            "cooldown_s": cooldown,
            "request_timeout_s": timeout,
            "seed": seed,
            "weights": weights,
            "n_questions": len(questions),
        },
        "levels": results,
        "summary": find_breaks(results),
    }


def describe(lv: dict[str, Any]) -> str:
    if "skipped" in lv:
        return f"{lv['concurrency']} user(s): skipped ({lv['skipped']})"
    return (
        f"{lv['concurrency']} user(s): {lv['requests']} requests, {lv['req_per_min']} req/min, "
        f"p50 {lv['p50_ms']} ms, p95 {lv['p95_ms']} ms, errors {lv['error_rate']:.1%} "
        f"(degraded {lv['degraded']}), 429s {lv['count_429']}, {lv['tokens']} tokens -> {lv['verdict']}"
    )


def markdown_table(report: dict[str, Any]) -> str:
    rows = [
        "| Users | Requests | req/min | p50 ms | p95 ms | Error rate | Degraded | 429s | Tokens | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lv in report["levels"]:
        if "skipped" in lv:
            rows.append(f"| {lv['concurrency']} | skipped: {lv['skipped']} | | | | | | | | |")
            continue
        cells = [
            lv["concurrency"],
            lv["requests"],
            f"{lv['req_per_min']}{'*' if lv.get('extrapolated') else ''}",
            lv["p50_ms"],
            lv["p95_ms"],
            f"{lv['error_rate']:.1%}",
            lv["degraded"],
            lv["count_429"],
            lv["tokens"],
            lv["verdict"],
        ]
        rows.append("| " + " | ".join(str(c) for c in cells) + " |")
    if any(lv.get("extrapolated") for lv in report["levels"]):
        rows.append(
            f"\n\\* a burst shorter than {STEADY_SECONDS} s, so req/min is extrapolated, not sustainable: "
            f"the Groq free tier allows about {GROQ_FREE_TPM:,} tokens a minute per model, "
            "i.e. a few questions a minute."
        )
    return "\n".join(rows)


# ---------------------------------------------------------------- CLI


async def preflight(client: httpx.AsyncClient) -> str | None:
    """Return a warning (or None). Raises httpx.HTTPError when the API is not reachable."""
    (await client.get("/health", timeout=10)).raise_for_status()
    docs = (await client.get("/documents", timeout=10)).json()
    docs = docs.get("documents", docs) if isinstance(docs, dict) else docs
    ready = [d for d in docs if d.get("status") in ("READY", "PARTIAL")]
    if not ready:
        return "no document is READY, so every question will be answered as general knowledge"
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Load test POST /query (small on purpose: it spends LLM quota).")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--levels", default="1,3,5", help="comma-separated concurrent users")
    ap.add_argument(
        "--duration", type=float, default=30, help="seconds per level (a cap; see --max-requests)"
    )
    ap.add_argument("--max-requests", type=int, default=6, help="requests per level (a cap)")
    ap.add_argument(
        "--token-budget", type=int, default=60000, help="stop issuing requests after this many tokens"
    )
    ap.add_argument("--cooldown", type=float, default=65, help="seconds between levels")
    ap.add_argument("--timeout", type=float, default=90, help="per-request timeout in seconds")
    ap.add_argument("--weights", default="document=0.5,general=0.3,mixed=0.2", help="question mix by route")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return ap.parse_args(argv)


async def amain(args: argparse.Namespace) -> int:
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    questions = load_questions(args.questions)
    async with httpx.AsyncClient(base_url=args.base_url) as client:
        try:
            warning = await preflight(client)
        except httpx.HTTPError as exc:
            print(
                f"error: cannot reach {args.base_url} ({type(exc).__name__}). Is the API running?",
                file=sys.stderr,
            )
            return 2
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
        budget = args.token_budget or None
        worst = sum(args.max_requests for _ in levels)
        print(
            f"{len(levels)} levels x up to {args.max_requests} requests ({worst} at most), "
            f"token budget {budget or 'none'}; about 1.5K-5K tokens per request."
        )
        report = await run_load_test(
            client,
            questions,
            levels=levels,
            duration=args.duration,
            max_requests=args.max_requests,
            token_budget=budget,
            cooldown=args.cooldown,
            timeout=args.timeout,
            seed=args.seed,
            weights=parse_weights(args.weights),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n" + markdown_table(report))
    s = report["summary"]
    print(
        f"\nfirst 429 at: {s['first_429_at'] or 'never'} user(s); first slow (p95) at: "
        f"{s['first_slow_at'] or 'never'}; breaks at: {s['breaks_at'] or 'did not break'} ({s['rule']})"
    )
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(amain(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
