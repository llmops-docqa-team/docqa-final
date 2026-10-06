"""Metric computations for the Metrics page (design §15). Plain functions over lists of dict rows, so they are
testable without Streamlit or a database; `ui/views/metrics.py` only renders what `build_dashboard` returns.

Rows are `requests` rows. Rows without a `status` (a 👍/👎 whose trace id was never logged) are ignored.
A stage that did not run is NULL in the log, so every latency figure is over requests that ran that stage.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Callable

from app.config import ObservabilityConfig, Threshold
from app.observability.failure_reasons import classify_ingest_failure

Row = dict[str, Any]

STAGES = {
    "total": "t_total_ms",
    "router": "t_router_ms",
    "embed": "t_embed_ms",
    "retrieve": "t_retrieve_ms",
    "llm": "t_llm_ms",
}

# time-range filter choices for the page: label -> how far back (None = everything)
RANGES: dict[str, timedelta | None] = {
    "Last hour": timedelta(hours=1),
    "Last 24 hours": timedelta(hours=24),
    "Last 7 days": timedelta(days=7),
    "Last 30 days": timedelta(days=30),
    "All time": None,
}

GREEN, AMBER, RED, GREY = "green", "amber", "red", "grey"


# ---------------------------------------------------------------- small statistics


def percentile(values: list[float], q: float) -> float | None:
    """q in [0, 100], linear interpolation between ranks (numpy's default). None for no data."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "mean": sum(values) / len(values) if values else None,
    }


@dataclass(frozen=True)
class Ratio:
    count: int
    total: int

    @property
    def value(self) -> float | None:
        return self.count / self.total if self.total else None


def ratio(rows: list[Row], hit: Callable[[Row], bool], of: Callable[[Row], bool] = lambda r: True) -> Ratio:
    """How many of the rows that satisfy `of` also satisfy `hit`."""
    pool = [r for r in rows if of(r)]
    return Ratio(sum(1 for r in pool if hit(r)), len(pool))


def _nums(rows: list[Row], col: str) -> list[float]:
    return [float(r[col]) for r in rows if r.get(col) is not None]


# ---------------------------------------------------------------- reading the database


def _connect_readonly(path: str | Path) -> sqlite3.Connection | None:
    p = Path(path)
    if not p.exists():
        return None
    conn = sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _read(path: str | Path, sql: str, params: tuple = ()) -> list[Row]:
    """Rows as dicts; an unreadable or not-yet-created database / table gives no rows."""
    conn = _connect_readonly(path)
    if conn is None:
        return []
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def since_iso(range_label: str, now: datetime | None = None) -> str | None:
    """Lower bound for `ts` in the log's own format (UTC ISO, so plain string comparison orders it)."""
    delta = RANGES[range_label]
    if delta is None:
        return None
    return ((now or datetime.now(UTC)) - delta).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def load_requests(path: str | Path, since: str | None = None) -> list[Row]:
    sql = "SELECT * FROM requests WHERE status IS NOT NULL" + (" AND ts >= ?" if since else "")
    return _read(path, sql + " ORDER BY ts", (since,) if since else ())


def load_documents(path: str | Path, since: str | None = None) -> list[Row]:
    sql = "SELECT * FROM documents" + (" WHERE created_at >= ?" if since else "")
    return _read(path, sql, (since,) if since else ())


def load_upload_failures(path: str | Path, since: str | None = None) -> list[Row]:
    sql = "SELECT * FROM upload_failures" + (" WHERE ts >= ?" if since else "")
    return _read(path, sql, (since,) if since else ())


# ---------------------------------------------------------------- the five categories


def _made_llm_call(r: Row) -> bool:
    return r.get("llm_backend") is not None  # NULL for rows from before step 11 and for no-LLM requests


def operational(rows: list[Row], documents: list[Row]) -> dict[str, Any]:
    tokens = [(r["tokens_in"] or 0) + (r["tokens_out"] or 0) for r in rows if r.get("tokens_in") is not None]
    costs = _nums(rows, "cost_usd_equiv")
    return {
        "latency_ms": {stage: summarize(_nums(rows, col)) for stage, col in STAGES.items()},
        "error_rate": ratio(rows, lambda r: r["status"] == "error"),
        # Step 11: of the requests that made an LLM call, how many fell back to Ollama / saw a 429.
        "llm_fallback_rate": ratio(rows, lambda r: (r.get("llm_fallbacks") or 0) > 0, of=_made_llm_call),
        "llm_rate_limited_rate": ratio(
            rows, lambda r: (r.get("llm_rate_limited") or 0) > 0, of=_made_llm_call
        ),
        "tokens_per_request": sum(tokens) / len(tokens) if tokens else None,
        "cost_per_request_usd": sum(costs) / len(costs) if costs else None,
        "cost_total_usd": sum(costs),
        "ingestion": ingestion_speed(documents),
    }


def ingestion_speed(documents: list[Row]) -> dict[str, Any]:
    """Seconds per page for READY documents, per document and overall (median and pooled)."""
    per_doc = []
    for d in documents:
        pages, secs = d.get("pages_total"), d.get("ingest_seconds")
        if d.get("status") == "READY" and pages and secs is not None:
            per_doc.append(
                {
                    "filename": d["filename"],
                    "pages": pages,
                    "seconds": round(secs, 1),
                    "s_per_page": round(secs / pages, 2),
                    "ocr_pages": d.get("n_ocr_pages") or 0,
                }
            )
    pooled = sum(p["seconds"] for p in per_doc) / sum(p["pages"] for p in per_doc) if per_doc else None
    return {
        "documents": per_doc,
        "median_s_per_page": median(p["s_per_page"] for p in per_doc) if per_doc else None,
        "pooled_s_per_page": pooled,
    }


def question_length_histogram(rows: list[Row], bin_width: int = 50, max_len: int = 500) -> list[dict]:
    """Counts per length bucket ("0-49", "50-99", ...); anything above `max_len` goes in the last bucket."""
    counts: Counter[int] = Counter()
    for v in _nums(rows, "question_len"):
        counts[min(int(v) // bin_width, max_len // bin_width)] += 1
    top = max_len // bin_width
    return [
        {
            "bucket": f"{i * bin_width}-{(i + 1) * bin_width - 1}" if i < top else f"{i * bin_width}+",
            "count": counts[i],
        }
        for i in range(top + 1)
    ]


def upload_failures_by_reason(rejections: list[Row], documents: list[Row]) -> dict[str, int]:
    """Refused uploads (reason as logged) plus documents that were accepted but ended FAILED."""
    counts: Counter[str] = Counter(r["reason"] for r in rejections)
    counts.update(classify_ingest_failure(d.get("error")) for d in documents if d.get("status") == "FAILED")
    return dict(counts.most_common())


def input_metrics(rows: list[Row], rejections: list[Row], documents: list[Row]) -> dict[str, Any]:
    routes = Counter(r["route"] for r in rows if r.get("route"))
    return {
        "question_len": summarize(_nums(rows, "question_len")),
        "question_len_histogram": question_length_histogram(rows),
        "route_mix": dict(routes),
        "upload_failures": upload_failures_by_reason(rejections, documents),
    }


def _router_ran(r: Row) -> bool:
    """The router made an LLM call (it is skipped when nothing is uploaded) or it failed trying."""
    return r.get("route") is not None and (r.get("t_router_ms") is not None or r.get("router_ok") == 0)


def output_metrics(rows: list[Row]) -> dict[str, Any]:
    answered = [r for r in rows if r["status"] == "answered"]
    reasons = Counter(
        r["abstain_reason"] for r in rows if r["status"] == "abstained" and r.get("abstain_reason")
    )
    return {
        "abstention_rate": ratio(rows, lambda r: r["status"] == "abstained"),
        "abstain_reasons": dict(reasons.most_common()),
        "citation_invalid_rate": ratio(
            rows, lambda r: r["citations_valid"] == 0, of=lambda r: r.get("citations_valid") is not None
        ),
        "number_check_fail_rate": ratio(
            rows,
            lambda r: r["number_check"] == "fail",
            of=lambda r: r.get("number_check") in ("pass", "fail"),
        ),
        "router_fallback_rate": ratio(rows, lambda r: r["router_ok"] == 0, of=_router_ran),
        "router_fallback_reasons": dict(
            Counter(r["router_fallback"] for r in rows if r.get("router_fallback")).most_common()
        ),
        "answer_chars": summarize(_nums(answered, "answer_chars")),
    }


def quality_metrics(rows: list[Row]) -> dict[str, Any]:
    up = sum(1 for r in rows if r.get("feedback") == 1)
    down = sum(1 for r in rows if r.get("feedback") == -1)
    answered = sum(1 for r in rows if r["status"] == "answered")
    judged_c = [r["judge_correct"] for r in rows if r.get("judge_correct") is not None]
    judged_g = [r["judge_grounded"] for r in rows if r.get("judge_grounded") is not None]
    return {
        "thumbs_up": up,
        "thumbs_down": down,
        "thumbs_down_share": Ratio(down, up + down),
        "rated_share_of_answers": Ratio(up + down, answered) if answered else Ratio(0, 0),
        # Filled by the step 09 judge script (requests.judge_correct / judge_grounded, 0/1).
        "judge_correct": Ratio(sum(judged_c), len(judged_c)),
        "judge_grounded": Ratio(sum(judged_g), len(judged_g)),
    }


def daily_median_top_score(rows: list[Row]) -> list[dict]:
    """One point per UTC day: median top retrieval score of that day's document-path requests."""
    by_day: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("top_score") is not None and r.get("ts"):
            by_day[r["ts"][:10]].append(float(r["top_score"]))
    return [{"date": d, "median_top_score": median(v), "n": len(v)} for d, v in sorted(by_day.items())]


def _week_start(ts: str) -> date:
    d = date.fromisoformat(ts[:10])
    return d - timedelta(days=d.weekday())  # Monday


def _mix(rows: list[Row]) -> dict[str, float]:
    counts = Counter(r["route"] for r in rows if r.get("route"))
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def route_mix_by_week(rows: list[Row]) -> list[dict]:
    weeks: dict[date, list[Row]] = defaultdict(list)
    for r in rows:
        if r.get("route") and r.get("ts"):
            weeks[_week_start(r["ts"])].append(r)
    out = []
    for wk, rs in sorted(weeks.items()):
        out.append({"week": wk.isoformat(), "n": len(rs), **{f"{k}": v for k, v in _mix(rs).items()}})
    return out


def route_mix_shift(rows: list[Row]) -> dict[str, Any] | None:
    """Latest week with data vs the calendar week before it: total variation distance
    (0 = same mix, 1 = disjoint). None when there is no earlier week to compare with."""
    weeks: dict[date, list[Row]] = defaultdict(list)
    for r in rows:
        if r.get("route") and r.get("ts"):
            weeks[_week_start(r["ts"])].append(r)
    if not weeks:
        return None
    this = max(weeks)
    prev = this - timedelta(days=7)
    if prev not in weeks:
        return None
    a, b = _mix(weeks[this]), _mix(weeks[prev])
    shift = 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))
    return {
        "this_week": this.isoformat(),
        "last_week": prev.isoformat(),
        "n_this": len(weeks[this]),
        "n_last": len(weeks[prev]),
        "mix_this": a,
        "mix_last": b,
        "shift": shift,
    }


def drift_metrics(rows: list[Row]) -> dict[str, Any]:
    return {
        "daily_top_score": daily_median_top_score(rows),
        "route_mix_by_week": route_mix_by_week(rows),
        "route_mix_shift": route_mix_shift(rows),
    }


# ---------------------------------------------------------------- colour indicators


def indicator(value: float | None, rule: Threshold | None, n: int, min_n: int) -> str:
    """green | amber | red, or grey when there is no rule, no value or too few requests to judge."""
    if rule is None or value is None or n < min_n:
        return GREY
    worse = (lambda x, t: x < t) if rule.lower_is_worse else (lambda x, t: x > t)
    if rule.red is not None and worse(value, rule.red):
        return RED
    if rule.amber is not None and worse(value, rule.amber):
        return AMBER
    return GREEN


def build_indicators(d: dict[str, Any], cfg: ObservabilityConfig) -> dict[str, dict[str, Any]]:
    """name -> {value, n, color} for every threshold the page shows."""
    out = d["output"]
    total = d["operational"]["latency_ms"]["total"]
    daily = d["drift"]["daily_top_score"]
    shift = d["drift"]["route_mix_shift"]
    latest = daily[-1] if daily else None
    raw = {
        "p95_total_ms": (total["p95"], total["n"]),
        "error_rate": (d["operational"]["error_rate"].value, d["operational"]["error_rate"].total),
        "abstention_rate": (out["abstention_rate"].value, out["abstention_rate"].total),
        "citation_invalid_rate": (out["citation_invalid_rate"].value, out["citation_invalid_rate"].total),
        "number_check_fail_rate": (out["number_check_fail_rate"].value, out["number_check_fail_rate"].total),
        "router_fallback_rate": (out["router_fallback_rate"].value, out["router_fallback_rate"].total),
        "thumbs_down_share": (
            d["quality"]["thumbs_down_share"].value,
            d["quality"]["thumbs_down_share"].total,
        ),
        "median_top_score": (latest["median_top_score"], latest["n"]) if latest else (None, 0),
        "route_mix_shift": (shift["shift"], min(shift["n_this"], shift["n_last"])) if shift else (None, 0),
    }
    return {
        name: {
            "value": value,
            "n": n,
            "color": indicator(value, cfg.thresholds.get(name), n, cfg.min_requests_for_indicator),
        }
        for name, (value, n) in raw.items()
    }


def build_dashboard(
    rows: list[Row], documents: list[Row], rejections: list[Row], cfg: ObservabilityConfig
) -> dict[str, Any]:
    d = {
        "n_requests": len(rows),
        "operational": operational(rows, documents),
        "input": input_metrics(rows, rejections, documents),
        "output": output_metrics(rows),
        "quality": quality_metrics(rows),
        "drift": drift_metrics(rows),
    }
    d["indicators"] = build_indicators(d, cfg)
    return d
