"""Metrics page (design §15): the five monitoring categories, read straight from the SQLite request log.

A thin view: every number comes from `app.observability.metrics.build_dashboard`. Colours come from the
thresholds in config.yaml (`observability.thresholds`); there is no alerting.
"""

import pandas as pd
import streamlit as st

from app import config
from app.observability import metrics as m

DOT = {m.GREEN: "🟢", m.AMBER: "🟠", m.RED: "🔴", m.GREY: "⚪"}


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def fmt_pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.1f}%"


def fmt_num(v: float | None, digits: int = 0) -> str:
    return "–" if v is None else f"{v:,.{digits}f}"


def fmt_usd(v: float | None) -> str:
    return "–" if v is None else f"${v:.5f}" if v < 0.01 else f"${v:.3f}"


def rule_caption(rule, fmt) -> str:
    if rule is None:
        return ""
    sign = "<" if rule.lower_is_worse else ">"
    parts = [
        f"red {sign} {fmt(rule.red)}" if rule.red is not None else "",
        f"amber {sign} {fmt(rule.amber)}" if rule.amber is not None else "",
    ]
    return " · ".join(p for p in parts if p)


def tile(col, label: str, value: str, name: str, fmt, ind: dict, cfg, help_text: str | None = None) -> None:
    """A metric with a colour dot (from the threshold named `name`) and the rule it is judged by."""
    with col:
        st.metric(f"{DOT[ind['color']]} {label}", value, help=help_text)
        rule = rule_caption(cfg.thresholds.get(name), fmt)
        few = ind["color"] == m.GREY and ind["value"] is not None
        st.caption(
            (rule + " · " if rule else "")
            + (f"only {ind['n']} requests, no colour yet" if few else f"n={ind['n']}")
        )


def ratio_tile(col, label, ratio, name, ind, cfg, help_text=None):
    tile(col, label, fmt_pct(ratio.value), name, fmt_pct, ind, cfg, help_text)


def bar(data: dict, label: str = "count") -> None:
    if not data:
        st.caption("Nothing to show yet.")
        return
    st.bar_chart(pd.DataFrame({label: data}), y=label, horizontal=True)


def latency_table(latency: dict) -> pd.DataFrame:
    rows = [
        {
            "stage": stage,
            "n": s["n"],
            "p50": fmt_ms(s["p50"]),
            "p95": fmt_ms(s["p95"]),
            "p99": fmt_ms(s["p99"]),
        }
        for stage, s in latency.items()
    ]
    return pd.DataFrame(rows).set_index("stage")


st.title("📊 Metrics")
cfg = config.get_settings().observability
db_path = config.get_settings().sqlite_path

top = st.columns([2, 1, 4])
range_label = top[0].selectbox("Time range", list(m.RANGES), index=2)
if top[1].button("↻ Refresh", help="Reload from the request log"):
    st.rerun()
since = m.since_iso(range_label)

rows = m.load_requests(db_path, since)
documents = m.load_documents(db_path, since)
rejections = m.load_upload_failures(db_path, since)

if not rows and not documents and not rejections:
    st.info("No requests logged yet in this range. Ask a question on the Ask page and come back.")
    st.stop()

d = m.build_dashboard(rows, documents, rejections, cfg)
ind = d["indicators"]
top[2].caption(f"{d['n_requests']} requests · reading `{db_path.name}` directly")

# ---------------------------------------------------------------- operational
st.header("Operational")
lat = d["operational"]["latency_ms"]
c = st.columns(4)
tile(c[0], "p95 total latency", fmt_ms(lat["total"]["p95"]), "p95_total_ms", fmt_ms, ind["p95_total_ms"], cfg)
ratio_tile(
    c[1],
    "Error rate",
    d["operational"]["error_rate"],
    "error_rate",
    ind["error_rate"],
    cfg,
    "Requests where a path failed (LLM unavailable or timed out after retries, or an internal error).",
)
c[2].metric("Tokens / request", fmt_num(d["operational"]["tokens_per_request"]))
c[3].metric(
    "Cost-equiv. / request",
    fmt_usd(d["operational"]["cost_per_request_usd"]),
    help="List-price equivalent (we use a free tier). Rates: observability.pricing_per_mtok in config.yaml.",
)
st.caption(
    "Latency per stage, over the requests that ran that stage. "
    f"Total cost-equivalent in range: {fmt_usd(d['operational']['cost_total_usd'])}."
)
st.dataframe(latency_table(lat))

_fb, _rl = d["operational"]["llm_fallback_rate"], d["operational"]["llm_rate_limited_rate"]
if _fb.total:
    st.caption(
        f"LLM backend: {_rl.count} of {_rl.total} requests saw a rate-limit (429) response; "
        f"{_fb.count} had a call fall back to Ollama."
    )

ing = d["operational"]["ingestion"]
if ing["documents"]:
    st.caption(
        f"Ingestion: median {fmt_num(ing['median_s_per_page'], 2)} s/page, "
        f"pooled {fmt_num(ing['pooled_s_per_page'], 2)} s/page "
        f"over {len(ing['documents'])} ready document(s)."
    )
    st.dataframe(pd.DataFrame(ing["documents"]).set_index("filename"))
else:
    st.caption("Ingestion: no finished documents in this range.")

# ---------------------------------------------------------------- input
st.header("Input")
inp = d["input"]
left, mid, right = st.columns(3)
with left:
    st.subheader("Route mix")
    bar(inp["route_mix"])
with mid:
    st.subheader("Question length")
    ql = inp["question_len"]
    st.caption(f"median {fmt_num(ql['p50'])} · p95 {fmt_num(ql['p95'])} characters")
    hist = pd.DataFrame(inp["question_len_histogram"]).set_index("bucket")
    st.bar_chart(hist, y="count")
with right:
    st.subheader("Upload failures")
    bar(inp["upload_failures"])

# ---------------------------------------------------------------- output
st.header("Output")
out = d["output"]
c = st.columns(4)
ratio_tile(
    c[0],
    "Abstention rate",
    out["abstention_rate"],
    "abstention_rate",
    ind["abstention_rate"],
    cfg,
    "Share of requests that answered 'not in your documents' (or were not ready).",
)
ratio_tile(
    c[1],
    "Citation-invalid rate",
    out["citation_invalid_rate"],
    "citation_invalid_rate",
    ind["citation_invalid_rate"],
    cfg,
    "Document answers where the model cited a source we never sent.",
)
ratio_tile(
    c[2],
    "Number-check failures",
    out["number_check_fail_rate"],
    "number_check_fail_rate",
    ind["number_check_fail_rate"],
    cfg,
    "Answers with a figure not found verbatim in the cited text (a correct sum also trips it).",
)
ratio_tile(
    c[3],
    "Router fallback rate",
    out["router_fallback_rate"],
    "router_fallback_rate",
    ind["router_fallback_rate"],
    cfg,
    "Router output was unusable (or the LLM was down), so the question went to the document path.",
)
left, right = st.columns(2)
with left:
    st.caption("Why it abstained")
    bar(out["abstain_reasons"])
with right:
    ac = out["answer_chars"]
    st.caption("Answer length (answered requests)")
    st.write(f"median **{fmt_num(ac['p50'])}** · p95 **{fmt_num(ac['p95'])}** characters (n={ac['n']})")
    if out["router_fallback_reasons"]:
        st.caption("Router fallback reasons")
        bar(out["router_fallback_reasons"])

# ---------------------------------------------------------------- quality
st.header("Quality")
q = d["quality"]
c = st.columns(4)
c[0].metric("👍 / 👎", f"{q['thumbs_up']} / {q['thumbs_down']}")
ratio_tile(
    c[1],
    "Thumbs-down share",
    q["thumbs_down_share"],
    "thumbs_down_share",
    ind["thumbs_down_share"],
    cfg,
    "👎 as a share of all ratings. Few people rate, so read it with the count.",
)
c[2].metric("Answers rated", fmt_pct(q["rated_share_of_answers"].value))
if q["judge_correct"].total or q["judge_grounded"].total:
    c[3].metric(
        "Judge: correct / grounded",
        f"{fmt_pct(q['judge_correct'].value)} / {fmt_pct(q['judge_grounded'].value)}",
        help=f"Judge scores so far: {q['judge_correct'].total} correctness, "
        f"{q['judge_grounded'].total} groundedness.",
    )
else:
    c[3].metric(
        "Judge scores",
        "–",
        help="Written by the judge script (step 09) into requests.judge_correct / judge_grounded.",
    )
    c[3].caption("No judged answers yet.")

# ---------------------------------------------------------------- drift
st.header("Drift")
dr = d["drift"]
left, right = st.columns(2)
with left:
    st.subheader("Median top retrieval score, per day")
    st.caption("Falling = questions are moving away from the documents.")
    daily = dr["daily_top_score"]
    latest = ind["median_top_score"]
    st.write(
        f"{DOT[latest['color']]} latest day: **{fmt_num(latest['value'], 3)}** "
        f"({rule_caption(cfg.thresholds.get('median_top_score'), lambda v: f'{v:g}')}, n={latest['n']})"
    )
    if daily:
        st.line_chart(pd.DataFrame(daily).set_index("date"), y="median_top_score")
    else:
        st.caption("No document-path requests yet.")
with right:
    st.subheader("Route mix, week over week")
    shift = dr["route_mix_shift"]
    s_ind = ind["route_mix_shift"]
    if shift:
        st.write(
            f"{DOT[s_ind['color']]} shift vs previous week: **{fmt_pct(shift['shift'])}** "
            f"({rule_caption(cfg.thresholds.get('route_mix_shift'), fmt_pct)})"
        )
    else:
        st.caption("Needs requests in two consecutive weeks.")
    weeks = dr["route_mix_by_week"]
    if weeks:
        st.dataframe(pd.DataFrame(weeks).set_index("week").fillna(0))
