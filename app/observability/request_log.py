"""Turn one finished /query into a `requests` row (design §15).

Pure functions, so they are testable without an app. Nothing here stores question or document text: the row
has the question's length, chunk ids of the citations, numbers, and short codes.

Choices worth knowing:
- `status` is the worst of the sections that ran (error > not_ready > abstained > answered), so a MIXED
  question whose general half failed counts as an error even though the document half answered.
- A stage that did not run (no router call, gate 1 stopped before the LLM, no document path) is stored as
  NULL, not 0, so percentiles are over requests that actually ran the stage.
- `t_llm_ms` is the slowest answer-LLM call: in a MIXED question both run at once.
- `cost_usd_equiv` prices each part's tokens at the rates in `observability.pricing_per_mtok`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.config import ObservabilityConfig, Rate

if TYPE_CHECKING:  # the storage layer imports this module for LOG_COLUMNS; keep it free of the answer code
    from app.answering.document import DocAnswer
    from app.answering.general import GeneralAnswer

ANSWERED, ABSTAINED, NOT_READY, ERROR = "answered", "abstained", "not_ready", "error"
_RANK = {ANSWERED: 0, ABSTAINED: 1, NOT_READY: 2, ERROR: 3}

# Columns the logger may write. `feedback` is deliberately absent: only /feedback touches it.
LOG_COLUMNS = (
    "trace_id", "ts", "question_len", "route", "router_ok", "router_fallback", "prompt_versions", "model_ids",
    "t_router_ms", "t_embed_ms", "t_retrieve_ms", "t_llm_ms", "t_total_ms", "top_score", "n_sources",
    "status", "abstain_reason", "citations_valid", "number_check", "tokens_in", "tokens_out",
    "cost_usd_equiv", "error", "answer_chars", "cited_chunks", "app_version",
    "llm_backend", "llm_fallbacks", "llm_rate_limited",
)  # fmt: skip


def overall_status(statuses: list[str]) -> str:
    """Worst status wins. An empty list (nothing ran) is an error."""
    known = [s if s in _RANK else ERROR for s in statuses]  # a status we do not know is not "fine"
    return max(known, key=_RANK.__getitem__) if known else ERROR


def _ms(value: float | None) -> float | None:
    """0 / missing means the stage did not run."""
    return round(float(value), 1) if value else None


def cost_usd(cfg: ObservabilityConfig, parts: list[tuple[str | None, int, int]]) -> float:
    """`parts` = (model id, prompt tokens, completion tokens). Unknown models use the "default" rate."""
    total = 0.0
    for model, prompt, completion in parts:
        rate: Rate | None = cfg.pricing_per_mtok.get(model or "") or cfg.pricing_per_mtok.get("default")
        if rate is None:
            continue
        total += (prompt * rate.input + completion * rate.output) / 1_000_000
    return round(total, 6)


def _json_or_none(d: dict[str, Any]) -> str | None:
    d = {k: v for k, v in d.items() if v}
    return json.dumps(d, sort_keys=True) if d else None


def _flag(v: bool | None) -> int | None:
    return None if v is None else int(bool(v))


def build_record(
    response: dict,
    *,
    question_len: int,
    doc: DocAnswer | None,
    gen: GeneralAnswer | None,
    obs: ObservabilityConfig,
    app_version: str,
) -> dict[str, Any]:
    """`response` is the dict `QueryService.run` returns; `doc` / `gen` are the raw path results (or None)."""
    sections = response["sections"]
    router = response["router"]
    timings = response["timings"]
    tokens = response["tokens"]
    doc_sec = next((s for s in sections if s["kind"] == "document"), None)
    llm = response.get("llm") or {}

    llm_ms = max(
        timings.get("document_llm_ms") or 0.0,
        (gen.timings.get("llm_ms") or 0.0) if gen else 0.0,
    )
    models = {
        "router": None if router["skipped"] else router["model"],
        "document": doc.model if doc else None,
        "general": gen.model if gen else None,
    }
    versions = {
        "router": None if router["skipped"] else router["prompt_version"],
        "document": doc.prompt_version if doc else None,
        "general": gen.prompt_version if gen else None,
    }
    errors = [s["abstain_reason"] or "error" for s in sections if s["status"] == ERROR]
    reasons = [s["abstain_reason"] for s in sections if s["abstain_reason"]]
    answered_text = [s["answer"] for s in sections if s["status"] == ANSWERED]

    return {
        "trace_id": response["trace_id"],
        "question_len": question_len,
        "route": response["route"],
        "router_ok": int(bool(response["router_ok"])),
        "router_fallback": router.get("fallback_reason"),
        "prompt_versions": _json_or_none(versions),
        "model_ids": _json_or_none(models),
        "t_router_ms": None if router["skipped"] else _ms(timings.get("router_ms")),
        "t_embed_ms": _ms(timings.get("embed_ms")),
        "t_retrieve_ms": _ms(timings.get("retrieve_ms")),
        "t_llm_ms": _ms(llm_ms),
        "t_total_ms": _ms(timings.get("total_ms")),
        "top_score": doc_sec["top_score"] if doc_sec else None,
        "n_sources": doc.n_sources if doc else None,
        "status": overall_status([s["status"] for s in sections]),
        "abstain_reason": reasons[0] if reasons else None,
        "citations_valid": _flag(doc.citations_valid) if doc else None,
        "number_check": doc_sec["number_check"] if doc_sec else None,
        "tokens_in": tokens["total"]["prompt"],
        "tokens_out": tokens["total"]["completion"],
        "cost_usd_equiv": cost_usd(
            obs,
            [
                (models["router"], tokens["router"]["prompt"], tokens["router"]["completion"]),
                (models["document"], tokens["document"]["prompt"], tokens["document"]["completion"]),
                (models["general"], tokens["general"]["prompt"], tokens["general"]["completion"]),
            ],
        ),
        "error": ", ".join(errors) if errors else None,
        "answer_chars": sum(len(t) for t in answered_text),
        "cited_chunks": json.dumps([c["chunk_id"] for c in doc_sec["citations"]]) if doc_sec else None,
        "app_version": app_version,
        # Backend(s) that served the LLM calls ("groq", "ollama", "groq+ollama"), calls that fell back to
        # Ollama, and 429s seen on the way. NULL = no LLM call was made.
        "llm_backend": "+".join(llm["backends"]) if llm.get("backends") else None,
        "llm_fallbacks": llm.get("fallback_calls") if llm.get("calls") else None,
        "llm_rate_limited": llm.get("rate_limited") if llm.get("calls") else None,
    }


def build_content_sections(response: dict, doc: DocAnswer | GeneralAnswer | None) -> list[dict[str, Any]]:
    """The text side of a request, for `request_content` (opt-in, see `observability.log_content`).

    One entry per section that ran. Only an answered section keeps its answer (an abstention message is
    boilerplate); the document section also keeps the full text of the chunks it cited."""
    out: list[dict[str, Any]] = []
    for s in response["sections"]:
        answered = s["status"] == ANSWERED
        entry: dict[str, Any] = {
            "kind": s["kind"],
            "status": s["status"],
            "question": s["question"],
            "answer": s["answer"] if answered else None,
            "sources": [],
        }
        if answered and s["kind"] == "document" and doc is not None:
            cited = getattr(doc, "cited_texts", {})
            entry["sources"] = [
                {
                    "chunk_id": c["chunk_id"],
                    "page_label": c["page_label"],
                    "pdf_page": c["pdf_page"],
                    "text": cited.get(c["chunk_id"], c["snippet"]),
                }
                for c in s["citations"]
            ]
        out.append(entry)
    return out


def error_record(
    trace_id: str, *, question_len: int, error: str, total_ms: float, app_version: str
) -> dict[str, Any]:
    """The request blew up before a response existed (a bug, not an LLM outage). `error` is a short code."""
    return {
        "trace_id": trace_id,
        "question_len": question_len,
        "status": ERROR,
        "error": error[:200],
        "t_total_ms": _ms(total_ms),
        "app_version": app_version,
    }
