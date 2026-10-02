"""Hand-made eval records (the shape in eval/records.py) so the aggregator can be tested without services."""

from __future__ import annotations

from typing import Any

from eval.schema import EvalRow


def rec(
    id: str = "D001",
    *,
    route: str = "DOCUMENT",
    slice: str = "table",
    answerable: bool | None = True,
    type: str = "numeric",
    gold_answer: str | None = "Rs 100.50 million (standalone, FY2025)",
    general_answer: str | None = None,
    pages: tuple[int, ...] = (56,),
    llm_route: str | None = None,
    keyword: str | None = None,
    router_ok: bool = True,
    top_score: float | None = 0.7,
    strict_rank: int | None = 1,
    pm1_rank: int | None = None,
    status: str | None = "answered",  # document section status; None = no document section
    answer: str = "Profit was 100.50 million.",
    abstain_reason: str | None = None,
    cited_pages: tuple[int, ...] = (56,),
    number_check: str = "pass",
    general_status: str | None = None,  # general section status; None = no general section
    general_text: str = "Paris.",
    judge: dict | None = None,
    live: int = 2,
    cached: int = 0,
    timings: dict | None = None,
    infra: bool = False,
    mode: str = "full",
    tokens: int = 3000,
    cost: float = 0.0005,
) -> dict[str, Any]:
    """`judge` = {"document": (correct, grounded), "general": (correct, None)} -> verdict dicts."""
    pred = llm_route or route
    sections: dict[str, dict] = {}
    if status is not None:
        sections["document"] = {
            "status": status,
            "answer": answer if status == "answered" else "I couldn't find this.",
            "question": f"question {id}",
            "abstain_reason": abstain_reason,
            "number_check": number_check,
            "top_score": top_score,
            "citations": [
                {
                    "doc": "report_a",
                    "doc_id": "d1",
                    "pdf_page": p,
                    "page_label": str(p),
                    "chunk_id": f"d1:{p}:0",
                }
                for p in cited_pages
            ]
            if status == "answered"
            else [],
            "sources": [
                {
                    "chunk_id": f"d1:{p}:0",
                    "pdf_page": p,
                    "page_label": str(p),
                    "text": f"source text of page {p}",
                }
                for p in cited_pages
            ]
            if status == "answered"
            else [],
        }
    if general_status is not None:
        sections["general"] = {
            "status": general_status,
            "answer": general_text,
            "question": f"general question {id}",
            "abstain_reason": None,
        }
    verdicts = {}
    for part, (correct, grounded) in (judge or {}).items():
        verdicts[part] = {"correct": correct, "grounded": grounded, "reason": "", "tokens": 100}
    out: dict[str, Any] = {
        "id": id,
        "row_hash": "h-" + id,
        "mode": mode,
        "question": f"question {id}",
        "gold": {
            "route": route,
            "slice": slice,
            "answerable": answerable,
            "type": type,
            "answer": gold_answer,
            "general_answer": general_answer,
            "pages": [{"doc": "report_a", "pdf_page": p, "label": str(p)} for p in pages],
        },
        "retrieval": {"top_score": top_score},
        "router": {"llm": pred, "ok": router_ok, "fallback_reason": None, "keyword": keyword or route},
        "sections": sections,
        "timings": timings
        if timings is not None
        else {"router_ms": 600.0, "total_ms": 3000.0, "document_ms": 2000.0, "document_llm_ms": 1500.0},
        "tokens": {
            "router": {"prompt": 1000, "completion": 100, "total": 1100},
            "document": {"prompt": 1700, "completion": 200, "total": 1900},
            "general": {"prompt": 0, "completion": 0, "total": 0},
            "total": {"prompt": 2700, "completion": 300, "total": tokens},
        },
        "cost_usd": cost,
        "calls": {"live": live, "cached": cached},
        "infra_error": infra,
    }
    if route == "DOCUMENT" and answerable:
        out["retrieval"].update({"strict_rank": strict_rank, "pm1_rank": pm1_rank or strict_rank})
    if verdicts:
        out["judge"] = {"model": "judge-m", "prompt_version": "v1", "verdicts": verdicts}
    return out


def row(
    id: str = "D001",
    *,
    route: str = "DOCUMENT",
    slice: str = "table",
    answerable: bool | None = True,
    question: str = "What was profit?",
    gold_answer: str | None = "Rs 100.50 million (standalone, FY2025)",
    pages: tuple[int, ...] = (1,),
    type: str = "numeric",
    **extra,
) -> EvalRow:
    data: dict[str, Any] = {
        "id": id,
        "question": question,
        "route": route,
        "slice": slice,
        "type": type,
        "answerable": answerable if route == "DOCUMENT" else (True if route == "MIXED" else None),
    }
    if gold_answer is not None:
        data["gold_answer"] = gold_answer
    if pages and answerable is not False and route != "GENERAL":
        data["gold_pages"] = [{"doc": "report_a", "page": p, "pdf_page": p} for p in pages]
    data.update(extra)
    return EvalRow(**data)
