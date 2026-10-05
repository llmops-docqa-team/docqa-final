"""Question router (design §9): one JSON-mode call to the small model decides DOCUMENT / GENERAL / MIXED and
splits a MIXED question into its two parts.

Fail-safe: unusable JSON twice (or an LLM error) -> treat the whole question as DOCUMENT and say so
(`router_ok=False`, a logged warning). The worst case is an abstention, never an invented document fact.
No documents at all -> no LLM call, everything is GENERAL.
Nothing here logs question text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from app.config import Settings
from app.llm.client import LLMClient, LLMError, Usage
from app.llm.json_reply import load_json_object
from app.llm.prompts import Prompt, load_prompt
from app.observability.logging import get_logger
from app.observability.tracing import get_tracer, usage_details
from app.storage import documents as st

DOCUMENT, GENERAL, MIXED = "DOCUMENT", "GENERAL", "MIXED"

RETRY_NUDGE = (
    "Your reply was not a valid JSON object of the required shape. Reply again with only "
    '{"route": "DOCUMENT" or "GENERAL" or "MIXED", "document_question": "...", "general_question": "..."}. '
    "For MIXED both questions must be non-empty."
)

_CONTROL = re.compile(r"[\x00-\x1f\x7f<>]")


class RouterDecision(BaseModel):
    """The router model's reply. MIXED needs both sub-questions; the other routes ignore them."""

    route: Literal["DOCUMENT", "GENERAL", "MIXED"]
    document_question: str = ""
    general_question: str = ""

    @field_validator("route", mode="before")
    @classmethod
    def _norm_route(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("document_question", "general_question", mode="before")
    @classmethod
    def _norm_question(cls, v: Any) -> Any:
        return "" if v is None else v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _mixed_needs_both(self) -> RouterDecision:
        if self.route == MIXED and not (self.document_question and self.general_question):
            raise ValueError("route MIXED needs a document_question and a general_question")
        return self


@dataclass
class RouteOutcome:
    route: str  # DOCUMENT | GENERAL | MIXED
    document_question: str  # what the document path should answer ("" when it does not run)
    general_question: str  # what the general path should answer ("" when it does not run)
    router_ok: bool = True  # False = fell back to DOCUMENT
    skipped: bool = False  # no documents uploaded: no LLM call was made
    fallback_reason: str | None = None  # bad_json | llm_unavailable
    model: str | None = None
    prompt_version: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    llm_calls: int = 0


def parse_router_json(content: str) -> RouterDecision:
    """Validate the router's reply. Raises ValueError (pydantic's ValidationError is one)."""
    return RouterDecision.model_validate(load_json_object(content))


def _doc_label(doc: dict[str, Any], title_chars: int) -> str:
    name = " ".join(_CONTROL.sub(" ", str(doc.get("filename") or "(unnamed)")).split())
    if len(name) > title_chars:
        name = name[: title_chars - 1].rstrip() + "…"
    status = doc.get("status")
    if status == st.READY:
        state = "READY"
    elif status == st.FAILED:
        state = "failed"
    elif status == st.PARTIAL or status == st.PROCESSING:
        total = doc.get("pages_total")
        state = f"processing {doc.get('pages_done') or 0}/{total if total else '?'} pages"
    else:
        state = "queued"
    return f"{name} ({state})"


def format_documents(docs: Sequence[dict[str, Any]], max_docs: int, title_chars: int) -> str:
    """One bullet per document: file name and state. Capped so a big library cannot blow up the prompt."""
    lines = [f"- {_doc_label(d, title_chars)}" for d in docs[:max_docs]]
    if len(docs) > max_docs:
        lines.append(f"- ... and {len(docs) - max_docs} more documents")
    return "\n".join(lines)


class Router:
    def __init__(self, llm: LLMClient, settings: Settings, prompt: Prompt | None = None):
        self.llm = llm
        self.settings = settings
        self.prompt = prompt or load_prompt(settings.prompts.router)

    def route(self, question: str, docs: Sequence[dict[str, Any]]) -> RouteOutcome:
        """`docs` are the rows of the `documents` table (any status)."""
        base = dict(prompt_version=self.prompt.version, model=self.settings.llm.router_model)
        if not docs:
            return RouteOutcome(GENERAL, "", question, skipped=True, **base)

        cfg = self.settings
        listing = format_documents(docs, cfg.query.router_max_docs, cfg.query.doc_title_chars)
        messages = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": self.prompt.render_user(documents=listing, question=question)},
        ]
        usage, latency_ms, calls = Usage(), 0.0, 0
        decision: RouterDecision | None = None
        model = cfg.llm.router_model
        backend = ""
        with get_tracer().generation(
            "router",
            model=model,
            input=question,
            metadata={
                "prompt": cfg.prompts.router,
                "prompt_version": self.prompt.version,
                "n_documents": len(docs),
            },
        ) as gen:
            try:
                for attempt in range(2):  # the second attempt is the one retry on bad JSON
                    resp = self.llm.chat(
                        messages,
                        model=cfg.llm.router_model,
                        temperature=0.0,
                        max_tokens=cfg.query.router_max_tokens,
                        reasoning_effort=cfg.llm.reasoning_effort,
                        json_mode=True,
                    )
                    usage.add(resp.usage)
                    latency_ms += resp.latency_ms
                    calls += 1
                    model = resp.model or model
                    backend = resp.backend or backend
                    try:
                        decision = parse_router_json(resp.content)
                        break
                    except ValueError:
                        if attempt == 0:
                            messages = messages + [
                                {"role": "assistant", "content": resp.content or "(empty)"},
                                {"role": "user", "content": RETRY_NUDGE},
                            ]
            except LLMError as exc:
                gen.update(level="ERROR", status_message=f"llm_unavailable (HTTP {exc.status})")
                stats = self._stats(model, usage, latency_ms, calls)
                return self._fallback("llm_unavailable", question, stats, str(exc))
            finally:
                gen.update(
                    model=model,
                    usage_details=usage_details(
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                    ),
                    metadata={"llm_calls": calls, "backend": backend},
                )
            gen.update(
                output=decision.route if decision else None,
                metadata={"route": decision.route if decision else None, "router_ok": decision is not None},
            )

        stats = self._stats(model, usage, latency_ms, calls)
        if decision is None:
            return self._fallback("bad_json", question, stats, "unusable JSON twice")
        if decision.route == MIXED:
            return RouteOutcome(MIXED, decision.document_question, decision.general_question, **stats)
        # A single route always answers the user's own words, not the model's rewrite of them.
        if decision.route == DOCUMENT:
            return RouteOutcome(DOCUMENT, question, "", **stats)
        return RouteOutcome(GENERAL, "", question, **stats)

    def _stats(self, model: str, usage: Usage, latency_ms: float, calls: int) -> dict[str, Any]:
        return dict(
            prompt_version=self.prompt.version,
            model=model,
            usage=usage,
            latency_ms=latency_ms,
            llm_calls=calls,
        )

    @staticmethod
    def _fallback(reason: str, question: str, stats: dict[str, Any], detail: str) -> RouteOutcome:
        log = get_logger()
        log.warning("router_fallback", reason=reason, detail=detail[:200], llm_calls=stats["llm_calls"])
        return RouteOutcome(DOCUMENT, question, "", router_ok=False, fallback_reason=reason, **stats)
