"""The query pipeline behind POST /query: route -> run the document and/or general path -> compose.

Every response has the same shape, so the UI needs no per-route logic:

    {trace_id, route, router_ok, router{...}, sections[...], timings{...}, tokens{...}}

`sections` holds one entry per path that ran, document first. Each section has the same keys
(`kind` document|general, `status`, `answer` = the text to show, `citations`, `label`, `abstain_reason`,
`hint`, `closest_pages`, `fallback_passages`, `coverage_note`, `warnings`, ...). For a MIXED question the two
parts run in parallel and fail independently: the document part may abstain while the general part answers.
A path that raises unexpectedly becomes an `error` section; the request itself still succeeds.
Every request is also written to the SQLite `requests` table (step 08); a failure there is logged and never
fails the request. Nothing here logs question or answer text.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from app.answering import document as doc_mod
from app.answering.document import DocAnswer, DocumentAnswerer
from app.answering.general import GeneralAnswer, GeneralAnswerer
from app.config import Settings
from app.llm import request_stats
from app.observability.logging import get_logger
from app.observability.request_log import build_content_sections, build_record, error_record
from app.observability.tracing import get_tracer
from app.observability.version import get_app_version
from app.routing.enhancer import Enhancement, company_periods, enhance, example_question
from app.routing.rewriter import QueryRewriter
from app.routing.router import DOCUMENT, GENERAL, MIXED, RouteOutcome, Router
from app.storage import documents as st
from app.storage.documents import DocumentStore
from app.storage.requests import RequestStore

SPECIFIC_HINT = "Try asking about something more specific: name the metric, the period and the document."
NO_DOCS_NOTICE = (
    "No documents are uploaded yet, so this answer is general knowledge. Upload a PDF to ask about it."
)
INTERNAL_ERROR = "Something went wrong while answering this part. Please try again."
NEEDS_COMPANY = "needs_company"
UNCLEAR = "unclear"
# Abstentions that mean "the evidence was weak", where a more specific question may help.
_HINT_REASONS = {
    doc_mod.LOW_SCORE,
    doc_mod.INSUFFICIENT,
    doc_mod.NO_RESULTS,
    doc_mod.NO_VALID_CITATIONS,
}


@dataclass
class Section:
    kind: str  # document | general
    status: str  # document: answered|abstained|not_ready|error; general: answered|error
    answer: str  # the text to show (the answer, the abstention message, or the error message)
    question: str = ""  # the (sub-)question this section answered
    citations: list[dict] = field(default_factory=list)
    label: str | None = None  # general sections: always "General knowledge — ..."
    abstain_reason: str | None = None
    hint: str | None = None  # "be more specific" on weak-evidence abstentions
    closest_pages: list[dict] = field(default_factory=list)
    fallback_passages: list[dict] = field(default_factory=list)  # document path when the LLM is down
    coverage_note: str | None = None
    warnings: list[str] = field(default_factory=list)
    number_check: str | None = None  # document sections: pass | fail | na
    computed_numbers: list[str] = field(default_factory=list)  # figures derived from cited ones
    top_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _PathResult:
    section: Section
    timings: dict[str, float]
    tokens: dict[str, int]
    wall_ms: float
    answer: DocAnswer | GeneralAnswer | None = None  # the raw result, for the request log


def _zero_tokens() -> dict[str, int]:
    return {"prompt": 0, "completion": 0, "total": 0}


def _document_section(question: str, da: DocAnswer, docs: list[dict[str, Any]]) -> Section:
    warnings: list[str] = []
    if da.status == doc_mod.ANSWERED:
        # An answer is shown: say what it did not have access to.
        warnings += da.notices
        searched = set(da.searched)
        for d in docs:
            if d["status"] == st.PARTIAL and d["filename"] in searched:
                warnings.append(
                    f"{d['filename']} is still processing ({d.get('pages_done') or 0}/"
                    f"{d.get('pages_total') or '?'} pages): this answer only uses the pages indexed so far "
                    "and may be incomplete or wrong. Ask again when the document is Ready."
                )
        if da.number_warning:
            warnings.append(
                "⚠ number not found verbatim in source: " + ", ".join(da.unmatched_numbers)
                if da.unmatched_numbers
                else "⚠ number not found verbatim in source"
            )
        if da.dropped_citations:
            warnings.append("Some citations did not match the retrieved sources and were dropped.")
    return Section(
        kind="document",
        status=da.status,
        answer=da.message,
        question=question,
        citations=[asdict(c) for c in da.citations],
        abstain_reason=da.abstain_reason,
        hint=SPECIFIC_HINT if da.abstain_reason in _HINT_REASONS else None,
        closest_pages=[asdict(c) for c in da.closest_pages],
        fallback_passages=[asdict(c) for c in da.fallback_passages],
        coverage_note=da.coverage_note,
        warnings=warnings,
        number_check=da.number_check,
        computed_numbers=da.computed_numbers,
        top_score=da.top_score,
    )


def _general_section(question: str, ga: GeneralAnswer, *, no_documents: bool) -> Section:
    return Section(
        kind="general",
        status=ga.status,
        answer=ga.message,
        question=question,
        label=ga.label,
        abstain_reason=ga.error_reason,
        warnings=[NO_DOCS_NOTICE] if no_documents else [],
    )


class QueryService:
    def __init__(
        self,
        store: DocumentStore,
        router: Router,
        doc_answerer: DocumentAnswerer,
        general_answerer: GeneralAnswerer,
        settings: Settings,
        request_store: RequestStore | None = None,
        rewriter: QueryRewriter | None = None,
    ):
        self.rewriter = rewriter  # None = no LLM rewrite (the rule-based enhancer still runs)
        self.request_store = request_store  # None = do not write the request log (unit tests)
        self.store = store
        self.router = router
        self.doc_answerer = doc_answerer
        self.general_answerer = general_answerer
        self.settings = settings

    def run(self, question: str, trace_id: str, company: str | None = None, enhance: bool = True) -> dict:
        return self.run_detailed(question, trace_id, company=company, enhance=enhance)[0]

    def run_detailed(
        self, question: str, trace_id: str, company: str | None = None, enhance: bool = True
    ) -> tuple[dict, dict[str, _PathResult]]:
        """`run`, plus the raw per-path results (`results["document"].answer` is the `DocAnswer`). The eval
        runner uses this to read the cited chunk text; `/query` only needs the response."""
        t_start = time.perf_counter()
        with get_tracer().trace(
            "query", trace_id, input=question, metadata={"question_len": len(question)}
        ) as root:
            try:
                response, results = self._run(question, trace_id, t_start, company, enhance)
            except Exception as exc:
                error = type(exc).__name__  # the code only; the message could echo user text
                root.update(level="ERROR", status_message=error)
                self._write_log(
                    lambda: error_record(
                        trace_id,
                        question_len=len(question),
                        error=error,
                        total_ms=(time.perf_counter() - t_start) * 1000,
                        app_version=get_app_version(),
                    )
                )
                raise
            root.update(output=self._trace_output(response), metadata=self._trace_summary(response))
        doc, gen = results.get("document"), results.get("general")
        self._write_log(
            lambda: build_record(
                response,
                question_len=len(question),
                doc=doc.answer if doc else None,
                gen=gen.answer if gen else None,
                obs=self.settings.observability,
                app_version=get_app_version(),
            )
        )
        if self.settings.observability.log_content:
            self._write_log_content(question, response, doc.answer if doc else None)
        return response, results

    def _write_log_content(
        self, question: str, response: dict, doc: DocAnswer | GeneralAnswer | None
    ) -> None:
        if self.request_store is None:
            return
        try:
            self.request_store.log_content(
                response["trace_id"], question, build_content_sections(response, doc)
            )
        except Exception as exc:  # noqa: BLE001  same rule as the request log: never fail a question
            get_logger().warning(
                "request_content_failed", error_type=type(exc).__name__, detail=str(exc)[:200]
            )

    def _write_log(self, make_record: Callable[[], dict]) -> None:
        """Write the request-log row. Never raises: a broken log must not fail a question."""
        if self.request_store is None:
            return
        try:
            self.request_store.log(make_record())
        except Exception as exc:  # noqa: BLE001
            get_logger().warning("request_log_failed", error_type=type(exc).__name__, detail=str(exc)[:200])

    def _run(
        self,
        question: str,
        trace_id: str,
        t_start: float,
        company: str | None = None,
        use_enhancer: bool = True,
    ) -> tuple[dict, dict[str, _PathResult]]:
        with request_stats.track() as llm_stats:
            docs = self.store.list()
            no_docs = not docs
            # Query enhancer first, for every question: the router and both paths see the cleaned-up wording.
            pre = self._enhance(question, docs, company, use_enhancer)
            enhanced: Enhancement | None = pre
            jobs: dict[str, Callable[[], _PathResult]] = {}
            if pre is not None and pre.unclear:
                outcome = RouteOutcome(
                    DOCUMENT, question, "", skipped=True, prompt_version=self.router.prompt.version
                )
                jobs["document"] = lambda: self._unclear(question, docs, company)
            else:
                outcome = self.router.route(pre.question if pre else question, docs)
                if outcome.route in (DOCUMENT, MIXED):
                    enhanced = self._scope(outcome.document_question, docs, company, pre, use_enhancer)
                    jobs["document"] = lambda: self._document_path(outcome.document_question, docs, enhanced)
                if outcome.route in (GENERAL, MIXED):
                    jobs["general"] = lambda: self._general_path(outcome.general_question, no_docs)

            results = self._run_jobs(jobs)
        sections = [results[k].section for k in ("document", "general") if k in results]  # document first
        response = {
            "trace_id": trace_id,
            "route": outcome.route,
            "router_ok": outcome.router_ok,
            "router": self._router_meta(outcome),
            "sections": [s.to_dict() for s in sections],
            "timings": self._timings(outcome, results, t_start),
            "tokens": self._tokens(outcome, results, enhanced),
            "llm": llm_stats.summary(),  # backends that served the calls, 429s seen, calls that fell back
            "enhancer": enhanced.to_dict() if enhanced else None,
        }
        self._log(response)
        return response, results

    # -- paths

    @staticmethod
    def _run_jobs(jobs: dict[str, Callable[[], _PathResult]]) -> dict[str, _PathResult]:
        if len(jobs) <= 1:
            return {name: fn() for name, fn in jobs.items()}
        # MIXED: both paths at once. Each gets its own copy of the context so request-id logging follows.
        with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="query") as pool:
            futures = {
                name: pool.submit(contextvars.copy_context().run, fn) for name, fn in jobs.items()
            }
            return {name: f.result() for name, f in futures.items()}

    def _enhance(
        self, question: str, docs: list[dict[str, Any]], company: str | None, use_enhancer: bool = True
    ) -> Enhancement | None:
        """Query enhancer, before routing: resolve the company (picked, named, or the only one loaded) and
        rewrite the wording with one small LLM call (spelling, short forms, company named). Text with no
        question in it is flagged `unclear`. A failure here only loses the boost: the question goes on as
        asked."""
        cfg = self.settings.query
        if not cfg.enhance or not docs:
            return None
        try:
            if not use_enhancer:  # the user switched it off: no rewrite call, no added words
                return enhance(question, docs, company, plain=True)
            first = enhance(question, docs, company)
            if self.rewriter is None or not cfg.rewrite:
                return first
            pinned = first.companies[0] if len(first.companies) == 1 else None
            if pinned is None and len(first.available_companies) == 1:
                pinned = first.available_companies[0]
            rw = self.rewriter.rewrite(question, pinned, company_periods(docs, pinned))
            first.question, first.rewritten, first.unclear = rw.question, rw.changed, not rw.clear
            first.rewrite_error, first.usage, first.rewrite_ms = rw.error, rw.usage, rw.latency_ms
            return first
        except Exception:  # noqa: BLE001
            get_logger().exception("query_enhancer_failed")
            return None

    def _scope(
        self,
        question: str,
        docs: list[dict[str, Any]],
        company: str | None,
        pre: Enhancement | None,
        use_enhancer: bool = True,
    ) -> Enhancement | None:
        """Document path: narrow the search to the company's reports for the periods in the (rewritten)
        question. Several companies loaded and none picked or named: `needs_company`, nothing is searched."""
        if pre is None:
            return None
        try:
            e = enhance(question, docs, company, plain=not use_enhancer)
        except Exception:  # noqa: BLE001
            get_logger().exception("query_enhancer_failed")
            return pre
        e.original, e.rewritten, e.rewrite_error = pre.original, pre.rewritten, pre.rewrite_error
        e.usage, e.rewrite_ms = pre.usage, pre.rewrite_ms
        if pre.original and pre.original.strip().lower() != question.strip().lower():
            # Search the user's own words too: a rewrite drops exact phrases the report uses ("for the
            # year ended March 31, 2025" -> "for FY25") that keyword search needs to find the statement.
            e.search_query = f"{pre.original} {e.search_query}"
        if self.settings.query.require_company and len(e.available_companies) > 1 and not e.companies:
            e.needs_company = True
        return e

    def _unclear(self, question: str, docs: list[dict[str, Any]], company: str | None) -> _PathResult:
        example = example_question(docs, company)
        text = "I couldn't find a question in that. Could you rephrase it?" + (
            f" For example: “{example}”" if example else ""
        )
        section = Section("document", UNCLEAR, text, question, abstain_reason=UNCLEAR)
        return _PathResult(section, {}, _zero_tokens(), 0.0)

    @staticmethod
    def _needs_company(question: str, enhanced: Enhancement) -> _PathResult:
        names = ", ".join(enhanced.available_companies)
        text = (
            f"Reports from several companies are loaded ({names}). Pick one in the menu above the chat box, "
            "or name the company in your question."
        )
        section = Section("document", NEEDS_COMPANY, text, question, abstain_reason=NEEDS_COMPANY)
        return _PathResult(section, {}, _zero_tokens(), 0.0)

    def _document_path(
        self, question: str, docs: list[dict[str, Any]], enhanced: Enhancement | None = None
    ) -> _PathResult:
        if enhanced is not None and enhanced.needs_company:
            return self._needs_company(question, enhanced)
        t0 = time.perf_counter()
        with get_tracer().span("document", input=question) as sp:
            try:
                if enhanced is None:
                    da = self.doc_answerer.answer(question)
                else:
                    da = self.doc_answerer.answer(
                        question, doc_ids=enhanced.doc_ids, search_query=enhanced.search_query
                    )
            except Exception:  # noqa: BLE001  a broken index must not take the general part down with it
                get_logger().exception("document_path_failed")
                sp.update(level="ERROR", status_message="internal_error")
                section = Section(
                    "document", doc_mod.ERROR, INTERNAL_ERROR, question, abstain_reason="internal_error"
                )
                return _PathResult(section, {}, _zero_tokens(), (time.perf_counter() - t0) * 1000)
            sp.update(
                output=da.answer,
                metadata={
                    "status": da.status,
                    "abstain_reason": da.abstain_reason,
                    "top_score": da.top_score,
                    "cited_chunks": [c.chunk_id for c in da.citations],
                    "number_check": da.number_check,
                },
            )
        section = _document_section(question, da, docs)
        if enhanced is not None:
            section.warnings = enhanced.notes + section.warnings
        return _PathResult(
            section,
            da.timings,
            da.tokens,
            (time.perf_counter() - t0) * 1000,
            da,
        )

    def _general_path(self, question: str, no_documents: bool) -> _PathResult:
        t0 = time.perf_counter()
        with get_tracer().span("general", input=question) as sp:
            try:
                ga = self.general_answerer.answer(question)
            except Exception:  # noqa: BLE001
                get_logger().exception("general_path_failed")
                sp.update(level="ERROR", status_message="internal_error")
                ga = GeneralAnswer("error", INTERNAL_ERROR, error_reason="internal_error")
            sp.update(output=ga.message if ga.status == "answered" else None, metadata={"status": ga.status})
        return _PathResult(
            _general_section(question, ga, no_documents=no_documents),
            ga.timings,
            ga.tokens,
            (time.perf_counter() - t0) * 1000,
            ga,
        )

    # -- response parts

    def _router_meta(self, o: RouteOutcome) -> dict:
        return {
            "ok": o.router_ok,
            "skipped": o.skipped,
            "fallback_reason": o.fallback_reason,
            "model": None if o.skipped else o.model,
            "prompt_version": o.prompt_version,
            "document_question": o.document_question or None,
            "general_question": o.general_question or None,
        }

    @staticmethod
    def _timings(o: RouteOutcome, results: dict[str, _PathResult], t_start: float) -> dict[str, float]:
        doc = results.get("document")
        gen = results.get("general")
        dt = doc.timings if doc else {}
        return {
            "router_ms": round(o.latency_ms, 1),
            "embed_ms": dt.get("embed_ms", 0.0),
            "retrieve_ms": dt.get("retrieve_ms", 0.0),
            "document_llm_ms": dt.get("llm_ms", 0.0),
            "document_ms": round(doc.wall_ms, 1) if doc else 0.0,
            "general_ms": round(gen.wall_ms, 1) if gen else 0.0,
            "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
        }

    @staticmethod
    def _tokens(
        o: RouteOutcome, results: dict[str, _PathResult], enhanced: Enhancement | None = None
    ) -> dict[str, dict[str, int]]:
        eu = enhanced.usage if enhanced else None
        parts = {
            "enhancer": {
                "prompt": eu.prompt_tokens if eu else 0,
                "completion": eu.completion_tokens if eu else 0,
                "total": eu.total_tokens if eu else 0,
            },
            "router": {
                "prompt": o.usage.prompt_tokens,
                "completion": o.usage.completion_tokens,
                "total": o.usage.total_tokens,
            },
            "document": results["document"].tokens if "document" in results else _zero_tokens(),
            "general": results["general"].tokens if "general" in results else _zero_tokens(),
        }
        parts["total"] = {k: sum(p[k] for p in parts.values()) for k in ("prompt", "completion", "total")}
        return parts

    @staticmethod
    def _trace_output(r: dict) -> str:
        return "\n\n".join(s["answer"] for s in r["sections"] if s["status"] == "answered")

    @staticmethod
    def _trace_summary(r: dict) -> dict:
        return {
            "route": r["route"],
            "router_ok": r["router_ok"],
            "statuses": {s["kind"]: s["status"] for s in r["sections"]},
            "tokens": r["tokens"]["total"],
            "timings_ms": r["timings"],
            "llm": r["llm"],
        }

    @staticmethod
    def _log(r: dict) -> None:
        get_logger().info(
            "query",
            trace_id=r["trace_id"],
            route=r["route"],
            router_ok=r["router_ok"],
            router_fallback=r["router"]["fallback_reason"],
            statuses={s["kind"]: s["status"] for s in r["sections"]},
            abstain_reasons={s["kind"]: s["abstain_reason"] for s in r["sections"] if s["abstain_reason"]},
            timings=r["timings"],
            tokens=r["tokens"]["total"],
        )