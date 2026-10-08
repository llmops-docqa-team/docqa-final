"""Answer a document question from retrieved chunks, with citations, or abstain (design §10-§11).

Flow: retrieve -> gate 1 (top score >= theta) -> build <source> context from the top-k chunks -> LLM
(JSON) -> gate 2 (status INSUFFICIENT) -> gate 3 (citation ids must map to the provided sources) ->
numeric grounding check (warn only). The LLM never writes page numbers: it returns source ids and we map
them to (filename, page label, PDF page, snippet).

`status` values match the `requests` table (step 08): answered | abstained | not_ready | error.
Nothing here logs question or document text, only ids, scores, counts and timings.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from app.answering.highlight import highlight_terms
from app.answering.numbers import check_numbers
from app.config import Settings
from app.llm.client import LLMClient, LLMError, LLMResponse, Usage
from app.llm.json_reply import load_json_object
from app.llm.prompts import Prompt, load_prompt
from app.observability.logging import get_logger
from app.observability.tracing import get_tracer, usage_details
from app.retrieval.retriever import DocStatus, RetrievalResult, RetrievedChunk, Retriever, fit_context
from app.storage import documents as st

ANSWERED, ABSTAINED, NOT_READY, ERROR = "answered", "abstained", "not_ready", "error"

# abstain_reason values
NO_DOCUMENTS = "no_documents"  # nothing uploaded
DOCS_NOT_READY = "documents_not_ready"  # uploaded, but none is queryable yet / all failed
NO_RESULTS = "no_results"  # queryable docs, but the search returned nothing
LOW_SCORE = "low_score"  # gate 1
INSUFFICIENT = "insufficient"  # gate 2
NO_VALID_CITATIONS = "no_valid_citations"  # gate 3
BAD_LLM_OUTPUT = "bad_llm_output"  # unparseable JSON twice in a row
LLM_UNAVAILABLE = "llm_unavailable"  # status "error"

UNAVAILABLE_MESSAGE = "The answer service is temporarily unavailable."
RETRY_NUDGE = (
    "Your reply was not a valid JSON object of the required shape. Reply again with only "
    '{"answer": "...", "citations": ["S1"], "status": "ANSWERED" or "INSUFFICIENT"}.'
)


# ---------------------------------------------------------------- result types


@dataclass
class Citation:
    id: str  # "S1"
    doc_id: str
    filename: str
    page_label: str  # printed page number
    pdf_page: int  # 1-based index in the PDF
    display: str  # "p.47 (PDF p.53)"
    snippet: str
    chunk_id: str
    score: float
    source_kind: str
    highlight_terms: list[str] = field(default_factory=list)  # the answer's figures as printed in this chunk
    primary: bool = False  # the citation holding the most of them: the UI opens it first


@dataclass
class DocAnswer:
    status: str  # answered | abstained | not_ready | error
    message: str  # what to show the user (the answer itself when answered)
    answer: str | None = None
    abstain_reason: str | None = None
    citations: list[Citation] = field(default_factory=list)
    closest_pages: list[Citation] = field(default_factory=list)  # abstention: nearest retrieved pages
    fallback_passages: list[Citation] = field(default_factory=list)  # error: top retrieved passages
    coverage_note: str | None = None
    notices: list[str] = field(default_factory=list)  # not-ready / failed documents
    number_check: str = "na"  # pass | fail | na
    number_warning: bool = False
    unmatched_numbers: list[str] = field(default_factory=list)
    computed_numbers: list[str] = field(default_factory=list)  # figures derived from cited ones
    citations_valid: bool | None = None  # None when the LLM was never asked
    dropped_citations: list[str] = field(default_factory=list)
    top_score: float | None = None
    n_sources: int = 0
    searched: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    llm_calls: int = 0
    model: str | None = None
    prompt_version: str = ""
    # Full text of the cited chunks (chunk id -> text), for the eval judge and the opt-in content log.
    # Never part of an API response (`Section` copies fields one by one) and dropped from `to_dict`.
    cited_texts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("cited_texts")
        return d


class LLMAnswer(BaseModel):
    answer: str = ""
    citations: list[str] = []
    status: Literal["ANSWERED", "INSUFFICIENT"]

    @field_validator("status", mode="before")
    @classmethod
    def _norm_status(cls, v):
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("citations", mode="before")
    @classmethod
    def _norm_citations(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, int)):
            v = [v]
        return [str(x) for x in v] if isinstance(v, list) else v

    @model_validator(mode="after")
    def _answered_needs_text(self):
        if self.status == "ANSWERED" and not self.answer.strip():
            raise ValueError("status ANSWERED with an empty answer")
        return self


# ---------------------------------------------------------------- helpers


def page_display(page_label: str, pdf_page: int) -> str:
    """'p.47 (PDF p.53)'; just 'p.47' when the printed label is the PDF page itself."""
    if str(page_label) == str(pdf_page):
        return f"p.{page_label}"
    return f"p.{page_label} (PDF p.{pdf_page})"


def _snippet(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def make_citation(source_id: str, chunk: RetrievedChunk, snippet_chars: int) -> Citation:
    return Citation(
        id=source_id,
        doc_id=chunk.doc_id,
        filename=chunk.filename,
        page_label=chunk.page_label,
        pdf_page=chunk.page,
        display=page_display(chunk.page_label, chunk.page),
        snippet=_snippet(chunk.text, snippet_chars),
        chunk_id=chunk.id,
        score=chunk.score,
        source_kind=chunk.source_kind,
    )


def _with_highlights(
    citations: list[Citation], answer: str, computed: tuple[str, ...], by_id: dict[str, RetrievedChunk]
) -> list[Citation]:
    """Fill `highlight_terms` on each citation and mark the one with the most terms as primary."""
    for c in citations:
        c.highlight_terms = highlight_terms(answer, list(computed), by_id[c.id].text)
    best = max(citations, key=lambda c: len(c.highlight_terms), default=None)
    if best is not None and best.highlight_terms:
        best.primary = True
    return citations


_SOURCE_TAG = re.compile(r"<(?=\s*/?\s*source\b)", re.IGNORECASE)


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, RetrievedChunk]]:
    """`<source id="S1" doc="..." page="47">...</source>` blocks, plus the id -> chunk map.

    A literal `<source` / `</source` inside document text is defused so a page cannot close the tag early
    and smuggle in text that looks like it is outside the data block."""
    blocks, by_id = [], {}
    for i, chunk in enumerate(chunks, start=1):
        sid = f"S{i}"
        by_id[sid] = chunk
        body = _SOURCE_TAG.sub("&lt;", chunk.text)
        doc = html.escape(chunk.filename, quote=True)
        page = html.escape(str(chunk.page_label), quote=True)
        blocks.append(f'<source id="{sid}" doc="{doc}" page="{page}">\n{body}\n</source>')
    return "\n\n".join(blocks), by_id


def parse_llm_json(content: str) -> LLMAnswer:
    """Validate the model's reply. Tolerates a ```json fence or text around one object. Raises ValueError."""
    return LLMAnswer.model_validate(load_json_object(content))  # pydantic's ValidationError is a ValueError


def _norm_source_id(raw: str) -> str:
    return raw.strip().strip("[]() ").upper()


def not_ready_notice(doc: DocStatus) -> str:
    if doc.status == st.FAILED:
        return f"{doc.filename} couldn't be processed: {doc.error or 'unknown error'}."
    done, total = doc.pages_done, doc.pages_total if doc.pages_total is not None else "?"
    return f"{doc.filename} is still processing ({done}/{total} pages). Ask again when it's Ready."


def _closest_pages(chunks: list[RetrievedChunk], limit: int, snippet_chars: int) -> list[Citation]:
    """Top retrieved pages (one entry per document page), best first."""
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for i, chunk in enumerate(chunks, start=1):
        key = (chunk.doc_id, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        out.append(make_citation(f"S{i}", chunk, snippet_chars))
        if len(out) == limit:
            break
    return out


# ---------------------------------------------------------------- the answerer


class DocumentAnswerer:
    def __init__(
        self, retriever: Retriever, llm: LLMClient, settings: Settings, prompt: Prompt | None = None
    ):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings
        self.prompt = prompt or load_prompt(settings.prompts.answer_doc)

    def answer(
        self,
        question: str,
        *,
        doc_ids: list[str] | None = None,
        retrieval: RetrievalResult | None = None,
        search_query: str | None = None,
    ) -> DocAnswer:
        """Retrieve (unless `retrieval` is passed in, as the router will) and answer.

        `timings["total_ms"]` covers retrieval only if it was done here."""
        t_start = time.perf_counter()
        if retrieval is None:
            retrieval = self._retrieve(search_query or question, doc_ids)
        result = self._answer(question, retrieval)
        result.timings.setdefault("llm_ms", 0.0)
        result.timings["embed_ms"] = round(retrieval.t_embed_ms, 1)
        result.timings["retrieve_ms"] = round(retrieval.t_retrieve_ms, 1)
        result.timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        self._log(result)
        return result

    # -- stages

    def _retrieve(self, question: str, doc_ids: list[str] | None) -> RetrievalResult:
        cfg = self.settings
        with get_tracer().span(
            "retrieve",
            input=question,
            metadata={"mode": cfg.retrieval.mode, "top_k": cfg.retrieval.top_k, "theta": cfg.retrieval.theta},
        ) as sp:
            retrieval = self.retriever.retrieve(question, doc_ids=doc_ids)
            sp.update(
                # Content (dropped unless tracing.capture_content): snippets of what the model will see.
                output=[
                    {
                        "chunk_id": c.id,
                        "page": c.page_label,
                        "snippet": _snippet(c.text, cfg.answer.snippet_chars),
                    }
                    for c in retrieval.chunks[: cfg.retrieval.top_k]
                ],
                metadata={
                    "source_ids": [c.id for c in retrieval.chunks],
                    "scores": [round(c.score, 3) for c in retrieval.chunks],
                    "top_score": retrieval.top_score,
                    "n_docs_searched": len(retrieval.searched),
                },
            )
        return retrieval

    def _answer(self, question: str, retrieval: RetrievalResult) -> DocAnswer:
        cfg = self.settings
        chunks = retrieval.chunks[: cfg.retrieval.top_k]
        base = dict(
            prompt_version=self.prompt.version,
            searched=[d.filename for d in retrieval.searched],
            top_score=retrieval.top_score,
            n_sources=len(chunks),
            notices=[not_ready_notice(d) for d in retrieval.not_ready],
        )

        if not chunks:
            return self._no_chunks(retrieval, base)

        # Gate 1: weak retrieval (no chunk similar enough) -> abstain without spending an LLM call.
        if max(c.score for c in chunks) < cfg.retrieval.theta:
            return self._abstain(LOW_SCORE, retrieval, chunks, base)

        # The rest of any split table a piece comes from, then the whole set cut to the context budget.
        siblings = getattr(self.retriever, "table_siblings", None) if cfg.retrieval.table_siblings else None
        chunks = fit_context(chunks, cfg.answer.context_max_tokens, siblings)
        base["n_sources"] = len(chunks)

        context, by_id = build_context(chunks)
        messages = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": self.prompt.render_user(sources=context, question=question)},
        ]
        usage, llm_ms, calls = Usage(), 0.0, 0
        parsed: LLMAnswer | None = None
        model = cfg.llm.answer_model
        backend = ""
        with get_tracer().generation(
            "generate",
            model=model,
            input=question,
            metadata={
                "prompt": cfg.prompts.answer_doc,
                "prompt_version": self.prompt.version,
                "path": "document",
                "source_ids": [c.id for c in chunks],
            },
        ) as gen:
            try:
                for attempt in range(2):  # the second attempt is the one retry on bad JSON
                    resp = self._call(messages)
                    usage.add(resp.usage)
                    llm_ms += resp.latency_ms
                    calls += 1
                    model = resp.model or model
                    backend = resp.backend or backend
                    try:
                        parsed = parse_llm_json(resp.content)
                        break
                    except ValueError:
                        if attempt == 0:
                            messages = messages + [
                                {"role": "assistant", "content": resp.content or "(empty)"},
                                {"role": "user", "content": RETRY_NUDGE},
                            ]
            except LLMError as exc:
                get_logger().warning("answer_doc_llm_error", error=str(exc), status=exc.status)
                gen.update(level="ERROR", status_message=f"llm_unavailable (HTTP {exc.status})")
                result = DocAnswer(
                    status=ERROR,
                    message=UNAVAILABLE_MESSAGE,
                    abstain_reason=LLM_UNAVAILABLE,
                    fallback_passages=_closest_pages(
                        chunks, cfg.answer.fallback_passages, cfg.answer.snippet_chars
                    ),
                    **base,
                )
                return self._with_llm_stats(result, usage, llm_ms, calls, model)
            finally:
                gen.update(
                    model=model,
                    usage_details=usage_details(
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                    ),
                    metadata={"llm_calls": calls, "backend": backend},
                )
            gen.update(
                output=parsed.answer if parsed else None,
                metadata={"reply_status": parsed.status if parsed else "unparseable"},
            )

        if parsed is None:
            result = self._abstain(BAD_LLM_OUTPUT, retrieval, chunks, base)
            return self._with_llm_stats(result, usage, llm_ms, calls, model)

        # Gate 2: the model says the sources do not contain the answer.
        if parsed.status == "INSUFFICIENT":
            result = self._abstain(INSUFFICIENT, retrieval, chunks, base)
            return self._with_llm_stats(result, usage, llm_ms, calls, model)

        # Gate 3: every cited id must be one we sent. Unknown ids are dropped; none left -> abstain.
        valid, dropped = [], []
        for raw in parsed.citations:
            sid = _norm_source_id(raw)
            if sid in by_id:
                if sid not in valid:
                    valid.append(sid)
            else:
                dropped.append(raw)
        if not valid:
            result = self._abstain(NO_VALID_CITATIONS, retrieval, chunks, base)
            result.citations_valid = False
            result.dropped_citations = dropped
            return self._with_llm_stats(result, usage, llm_ms, calls, model)

        cited_chunks = [by_id[s] for s in valid]
        check = check_numbers(parsed.answer, [c.text for c in cited_chunks])
        result = DocAnswer(
            status=ANSWERED,
            message=parsed.answer.strip(),
            answer=parsed.answer.strip(),
            citations=_with_highlights(
                [make_citation(s, by_id[s], cfg.answer.snippet_chars) for s in valid],
                parsed.answer,
                check.computed,
                by_id,
            ),
            number_check=check.status,
            number_warning=check.warning,
            unmatched_numbers=list(check.missing),
            computed_numbers=list(check.computed),
            citations_valid=not dropped,
            dropped_citations=dropped,
            cited_texts={c.id: c.text for c in cited_chunks},
            **base,
        )
        return self._with_llm_stats(result, usage, llm_ms, calls, model)

    def _call(self, messages: list[dict[str, str]]) -> LLMResponse:
        cfg = self.settings
        return self.llm.chat(
            messages,
            model=cfg.llm.answer_model,
            temperature=0.0,
            max_tokens=cfg.answer.max_tokens,
            reasoning_effort=cfg.llm.reasoning_effort,
            json_mode=True,
        )

    def _no_chunks(self, retrieval: RetrievalResult, base: dict) -> DocAnswer:
        """Nothing to read: no documents, none ready, or an empty search."""
        if retrieval.searched:
            return self._abstain(NO_RESULTS, retrieval, [], base)
        if retrieval.not_ready:
            return DocAnswer(
                status=NOT_READY, message=" ".join(base["notices"]), abstain_reason=DOCS_NOT_READY, **base
            )
        return DocAnswer(
            status=ABSTAINED,
            abstain_reason=NO_DOCUMENTS,
            **{**base, "notices": []},
            message="No documents have been uploaded yet. Upload a PDF to ask about it.",
        )

    def _abstain(
        self, reason: str, retrieval: RetrievalResult, chunks: list[RetrievedChunk], base: dict
    ) -> DocAnswer:
        cfg = self.settings
        closest = _closest_pages(chunks, cfg.answer.closest_pages, cfg.answer.snippet_chars)
        coverage = retrieval.coverage_note()
        searched = ", ".join(base["searched"]) or "your documents"
        if reason == BAD_LLM_OUTPUT:
            message = (
                "I couldn't produce a reliable answer: the answer model returned an unusable reply. "
                "Please try again."
            )
        else:
            message = f"I couldn't find this in your documents (searched: {searched}). I won't guess."
        if closest:
            message += " Closest sections: " + ", ".join(f"{c.filename} {c.display}" for c in closest) + "."
        parts = [message] + ([coverage] if coverage else []) + base["notices"]
        return DocAnswer(
            status=ABSTAINED,
            message=" ".join(parts),
            abstain_reason=reason,
            closest_pages=closest,
            coverage_note=coverage,
            **base,
        )

    @staticmethod
    def _with_llm_stats(result: DocAnswer, usage: Usage, llm_ms: float, calls: int, model: str) -> DocAnswer:
        result.timings["llm_ms"] = round(llm_ms, 1)
        result.tokens = {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens,
        }
        result.llm_calls = calls
        result.model = model
        return result

    @staticmethod
    def _log(r: DocAnswer) -> None:
        get_logger().info(
            "answer_doc",
            status=r.status,
            abstain_reason=r.abstain_reason,
            top_score=r.top_score,
            n_sources=r.n_sources,
            citations=[c.chunk_id for c in r.citations],
            citations_valid=r.citations_valid,
            number_check=r.number_check,
            llm_calls=r.llm_calls,
            tokens=r.tokens,
            timings=r.timings,
            prompt_version=r.prompt_version,
            model=r.model,
        )