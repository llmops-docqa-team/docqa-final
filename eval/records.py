"""The per-question record, the run file it is stored in, and the resume rules.

A run is an append-only JSONL file (`eval/results/runs/<name>.jsonl`). The first line is a header with the
fingerprint of everything that decides what a result means (models, prompts, theta, chunking, the indexed
documents); every later line is one question's record. Appending a record for an id that is already there
replaces it (last line wins), which is how a judge verdict is added to a record later. So a run can be split
across days and slices, and a crash loses at most the question in flight.

Record shape (all JSON):
    id, row_hash, mode ("full" | "router"), ts, git, question
    gold      {route, slice, answerable, type, answer, general_answer, pages:[{doc, pdf_page, label}]}
    retrieval {top_score, strict_rank, pm1_rank}               (ranks: answerable DOCUMENT rows only)
    router    {llm, ok, fallback_reason, keyword}
    sections  {document: {...}, general: {...}}                (full mode; only the paths that ran)
    timings, tokens, cost_usd, calls {live, cached}
    infra_error  true when the LLM was unavailable: not a verdict on the system, so it is re-run on resume
    judge     {model, prompt_version, verdicts: {document|general: {correct, grounded, reason, tokens}}}
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.ingestion.index import INGEST_VERSION
from eval.judge import Source
from eval.schema import EvalRow

HEADER = "_header"
INFRA_REASONS = {"llm_unavailable", "internal_error"}


# ---- identity ------------------------------------------------------------------------------------------
def row_hash(row: EvalRow) -> str:
    """Changes when the question, its gold answer or its labels change, so an edited row is re-run."""
    return hashlib.sha256(row.model_dump_json().encode("utf-8")).hexdigest()[:12]


def pipeline_fingerprint(settings: Settings, doc_ids: dict[str, str], embedding_model: str) -> dict[str, Any]:
    """What must be the same for two records to be comparable. The judge is separate (see below)."""
    p = settings.prompts
    return {
        "answer_model": settings.llm.answer_model,
        "router_model": settings.llm.router_model,
        "reasoning_effort": settings.llm.reasoning_effort,
        "prompts": {"router": p.router, "answer_doc": p.answer_doc, "answer_general": p.answer_general},
        "theta": settings.retrieval.theta,
        "fetch_k": settings.retrieval.fetch_k,
        "top_k": settings.retrieval.top_k,
        "retrieval_mode": settings.retrieval.mode,
        "bm25_weight": settings.retrieval.bm25_weight,
        "retrieval_pool": settings.retrieval.pool,
        "ingest_version": INGEST_VERSION,
        "chunk_size": settings.chunking.size_tokens,
        "chunk_overlap": settings.chunking.overlap_tokens,
        "embedding_model": embedding_model,
        "query_prefix": settings.embedding.query_prefix,
        "answer_max_tokens": settings.answer.max_tokens,
        "docs": dict(sorted(doc_ids.items())),
    }


def fingerprint_diff(old: dict, new: dict) -> list[str]:
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def _json_default(o: Any) -> Any:
    """numpy scalars (a vector store may hand back float32) become plain numbers; anything else, text."""
    item = getattr(o, "item", None)
    return item() if callable(item) else str(o)


# ---- the run file --------------------------------------------------------------------------------------
class RunFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> tuple[dict | None, dict[str, dict]]:
        """(latest header or None, records by id). A torn last line (a crash mid-write) is ignored."""
        header: dict | None = None
        records: dict[str, dict] = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None, records
        for line in lines:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            if HEADER in obj:
                header = obj[HEADER]
            elif "id" in obj:
                records[obj["id"]] = obj
        return header, records

    def _append(self, obj: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n"
        with self.path.open("a+b") as f:
            # A crash can leave a torn last line with no newline: start on a fresh line so the torn
            # fragment stays one bad line (ignored by `load`) instead of eating this record too.
            if f.tell() > 0:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            f.write(line.encode("utf-8"))
            f.flush()

    def write_header(self, header: dict) -> None:
        self._append({HEADER: header})

    def append(self, record: dict) -> None:
        self._append(record)

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


# ---- what to judge -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    part: str  # "document" | "general"
    question: str
    reference: str | None
    answer: str
    sources: tuple[Source, ...] = ()


def judge_targets(rec: dict) -> list[Target]:
    """The answers in this record that deserve a judge verdict.

    An answerable DOCUMENT question's answer is judged (correctness and groundedness); so is the document
    half of a MIXED question. A GENERAL answer (and the general half of a MIXED question) is judged for
    correctness against its reference. An unanswerable question that was answered has nothing to compare."""
    gold, secs = rec["gold"], rec.get("sections") or {}
    out: list[Target] = []
    doc = secs.get("document")
    if (
        doc
        and doc["status"] == "answered"
        and (gold["route"] == "MIXED" or (gold["route"] == "DOCUMENT" and gold["answerable"]))
    ):
        sources = tuple(Source(f"p.{s['page_label']}", s["text"]) for s in doc.get("sources") or [])
        out.append(Target("document", doc["question"], gold.get("answer"), doc["answer"], sources))
    gen = secs.get("general")
    if gen and gen["status"] == "answered" and gold["route"] in ("GENERAL", "MIXED"):
        ref = gold.get("general_answer") if gold["route"] == "MIXED" else gold.get("answer")
        out.append(Target("general", gen["question"], ref, gen["answer"]))
    return out


def judge_stamp(rec: dict) -> tuple[str | None, str | None]:
    j = rec.get("judge") or {}
    return j.get("model"), j.get("prompt_version")


def judge_needed(rec: dict, *, rejudge: bool = False, current: tuple[str, str] | None = None) -> bool:
    """Is there an answer in this record without a verdict from the judge in use (`current` = its model and
    prompt version)? Verdicts from a different judge model or prompt do not count."""
    targets = judge_targets(rec)
    if rejudge:
        return bool(targets)
    stale = current is not None and rec.get("judge") and judge_stamp(rec) != current
    done = {} if stale else (rec.get("judge") or {}).get("verdicts") or {}
    return any(t.part not in done for t in targets)


def verdict_of(rec: dict, part: str) -> dict | None:
    return ((rec.get("judge") or {}).get("verdicts") or {}).get(part)


# ---- resume --------------------------------------------------------------------------------------------
def plan(
    rec: dict | None,
    row: EvalRow,
    *,
    mode: str,
    judge_enabled: bool,
    rejudge: bool = False,
    judge_current: tuple[str, str] | None = None,
) -> tuple[bool, bool]:
    """(run the pipeline, run the judge) for one question, given what the run file already holds.

    The pipeline re-runs when there is no record, the row changed, the record only has the router (and this
    is a full run), or the LLM was unavailable last time. The judge runs when a verdict is missing; it never
    needs the pipeline again, because the answer and its cited text are in the record."""
    if rec is None or rec.get("row_hash") != row_hash(row):
        return True, judge_enabled
    complete = rec.get("mode") == "full" or mode == "router"
    if not complete or rec.get("infra_error"):
        return True, judge_enabled
    return False, judge_enabled and judge_needed(rec, rejudge=rejudge, current=judge_current)


def infra_error(sections: Iterable[dict], router_fallback: str | None) -> bool:
    """True when something failed because an LLM was unavailable (or the code crashed), not because the
    system gave a wrong answer."""
    return router_fallback == "llm_unavailable" or any(
        s.get("status") == "error" and s.get("abstain_reason") in INFRA_REASONS for s in sections
    )
