"""Online judge: score recent answers from the request log and store the scores where the Metrics page reads
them (`requests.judge_correct` / `judge_grounded`). The offline eval grades against gold answers; live traffic
has none, so here `correct` means "answers the question and agrees with the cited sources" and `grounded`
means "every claim is in the cited sources". Only the document part of an answered request is judged: a
general-knowledge answer cites nothing to check it against.

The request log holds no text (privacy). The question, the answer and the cited chunk text are only kept
when `observability.log_content` is true, in the separate `request_content` table, so requests logged
while it was off cannot be judged and are reported as such.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.observability.tracing import get_tracer
from app.storage.requests import RequestStore
from eval.judge import Judge, JudgeError, JudgeInput, Source


@dataclass
class OnlineResult:
    judged: int = 0
    correct: int = 0
    grounded: int = 0
    grounded_n: int = 0
    no_content: int = 0  # answered, but no text was stored (log_content was off)
    not_document: int = 0  # answered by the general path only: nothing to check it against
    failed: int = 0  # the judge could not be reached or gave an unusable reply
    details: list[str] = field(default_factory=list)


def _document_section(sections: list[dict]) -> dict | None:
    return next((s for s in sections if s["kind"] == "document" and s["status"] == "answered"), None)


def judge_recent(
    store: RequestStore, judge: Judge, last: int, *, rejudge: bool = False, out: Callable[[str], None] = print
) -> OnlineResult:
    """Judge the last `last` answered requests (by default only those not yet judged), newest first."""
    res = OnlineResult()
    for row in store.recent_answered(last, only_unjudged=not rejudge):
        if not row["sections"]:
            res.no_content += 1
            continue
        sec = _document_section(row["sections"])
        if sec is None or not sec.get("answer"):
            res.not_document += 1
            continue
        sources = tuple(Source(f"p.{s['page_label']}", s["text"]) for s in sec.get("sources") or [])
        try:
            v = judge.judge(JudgeInput(sec["question"] or row["question"], sec["answer"], None, sources))
        except JudgeError as exc:
            res.failed += 1
            res.details.append(f"{row['trace_id']}: {exc}")
            out(f"  {row['trace_id']}: judge failed ({str(exc)[:80]})")
            continue
        store.set_judge(row["trace_id"], v.correct, v.grounded)
        tracer = get_tracer()  # the verdict goes on the request's Langfuse trace too (no-op when off)
        tracer.score(row["trace_id"], "judge_correct", v.correct, boolean=True)
        if v.grounded is not None:
            tracer.score(row["trace_id"], "judge_grounded", v.grounded, boolean=True)
        res.judged += 1
        res.correct += int(v.correct)
        if v.grounded is not None:
            res.grounded_n += 1
            res.grounded += int(v.grounded)
        grounded = "-" if v.grounded is None else int(v.grounded)
        out(f"  {row['trace_id']}: correct={int(v.correct)} grounded={grounded}")
    return res


def summary(res: OnlineResult) -> str:
    lines = [f"judged {res.judged} answer(s)"]
    if res.judged:
        lines.append(f"  correct  {res.correct}/{res.judged} ({res.correct / res.judged * 100:.0f}%)")
    if res.grounded_n:
        lines.append(
            f"  grounded {res.grounded}/{res.grounded_n} ({res.grounded / res.grounded_n * 100:.0f}%)"
        )
    if res.no_content:
        lines.append(
            f"  {res.no_content} answered request(s) have no stored text and cannot be judged: set "
            "observability.log_content: true in config.yaml so new requests keep their text"
        )
    if res.not_document:
        lines.append(
            f"  {res.not_document} answered request(s) had no document answer (general knowledge only)"
        )
    if res.failed:
        lines.append(f"  {res.failed} judge call(s) failed; run the script again to retry them")
    return "\n".join(lines)
