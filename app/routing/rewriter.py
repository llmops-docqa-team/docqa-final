"""LLM half of the query enhancer: fixes spelling/grammar, spells out short forms, names the pinned company.

One small JSON-mode call on `llm.rewrite_model` (gpt-oss-20b on Groq's free tier: its rate limit is separate
from the answer model's). Fail-safe: any error, bad JSON or a suspicious reply keeps the user's own question.
Nothing here logs question text.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.llm.client import LLMClient, LLMError, Usage
from app.llm.json_reply import load_json_object
from app.llm.prompts import Prompt, load_prompt
from app.observability.logging import get_logger
from app.observability.tracing import get_tracer, usage_details


@dataclass
class Rewrite:
    question: str  # the rewritten question, or the original when the rewrite was not usable
    changed: bool = False
    error: str | None = None  # llm_unavailable | bad_json | rejected
    clear: bool = True  # False: the model could not find a question in the text (keyboard mash, noise)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


def acceptable(original: str, rewritten: str, max_chars: int) -> bool:
    """A rewrite that is empty, far longer than the question, over the length limit, or that changes a year
    or a number the user gave is not used (the user's own words are searched instead)."""
    r = rewritten.strip()
    return (
        bool(r) and len(r) <= max_chars and len(r) <= 3 * len(original) + 120
        and same_years_and_numbers(original, r)
    )


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DAY_FIRST = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?={_MONTH})", re.IGNORECASE)  # "31 March"
# "March 31" (not "March 2025")
_DAY_AFTER = re.compile(rf"\b({_MONTH})\s*\d{{1,2}}(?:st|nd|rd|th)?\b(?![\d,.]\d)", re.IGNORECASE)
_RANGE = re.compile(r"\b(?:fy\s?)?(19|20)(\d\d)\s*[-–/]\s*(\d\d)\b", re.IGNORECASE)  # 2024-25, FY2024-25
_FY = re.compile(r"\bfy\s?'?(\d{4}|\d{2})\b", re.IGNORECASE)  # FY25, FY 2025
_YEAR = re.compile(r"\b(19|20)\d\d\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _years_and_numbers(text: str) -> tuple[set[int], set[str]]:
    """Years a question refers to, as the year a period ends in ("2024-25" and "FY25" -> 2025,
    "March 31, 2024" -> 2024), and its other numbers. The day in a date is dropped: "year ended
    March 31, 2024" is "FY24"."""
    t = _DAY_AFTER.sub(r"\1 ", _DAY_FIRST.sub(" ", text))
    years: set[int] = set()

    def take(m: re.Match, value: int) -> str:
        years.add(value)
        return " "

    t = _RANGE.sub(lambda m: take(m, int(m.group(1) + m.group(3))), t)
    t = _FY.sub(lambda m: take(m, int(m.group(1)) if len(m.group(1)) == 4 else 2000 + int(m.group(1))), t)
    t = _YEAR.sub(lambda m: take(m, int(m.group(0))), t)
    return years, {n.replace(",", "") for n in _NUMBER.findall(t)}


def same_years_and_numbers(original: str, rewritten: str) -> bool:
    """The rewrite keeps every year and number of the question (spelled any way: "fy2026" -> "FY26" is fine,
    "year ended March 31, 2024" -> "FY25" is not) and adds no year of its own."""
    oy, on = _years_and_numbers(original)
    ry, rn = _years_and_numbers(rewritten)
    return oy == ry and on <= rn


class QueryRewriter:
    def __init__(self, llm: LLMClient, settings: Settings, prompt: Prompt | None = None):
        self.llm = llm
        self.settings = settings
        self.prompt = prompt or load_prompt(settings.prompts.rewrite)

    def rewrite(self, question: str, company: str | None, periods: list[str]) -> Rewrite:
        cfg = self.settings
        messages = [
            {"role": "system", "content": self.prompt.system},
            {
                "role": "user",
                "content": self.prompt.render_user(
                    company=company or "(none)",
                    periods=", ".join(periods) or "(unknown)",
                    question=question,
                ),
            },
        ]
        model = cfg.llm.rewrite_model or cfg.llm.router_model
        t0 = time.perf_counter()
        with get_tracer().generation(
            "rewrite", model=model, input=question, metadata={"prompt_version": self.prompt.version}
        ) as gen:
            try:
                resp = self.llm.chat(
                    messages,
                    model=model,
                    temperature=0.0,
                    max_tokens=cfg.query.rewrite_max_tokens,
                    reasoning_effort=cfg.llm.reasoning_effort,
                    json_mode=True,
                )
            except LLMError as exc:
                gen.update(level="ERROR", status_message=f"llm_unavailable (HTTP {exc.status})")
                get_logger().warning("rewrite_failed", reason="llm_unavailable", status=exc.status)
                ms = (time.perf_counter() - t0) * 1000
                return Rewrite(question, error="llm_unavailable", latency_ms=ms)
            gen.update(
                model=resp.model or model,
                usage_details=usage_details(
                    resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens
                ),
            )
        base = dict(usage=resp.usage, latency_ms=resp.latency_ms)
        try:
            reply = load_json_object(resp.content)
        except ValueError:
            get_logger().warning("rewrite_failed", reason="bad_json")
            return Rewrite(question, error="bad_json", **base)
        if reply.get("clear") is False:
            return Rewrite(question, clear=False, **base)
        text = " ".join(str(reply.get("question") or "").split())
        if not acceptable(question, text, cfg.query.max_question_chars):
            get_logger().warning("rewrite_failed", reason="rejected")
            return Rewrite(question, error="rejected", **base)
        return Rewrite(text, changed=text != question.strip(), **base)