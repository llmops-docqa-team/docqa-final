"""LLM-as-judge behind a small interface, so the judge can be swapped without touching the runner.

`Judge.judge(JudgeInput) -> Verdict` is the whole contract. `LLMJudge` is the implementation: Gemini Flash
through its OpenAI-compatible endpoint, a different model family from the generator (design §12) so the
judge is not grading its own style. It reuses `LLMClient`, so it gets the same retries and the same dev
cache for free, and it spaces live calls to stay under the free tier's requests-per-minute.

One call returns two verdicts: `correct` (against a reference answer) and `grounded` (against the cited
sources). Online, where there is no reference answer, `reference` is None and `correct` means "answers the
question and agrees with the sources".
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.llm.client import DiskCache, LLMClient, LLMError, cache_enabled
from app.llm.json_reply import load_json_object
from app.llm.prompts import Prompt, load_prompt

NO_REFERENCE = "(none)"
NO_SOURCES = "(none)"
_SOURCE_TAG = re.compile(r"<(?=\s*/?\s*(?:source|sources|system_answer|reference_answer|question)\b)", re.I)


class JudgeError(Exception):
    """The judge could not give a usable verdict (API down, quota, unparseable reply). Not a bad answer."""


class JudgeQuotaExhausted(JudgeError):
    """A per-day quota is used up: waiting a few seconds will not help, so callers should stop asking."""


@dataclass(frozen=True)
class Source:
    label: str  # the page, for the judge's benefit ("p.56")
    text: str


@dataclass(frozen=True)
class JudgeInput:
    question: str
    answer: str
    reference: str | None = None
    sources: tuple[Source, ...] = ()


@dataclass(frozen=True)
class Verdict:
    correct: bool
    grounded: bool | None  # None: nothing to be grounded in (a general-knowledge answer)
    reason: str = ""
    tokens: int = 0
    cached: bool = False


class Judge(Protocol):
    model: str
    prompt_version: str

    def judge(self, item: JudgeInput) -> Verdict: ...


def _binary(value: object, field: str, *, nullable: bool = False) -> bool | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("0", "1", "true", "false", "yes", "no"):
        return value.strip().lower() in ("1", "true", "yes")
    raise ValueError(f"'{field}' must be 0 or 1{' or null' if nullable else ''}, got {value!r}")


def parse_verdict(content: str, *, tokens: int = 0, cached: bool = False) -> Verdict:
    """Validate the judge's reply. Raises ValueError (so the caller can treat it as a judge error)."""
    data = load_json_object(content)
    if "correct" not in data:
        raise ValueError("the judge reply has no 'correct'")
    correct = _binary(data["correct"], "correct")
    grounded = _binary(data.get("grounded"), "grounded", nullable=True)
    reason = " ".join(str(data.get("reason") or "").split())[:300]
    return Verdict(bool(correct), grounded, reason, tokens, cached)


def format_sources(sources: Sequence[Source], max_chars: int) -> str:
    """`<source id="S1" page="p.56">...</source>` blocks. A literal tag inside source text is defused so a
    page cannot close the block early."""
    if not sources:
        return NO_SOURCES
    blocks = []
    for i, s in enumerate(sources, start=1):
        text = _SOURCE_TAG.sub("&lt;", s.text.strip())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " [cut]"
        label = s.label.replace('"', "").replace("<", "").replace(">", "")
        blocks.append(f'<source id="S{i}" page="{label}">\n{text}\n</source>')
    return "\n\n".join(blocks)


def _defuse(text: str) -> str:
    return _SOURCE_TAG.sub("&lt;", text)


class LLMJudge:
    def __init__(
        self,
        llm: LLMClient,
        settings: Settings,
        prompt: Prompt | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.llm = llm
        self.cfg = settings
        self.prompt = prompt or load_prompt(settings.prompts.judge)
        self.model = settings.llm.judge_model
        self.prompt_version = self.prompt.version
        self._clock, self._sleep = clock, sleep
        self._last_live: float | None = None  # when the last call that really hit the API finished

    def judge(self, item: JudgeInput) -> Verdict:
        cfg = self.cfg
        messages = [
            {"role": "system", "content": self.prompt.system},
            {
                "role": "user",
                "content": self.prompt.render_user(
                    question=_defuse(item.question),
                    reference=_defuse(item.reference) if item.reference else NO_REFERENCE,
                    answer=_defuse(item.answer),
                    sources=format_sources(item.sources, cfg.eval.judge_source_chars),
                ),
            },
        ]
        resp = None
        for attempt in range(cfg.eval.judge_rate_limit_retries + 1):
            self._pace()
            try:
                resp = self.llm.chat(
                    messages,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=cfg.eval.judge_max_tokens,
                    reasoning_effort=cfg.llm.judge_reasoning_effort or None,
                    json_mode=True,
                )
                break
            except LLMError as exc:
                # A rate limit clears with time: wait it out here instead of failing the verdict.
                if exc.status == 429 and "PerDay" in str(exc):
                    retry = re.search(r"retry in ([\w.]+)", str(exc))
                    raise JudgeQuotaExhausted(
                        "the judge's daily quota is used up"
                        + (f" (resets in about {retry.group(1)})" if retry else "")
                    ) from exc
                if exc.status == 429 and attempt < cfg.eval.judge_rate_limit_retries:
                    self._sleep(cfg.eval.judge_rate_limit_wait_seconds)
                    self._last_live = self._clock()
                    continue
                raise JudgeError(f"judge call failed: {exc}") from exc
        assert resp is not None
        if not resp.cached:
            self._last_live = self._clock()
        if resp.finish_reason == "length":
            raise JudgeError("the judge's reply was cut off at max_tokens (raise eval.judge_max_tokens)")
        try:
            return parse_verdict(resp.content, tokens=resp.usage.total_tokens, cached=resp.cached)
        except ValueError as exc:
            raise JudgeError(f"unusable judge reply: {exc}") from exc

    def _pace(self) -> None:
        """Space live calls `judge_min_interval_seconds` apart. A cached replay never sets `_last_live`, so a
        run that is mostly cached does not wait."""
        if self._last_live is None:
            return
        remaining = self.cfg.eval.judge_min_interval_seconds - (self._clock() - self._last_live)
        if remaining > 0:
            self._sleep(remaining)


def judge_llm_from_settings(settings: Settings, *, use_cache: bool = True) -> LLMClient | None:
    """An `LLMClient` aimed at the judge's endpoint with the Gemini key. None if GEMINI_API_KEY is not set."""
    if not settings.gemini_api_key:
        return None
    s = settings.model_copy(deep=True)
    s.llm.base_url = settings.llm.judge_base_url
    s.groq_api_key = settings.gemini_api_key  # LLMClient sends whatever key it holds; here the Gemini one
    cache = DiskCache(settings.llm_cache_dir) if use_cache and cache_enabled(True) else None
    return LLMClient(s, cache=cache)


def judge_from_settings(settings: Settings, *, use_cache: bool = True) -> LLMJudge | None:
    llm = judge_llm_from_settings(settings, use_cache=use_cache)
    return None if llm is None else LLMJudge(llm, settings)
