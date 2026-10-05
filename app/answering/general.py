"""General-knowledge answer path (design §11): answered from the model's own knowledge, never from the
user's documents, and always carrying the label below. No retrieval, no citations.

An LLM failure or an empty reply gives `status="error"` (the label is still attached), never an exception.
Nothing here logs question or answer text.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from app.config import Settings
from app.llm.client import LLMClient, LLMError, Usage
from app.llm.prompts import Prompt, load_prompt
from app.observability.logging import get_logger
from app.observability.tracing import get_tracer, usage_details

GENERAL_LABEL = "General knowledge — not from your documents; may be out of date."
UNAVAILABLE_MESSAGE = "The answer service is temporarily unavailable."
EMPTY_MESSAGE = "I couldn't produce an answer this time. Please try again."

ANSWERED, ERROR = "answered", "error"


@dataclass
class GeneralAnswer:
    status: str  # answered | error
    message: str  # the answer, or what to show instead
    label: str = GENERAL_LABEL
    error_reason: str | None = None  # llm_unavailable | empty_reply
    timings: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    model: str | None = None
    prompt_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class GeneralAnswerer:
    def __init__(self, llm: LLMClient, settings: Settings, prompt: Prompt | None = None):
        self.llm = llm
        self.settings = settings
        self.prompt = prompt or load_prompt(settings.prompts.answer_general)

    def answer(self, question: str) -> GeneralAnswer:
        cfg = self.settings
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": self.prompt.render_user(question=question)},
        ]
        base = dict(prompt_version=self.prompt.version, model=cfg.llm.answer_model)
        with get_tracer().generation(
            "generate",
            model=cfg.llm.answer_model,
            input=question,
            metadata={
                "prompt": cfg.prompts.answer_general,
                "prompt_version": self.prompt.version,
                "path": "general",
            },
        ) as gen:
            try:
                resp = self.llm.chat(
                    messages,
                    model=cfg.llm.answer_model,
                    temperature=0.0,
                    max_tokens=cfg.query.general_max_tokens,
                    reasoning_effort=cfg.llm.reasoning_effort,
                )
            except LLMError as exc:
                get_logger().warning("answer_general_llm_error", error=str(exc), status=exc.status)
                gen.update(level="ERROR", status_message=f"llm_unavailable (HTTP {exc.status})")
                resp = None
            else:
                u = resp.usage
                gen.update(
                    model=resp.model or None,
                    output=resp.content.strip() or None,
                    usage_details=usage_details(u.prompt_tokens, u.completion_tokens, u.total_tokens),
                    metadata={"backend": resp.backend, "finish_reason": resp.finish_reason},
                )
        if resp is None:
            result = GeneralAnswer(ERROR, UNAVAILABLE_MESSAGE, error_reason="llm_unavailable", **base)
        else:
            text = resp.content.strip()
            usage: Usage = resp.usage
            base["model"] = resp.model or base["model"]
            tokens = {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            }
            if text:
                result = GeneralAnswer(ANSWERED, text, tokens=tokens, **base)
            else:  # e.g. the whole completion budget went on reasoning
                result = GeneralAnswer(
                    ERROR, EMPTY_MESSAGE, error_reason="empty_reply", tokens=tokens, **base
                )
            result.timings["llm_ms"] = round(resp.latency_ms, 1)
        result.timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        get_logger().info(
            "answer_general",
            status=result.status,
            error_reason=result.error_reason,
            tokens=result.tokens,
            timings=result.timings,
            prompt_version=result.prompt_version,
            model=result.model,
        )
        return result
