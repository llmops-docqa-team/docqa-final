"""A scripted LLM that answers by role, so tests do not depend on the order of concurrent calls.

Roles (decided from the call itself): `router` = the router model, `doc` = the answer model in JSON mode,
`general` = the answer model without JSON mode.
"""

from __future__ import annotations

import threading

from app.config import Settings
from app.llm.client import LLMResponse, Usage


class FakeLLM:
    def __init__(
        self,
        settings: Settings,
        *,
        router=(),
        doc=(),
        general=(),
        barrier: threading.Barrier | None = None,
        usage: Usage | None = None,
    ):
        self.settings = settings
        self.queues = {"router": list(router), "doc": list(doc), "general": list(general)}
        self.calls: list[dict] = []
        self.barrier = barrier  # when set, doc and general calls wait for each other (proves concurrency)
        self.usage = usage or Usage(50, 10, 60)
        self._lock = threading.Lock()

    def role_of(self, model: str | None, json_mode: bool) -> str:
        if model == self.settings.llm.router_model:
            return "router"
        return "doc" if json_mode else "general"

    def calls_for(self, role: str) -> list[dict]:
        return [c for c in self.calls if c["role"] == role]

    def chat(self, messages, *, model=None, json_mode=False, **kw):
        role = self.role_of(model, json_mode)
        with self._lock:
            self.calls.append(
                {"role": role, "messages": list(messages), "model": model, "json_mode": json_mode, "kw": kw}
            )
            queue = self.queues[role]
            assert queue, f"unexpected {role} LLM call"
            reply = queue.pop(0)
        if self.barrier is not None and role in ("doc", "general"):
            self.barrier.wait(timeout=5)  # BrokenBarrierError if the other path never starts
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(content=reply, usage=self.usage, model=f"fake-{role}", latency_ms=5.0)
