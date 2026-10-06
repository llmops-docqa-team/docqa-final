"""POST /feedback: 👍 / 👎 on an answer, stored on the request's row."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.observability.logging import get_logger
from app.observability.tracing import get_tracer

router = APIRouter(tags=["feedback"])
log = get_logger("api.feedback")


class FeedbackIn(BaseModel):
    trace_id: str = Field(min_length=1, max_length=128)
    value: Literal[1, -1]  # 1 = 👍, -1 = 👎


@router.post("/feedback")
def feedback(body: FeedbackIn, request: Request):
    request.app.state.request_store.set_feedback(body.trace_id, body.value)
    get_tracer().score(body.trace_id, "user_feedback", body.value)  # queued; a no-op without Langfuse
    log.info("feedback", trace_id=body.trace_id, value=body.value)
    return {"trace_id": body.trace_id, "value": body.value}
