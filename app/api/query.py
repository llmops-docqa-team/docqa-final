"""POST /query: route the question, answer from documents and/or general knowledge, compose one response."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["query"])


class QueryIn(BaseModel):
    question: str
    company: str | None = None  # limit the document search to this catalog company (the UI dropdown)
    enhance: bool = True  # False: search the question as typed (no rewrite, no period narrowing)


@router.post("/query")
def query(body: QueryIn, request: Request):
    """A sync endpoint: FastAPI runs it in a worker thread, so blocking LLM calls do not stall the loop."""
    limit = request.app.state.settings.query.max_question_chars
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="The question is empty.")
    if len(question) > limit:
        raise HTTPException(
            status_code=422,
            detail=f"The question is too long ({len(question)} characters; the limit is {limit}). "
            "Please shorten it.",
        )
    company = (body.company or "").strip() or None
    return request.app.state.query_service.run(
        question, request.state.request_id, company=company, enhance=body.enhance
    )
