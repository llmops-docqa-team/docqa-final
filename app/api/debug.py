"""Debug endpoints. Hidden (404) unless `api.debug_endpoints` is true or the FINCHAT_DEBUG env var is set:
they have no auth, and /debug/answer_doc spends LLM quota. The real entry point is POST /query.

POST /debug/retrieve    what retrieval returns for a question (no LLM involved)
POST /debug/answer_doc  retrieve + grounded document answer, or an abstention (one LLM call, no router)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field


def require_debug(request: Request) -> None:
    if not request.app.state.settings.api.debug_endpoints:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(prefix="/debug", tags=["debug"], dependencies=[Depends(require_debug)])


class RetrieveIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)  # default: retrieval.fetch_k
    doc_ids: list[str] | None = None


class AnswerDocIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    doc_ids: list[str] | None = None


@router.post("/retrieve")
def debug_retrieve(body: RetrieveIn, request: Request):
    result = request.app.state.retriever.retrieve(body.question, body.top_k, body.doc_ids)
    return result.to_dict()


@router.post("/answer_doc")
def debug_answer_doc(body: AnswerDocIn, request: Request):
    # A sync endpoint: FastAPI runs it in a worker thread, so the blocking LLM call does not stall the loop.
    return request.app.state.doc_answerer.answer(body.question, doc_ids=body.doc_ids).to_dict()
