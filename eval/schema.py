"""Schema for one row of eval/questions.jsonl."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Route(str, Enum):
    DOCUMENT = "DOCUMENT"
    GENERAL = "GENERAL"
    MIXED = "MIXED"


class Slice(str, Enum):
    TEXT = "text"
    TABLE = "table"
    SCANNED = "scanned"
    GENERAL = "general"
    MIXED = "mixed"


class QType(str, Enum):
    NUMERIC = "numeric"
    NARRATIVE = "narrative"
    EXPLAIN = "explain"
    DEFINITION = "definition"
    NEAR_MISS = "near_miss"
    OTHER = "other"


class GoldPage(BaseModel):
    """One gold page. `page` is the printed label, `pdf_page` the 1-based PDF index."""

    model_config = ConfigDict(extra="forbid")

    doc: str = Field(min_length=1)
    page: int | str
    pdf_page: int | None = Field(default=None, ge=1)


class EvalRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=500)
    route: Route
    slice: Slice
    type: QType = QType.OTHER
    # Document questions: False means the gold outcome is ABSTAIN.
    answerable: bool | None = None
    gold_answer: str | None = None
    gold_pages: list[GoldPage] = Field(default_factory=list)
    # MIXED only
    gold_document_question: str | None = None
    gold_general_question: str | None = None
    gold_general_answer: str | None = None
    # Free text: who wrote / verified the row
    author: str | None = None
    verified_by: str | None = None
    notes: str | None = None

    @field_validator("id", "question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def _route_consistency(self) -> EvalRow:
        if self.route is Route.GENERAL and self.slice is not Slice.GENERAL:
            raise ValueError("GENERAL route requires slice 'general'")
        if self.route is Route.MIXED and self.slice is not Slice.MIXED:
            raise ValueError("MIXED route requires slice 'mixed'")
        if self.route is Route.DOCUMENT and self.slice in (Slice.GENERAL, Slice.MIXED):
            raise ValueError("DOCUMENT route requires slice text/table/scanned")
        if self.route is Route.DOCUMENT and self.answerable is None:
            raise ValueError("DOCUMENT rows must set 'answerable'")
        if self.route is Route.MIXED:
            missing = [
                f
                for f in ("gold_document_question", "gold_general_question", "gold_general_answer")
                if not getattr(self, f)
            ]
            if missing:
                raise ValueError(f"MIXED rows require {', '.join(missing)}")
        return self
