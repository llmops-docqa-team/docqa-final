"""Short reason codes for failed uploads, so the Metrics page can count them ("upload failures by reason").

Two kinds of failure: an upload refused at the door (HTTP 4xx, recorded in `upload_failures`) and a document
that was accepted but whose ingestion ended FAILED (`documents.error`). Messages are matched by their wording
in app/api/documents.py, app/ingestion/validate.py and app/ingestion/worker.py; anything unrecognised is
"rejected" / "processing_error" rather than lost.
"""

from __future__ import annotations


def classify_upload_rejection(status_code: int, detail: str) -> str:
    text = (detail or "").lower()
    if status_code == 415:
        return "not_a_pdf"
    if status_code == 413:
        return "too_large"
    if "password" in text or "encrypted" in text:
        return "encrypted"
    if "limit is" in text and "pages" in text:
        return "too_many_pages"
    if "no pages" in text:
        return "empty_pdf"
    if "could not read" in text:
        return "unreadable"
    return "rejected"


def classify_ingest_failure(error: str | None) -> str:
    text = (error or "").lower()
    if "no text could be extracted" in text:
        return "no_text"
    if "could not be read" in text:
        return "pages_unreadable"
    return "processing_error"
