"""Upload checks that need the file on disk. Size is enforced earlier, while the upload streams in."""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from app.config import UploadConfig


class UploadRejected(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def looks_like_pdf(head: bytes) -> bool:
    """The spec allows a little junk before the header, so look in the first KB."""
    return b"%PDF-" in head[:1024]


def validate_pdf(path: str | Path, cfg: UploadConfig) -> int:
    """Return the page count, or raise UploadRejected for a file we won't ingest."""
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise UploadRejected(422, f"Could not read this PDF: {exc}") from exc
    try:
        if doc.needs_pass:
            raise UploadRejected(
                422, "PDF is password protected / encrypted. Remove the password and re-upload."
            )
        pages = doc.page_count
    finally:
        doc.close()
    if pages < 1:
        raise UploadRejected(422, "PDF has no pages.")
    if pages > cfg.max_pages:
        raise UploadRejected(422, f"PDF has {pages} pages; the limit is {cfg.max_pages}.")
    return pages
