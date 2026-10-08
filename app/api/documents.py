"""POST/GET/DELETE /documents."""
from __future__ import annotations

import hashlib
import re
import uuid
from functools import lru_cache
from pathlib import Path as FsPath
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response, UploadFile
from pydantic import BaseModel

from app.answering.highlight import MAX_TERMS, TERM_PATTERN, PageOutOfRange, render_highlight
from app.ingestion.validate import UploadRejected, looks_like_pdf, validate_pdf
from app.ingestion.worker import upload_path
from app.observability.failure_reasons import classify_upload_rejection
from app.observability.logging import get_logger
from app.storage import documents as st

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger("api.documents")

_READ_CHUNK = 1024 * 1024


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    pages_total: int | None = None
    pages_done: int = 0
    chunks: int = 0
    error: str | None = None
    ingest_seconds: float | None = None
    embed_seconds: float | None = None
    embedding_model: str | None = None
    n_text_pages: int | None = None
    n_table_pages: int | None = None
    n_ocr_pages: int | None = None
    n_failed_pages: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    company: str | None = None
    report_type: str | None = None
    period: str | None = None


class DocumentDetail(DocumentOut):
    page_timings: list[float] | None = None


class UploadResponse(DocumentOut):
    doc_id: str
    duplicate: bool = False


def _safe_filename(name: str | None) -> str:
    base = re.split(r"[\\/]", name or "")[-1].strip()
    return re.sub(r"[\x00-\x1f]", "", base)[:200] or "document.pdf"


def _stream_to_disk(file: UploadFile, dest: FsPath, max_bytes: int) -> str:
    """Copy the upload to `dest` while hashing it. Rejects non-PDFs and oversize files early."""
    sha = hashlib.sha256()
    size = 0
    first = True
    with dest.open("wb") as out:
        while chunk := file.file.read(_READ_CHUNK):
            if first:
                if not looks_like_pdf(chunk):
                    raise UploadRejected(415, "Not a PDF: the file does not start with a PDF header.")
                first = False
            size += len(chunk)
            if size > max_bytes:
                raise UploadRejected(413, f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.")
            sha.update(chunk)
            out.write(chunk)
    if first:
        raise UploadRejected(415, "Not a PDF: the file is empty.")
    return sha.hexdigest()


@router.post("", status_code=202, response_model=UploadResponse)
def upload_document(request: Request, response: Response, file: UploadFile):
    app = request.app
    settings, store, worker = app.state.settings, app.state.store, app.state.worker
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.upload_dir / f".upload-{uuid.uuid4().hex}.tmp"
    try:
        sha = _stream_to_disk(file, tmp, settings.upload.max_mb * 1024 * 1024)

        existing = store.get_by_sha(sha)
        if existing and existing["status"] != st.FAILED:
            response.status_code = 200
            return _upload_out(existing, duplicate=True)

        pages = validate_pdf(tmp, settings.upload)
        doc_id = existing["id"] if existing else sha[:16]
        tmp.replace(upload_path(settings, doc_id))
        if existing:  # same file, earlier attempt FAILED: run it again instead of leaving it dead
            store.reset_for_processing(doc_id, st.QUEUED)
            store.update(doc_id, pages_total=pages)
        elif not store.insert(doc_id, _safe_filename(file.filename), sha, pages):
            # Two identical uploads raced; the other request won the insert.
            return _upload_out(store.get(doc_id), duplicate=True)
    except UploadRejected as exc:
        _record_rejection(app, exc)
        raise HTTPException(exc.status_code, exc.detail) from exc
    finally:
        tmp.unlink(missing_ok=True)

    worker.submit(doc_id)
    log.info("document_queued", doc_id=doc_id, filename=file.filename, pages=pages)
    return _upload_out(store.get(doc_id), duplicate=bool(existing))


def _record_rejection(app, exc: UploadRejected) -> None:
    """Count the refusal for the Metrics page (reason code only, no file name). Never fails the request."""
    try:
        reason = classify_upload_rejection(exc.status_code, exc.detail)
        app.state.request_store.log_upload_failure(exc.status_code, reason)
    except Exception as err:  # noqa: BLE001
        log.warning("upload_failure_log_failed", error_type=type(err).__name__)


def _upload_out(doc: dict, *, duplicate: bool) -> UploadResponse:
    return UploadResponse(**{**doc, "doc_id": doc["id"], "duplicate": duplicate})


@router.get("", response_model=list[DocumentOut])
def list_documents(request: Request):
    return request.app.state.store.list()


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str, request: Request):
    doc = request.app.state.store.get(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found.")
    return doc


@lru_cache(maxsize=64)
def _highlight_png(pdf_path: str, pdf_page: int, terms: tuple[str, ...], context: str) -> tuple[bytes, int]:
    return render_highlight(pdf_path, pdf_page, list(terms), context=context)


@router.get("/{doc_id}/pages/{pdf_page}/highlight")
def page_highlight(
    doc_id: str,
    pdf_page: Annotated[int, Path(ge=1)],
    request: Request,
    term: Annotated[list[str], Query(max_length=MAX_TERMS)] = [],  # noqa: B006 (FastAPI reads, never mutates)
    ctx: Annotated[str, Query(max_length=600)] = "",
):
    """PNG of one PDF page, cropped around the rows holding the `term` figures (see answering/highlight.py).
    The file is the stored upload of `doc_id`; nothing in the request names a path."""
    doc = request.app.state.store.get(doc_id)
    pdf = upload_path(request.app.state.settings, doc["id"]) if doc else None
    if pdf is None or not pdf.is_file():
        raise HTTPException(404, "Document not found.")
    if not all(TERM_PATTERN.fullmatch(t) for t in term):
        raise HTTPException(422, "A term holds only digits, letters and , . % ( ) -, 32 characters at most.")
    try:
        png, matches = _highlight_png(str(pdf), pdf_page, tuple(term), ctx)
    except PageOutOfRange as exc:
        raise HTTPException(422, f"No such page. {exc}") from exc
    return Response(
        png,
        media_type="image/png",
        headers={"X-Highlight-Matches": str(matches), "Cache-Control": "private, max-age=3600"},
    )


class DocumentPatch(BaseModel):
    """Correct the catalog entry the file name gave (e.g. company 'EIG' -> 'Ellenbarrie Industrial Gases')."""

    company: str | None = None
    report_type: str | None = None
    period: str | None = None


@router.patch("/{doc_id}", response_model=DocumentOut)
def patch_document(doc_id: str, body: DocumentPatch, request: Request):
    from app.catalog import normalize_period

    store = request.app.state.store
    if store.get(doc_id) is None:
        raise HTTPException(404, "Document not found.")
    fields = {k: " ".join(v.split())[:120] for k, v in body.model_dump(exclude_none=True).items()}
    if "period" in fields and fields["period"]:
        period = normalize_period(fields["period"])
        if period is None:
            raise HTTPException(422, "Period not recognised. Use FY26 or Q1 FY26.")
        fields["period"] = period
    if fields:
        store.update(doc_id, **{k: v or None for k, v in fields.items()})
        log.info("document_catalog_updated", doc_id=doc_id, fields=sorted(fields))
    return store.get(doc_id)


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str, request: Request):
    app = request.app
    store, worker, index, settings = app.state.store, app.state.worker, app.state.index, app.state.settings
    if store.get(doc_id) is None:
        raise HTTPException(404, "Document not found.")
    worker.cancel(doc_id)  # a job in flight stops at its next page and cleans up after itself
    store.delete(doc_id)
    try:
        upload_path(settings, doc_id).unlink(missing_ok=True)
    except OSError:
        pass  # Windows: the worker still has it open; it removes the file when it stops (see worker)
    index.delete_doc(doc_id)
    log.info("document_deleted", doc_id=doc_id)
    return Response(status_code=204)
