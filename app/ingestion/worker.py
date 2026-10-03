"""Background ingestion: one worker thread, one in-process queue, one document at a time.

QUEUED -> PROCESSING -> PARTIAL (first batch indexed; queryable) -> READY, or FAILED(reason).
Re-running a job is safe: its old vectors are removed first and chunk IDs are deterministic.
"""
from __future__ import annotations

import contextlib
import queue
import threading
import time
from pathlib import Path

from app.config import Settings
from app.ingestion.chunking import Chunk, chunk_page, title_from_filename
from app.ingestion.embedder import Embedder
from app.ingestion.index import INGEST_VERSION, VectorIndex
from app.ingestion.pdf_parse import OcrFn, PageRecord, parse_pdf
from app.observability.logging import get_logger
from app.storage import documents as st
from app.storage.documents import DocumentStore

log = get_logger("ingestion.worker")

_STOP = object()


class _Cancelled(Exception):
    """The document was deleted while it was being processed."""


class IngestionWorker:
    def __init__(
        self,
        settings: Settings,
        store: DocumentStore,
        index: VectorIndex,
        embedder: Embedder,
        *,
        ocr_fn: OcrFn | None = None,
    ):
        self.settings = settings
        self.store = store
        self.index = index
        self.embedder = embedder
        self.ocr_fn = ocr_fn
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    # ---- lifecycle ------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="ingestion-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the worker to exit after the current job. A job still running past `timeout` is left to
        the crash-recovery path on the next start."""
        if self._thread:
            self._queue.put(_STOP)
            self._thread.join(timeout)

    def submit(self, doc_id: str) -> None:
        with self._lock:
            self._cancelled.discard(doc_id)
        self._queue.put(doc_id)

    def cancel(self, doc_id: str) -> None:
        """Stop working on a document that is being deleted (takes effect at the next page)."""
        with self._lock:
            self._cancelled.add(doc_id)

    def requeue_unfinished(self) -> list[str]:
        """Crash recovery: anything left QUEUED/PROCESSING/PARTIAL starts over from scratch."""
        ids = self.store.unfinished_ids()
        for doc_id in ids:
            self.store.reset_for_processing(doc_id, st.QUEUED)
            self.submit(doc_id)
        if ids:
            log.info("requeued_unfinished", doc_ids=ids)
        return ids

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until the queue is drained (used by tests and scripts)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return False

    # ---- loop -----------------------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self.process(item)
            except Exception:  # last line of defence: nothing a job does may kill the worker
                log.exception("worker_unexpected_error", doc_id=item)
            finally:
                self._queue.task_done()

    def process(self, doc_id: str) -> None:
        """Run one job. Never raises for a bad document: it is marked FAILED instead."""
        doc = self.store.get(doc_id)
        if doc is None:  # deleted while queued
            return
        started = time.perf_counter()
        try:
            self._run(doc)
        except _Cancelled:
            self.index.delete_doc(doc_id)
            # DELETE may have been unable to remove the file while we had it open.
            with contextlib.suppress(OSError):
                upload_path(self.settings, doc_id).unlink(missing_ok=True)
            log.info("ingest_cancelled", doc_id=doc_id)
        except Exception as exc:
            log.exception("ingest_failed", doc_id=doc_id, filename=doc["filename"])
            self._fail(doc_id, _readable(exc), time.perf_counter() - started)

    def _fail(self, doc_id: str, message: str, seconds: float) -> None:
        try:
            self.index.delete_doc(doc_id)  # a FAILED document must not stay half-searchable
            self.store.update(doc_id, status=st.FAILED, error=message, ingest_seconds=round(seconds, 2))
        except Exception:
            log.exception("could_not_record_failure", doc_id=doc_id)

    # ---- one document ---------------------------------------------------------------------------
    def _run(self, doc: dict) -> None:
        cfg = self.settings
        doc_id, filename = doc["id"], doc["filename"]
        path = upload_path(cfg, doc_id)
        if not path.exists():
            raise FileNotFoundError("the uploaded file is missing from disk; upload it again")

        started = time.perf_counter()
        self.index.delete_doc(doc_id)  # start clean so chunks from an older run/config can't linger
        self.store.reset_for_processing(doc_id, st.PROCESSING)
        title = title_from_filename(filename)
        flush_every = max(1, cfg.ingestion.upsert_every_pages)

        buffered: list[Chunk] = []
        timings: dict[int, float] = {}  # pdf_page -> seconds; pages arrive text-first, OCR last
        kinds = {"text": 0, "ocr": 0, "table": 0, "failed": 0}
        first_error: str | None = None
        n_chunks = 0
        embed_seconds = 0.0
        pages_done = 0

        def flush(final: bool = False) -> None:
            nonlocal buffered, n_chunks, embed_seconds
            if buffered:
                t0 = time.perf_counter()
                vectors = self._embed(buffered)
                embed_seconds += time.perf_counter() - t0
                self.index.upsert(buffered, vectors)
                n_chunks += len(buffered)
                buffered = []
            if self._is_cancelled(doc_id):
                raise _Cancelled
            if final:  # the READY update below records the totals
                return
            # PARTIAL means "queryable": the first batch is in the index. READY is set after the loop.
            self.store.update(
                doc_id,
                status=st.PARTIAL if n_chunks else st.PROCESSING,
                pages_done=pages_done,
                chunks=n_chunks,
                embed_seconds=round(embed_seconds, 2),
            )

        pages = parse_pdf(path, cfg.ocr, cfg.parsing, ocr_fn=self.ocr_fn)
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    record = next(pages)
                except StopIteration:
                    break
                page_chunks = self._chunk(record, doc_id, filename, title)
                # Per-page time = parse (+OCR) + chunking. The first page yielded also carries the
                # header/footer pre-pass.
                timings[record.pdf_page] = round(time.perf_counter() - t0, 3)

                if record.error:
                    kinds["failed"] += 1
                    first_error = first_error or record.error
                else:
                    kinds[record.source_kind] += 1
                    kinds["table"] += bool(record.tables)
                buffered.extend(page_chunks)
                pages_done += 1
                if self._is_cancelled(doc_id):
                    raise _Cancelled
                if pages_done % flush_every == 0:
                    flush()
            flush(final=True)
        finally:
            pages.close()  # releases the PDF file handle (Windows can't delete a file that is still open)

        failed_ratio = kinds["failed"] / pages_done if pages_done else 0.0
        if pages_done == 0:
            raise ValueError("the PDF has no readable pages")
        if failed_ratio > cfg.ingestion.max_failed_page_ratio:
            raise RuntimeError(
                f"{kinds['failed']} of {pages_done} pages could not be read ({first_error})"
            )
        if n_chunks == 0:
            raise RuntimeError("no text could be extracted from this PDF")

        # Non-fatal page errors don't fail the document; they are kept as a visible warning.
        warning = (
            f"{kinds['failed']} of {pages_done} pages could not be read and are not searchable "
            f"({first_error})"
            if kinds["failed"]
            else None
        )
        seconds = time.perf_counter() - started
        self.store.update(
            doc_id,
            status=st.READY,
            pages_total=pages_done,
            pages_done=pages_done,
            chunks=n_chunks,
            error=warning,
            ingest_seconds=round(seconds, 2),
            embed_seconds=round(embed_seconds, 2),
            embedding_model=self.embedder.model_name,
            ingest_version=INGEST_VERSION,
            n_text_pages=kinds["text"],
            n_table_pages=kinds["table"],
            n_ocr_pages=kinds["ocr"],
            n_failed_pages=kinds["failed"],
            page_timings=[timings[p] for p in sorted(timings)],
        )
        log.info(
            "ingest_done", doc_id=doc_id, filename=filename, pages=pages_done, chunks=n_chunks,
            text_pages=kinds["text"], table_pages=kinds["table"], ocr_pages=kinds["ocr"],
            failed_pages=kinds["failed"], seconds=round(seconds, 2),
            embed_seconds=round(embed_seconds, 2),
            seconds_per_page=round(seconds / pages_done, 3),
        )

    # ---- helpers --------------------------------------------------------------------------------
    def _chunk(self, record: PageRecord, doc_id: str, filename: str, title: str) -> list[Chunk]:
        if record.error:
            return []
        return chunk_page(
            record, doc_id=doc_id, filename=filename, doc_title=title, cfg=self.settings.chunking
        )

    def _embed(self, chunks: list[Chunk]) -> list[list[float]]:
        size = max(1, self.settings.embedding.batch_size)
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), size):
            vectors.extend(self.embedder.embed_documents([c.embed_text for c in chunks[i : i + size]]))
        return vectors

    def _is_cancelled(self, doc_id: str) -> bool:
        with self._lock:
            return doc_id in self._cancelled


def _readable(exc: Exception) -> str:
    msg = str(exc).strip() or type(exc).__name__
    if isinstance(exc, (FileNotFoundError, ValueError, RuntimeError)):
        out = msg
    else:
        out = f"{type(exc).__name__}: {msg}"
    return out[:500]


def upload_path(settings: Settings, doc_id: str) -> Path:
    return settings.upload_dir / f"{doc_id}.pdf"
