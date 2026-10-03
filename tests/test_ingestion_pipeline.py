"""Step 03: upload API, worker, status flow, crash recovery, embedding guard. Offline: fake embedder,
fake OCR, temp SQLite/Chroma dirs."""
from __future__ import annotations

import sqlite3
import threading

import pymupdf as fitz
import pytest

from app.ingestion.index import INGEST_VERSION, EmbeddingModelMismatch, VectorIndex
from app.ingestion.worker import IngestionWorker
from app.storage import documents as st
from app.storage.db import ADDED_DOCUMENT_COLUMNS, init_db
from app.storage.documents import DocumentStore
from tests.conftest import (
    FakeEmbedder,
    make_pdf,
    make_scanned_pdf,
    make_table_pdf,
    upload,
    wait_for,
)

DONE = (st.READY, st.FAILED)


# ---- upload validation ------------------------------------------------------------------------

def test_upload_ok_indexes_and_reaches_ready(make_client, embedder, settings, tmp_path):
    pdf = tmp_path / "in.pdf"
    make_pdf(pdf, 5)
    with make_client(embedder) as client:
        r = upload(client, pdf, "Annual Report FY25.pdf")
        assert r.status_code == 202
        body = r.json()
        assert body["doc_id"] == body["id"] and body["duplicate"] is False
        assert body["status"] in (st.QUEUED, st.PROCESSING, st.PARTIAL, st.READY)
        assert body["pages_total"] == 5

        doc = wait_for(client, body["id"], DONE)
        assert doc["status"] == st.READY and doc["error"] is None
        assert doc["pages_done"] == doc["pages_total"] == 5
        assert doc["chunks"] >= 5
        assert doc["embedding_model"] == embedder.model_name
        assert doc["ingest_seconds"] is not None and doc["embed_seconds"] is not None
        assert len(doc["page_timings"]) == 5
        assert (settings.upload_dir / f"{doc['id']}.pdf").exists()
        assert client.app.state.index.count(doc["id"]) == doc["chunks"]


def test_upload_rejects_non_pdf(make_client, embedder):
    with make_client(embedder) as client:
        r = upload(client, b"just some text, not a pdf", "notes.pdf")
        assert r.status_code == 415
        assert upload(client, b"", "empty.pdf").status_code == 415
        assert client.get("/documents").json() == []


def test_upload_rejects_corrupt_pdf(make_client, embedder, settings):
    with make_client(embedder) as client:
        r = upload(client, b"%PDF-1.7\nthis is not really a pdf\n%%EOF", "bad.pdf")
        assert r.status_code == 422
        assert client.get("/documents").json() == []
        assert list(settings.upload_dir.iterdir()) == []  # temp file cleaned up


def test_upload_rejects_too_big(make_client, embedder, settings):
    settings.upload.max_mb = 1
    with make_client(embedder) as client:
        r = upload(client, b"%PDF-1.7\n" + b"0" * (1024 * 1024), "big.pdf")
        assert r.status_code == 413
        assert "1 MB" in r.json()["detail"]
        assert list(settings.upload_dir.iterdir()) == []


def test_upload_rejects_too_many_pages(make_client, embedder, settings, tmp_path):
    settings.upload.max_pages = 2
    pdf = tmp_path / "long.pdf"
    make_pdf(pdf, 3)
    with make_client(embedder) as client:
        r = upload(client, pdf)
        assert r.status_code == 422 and "limit is 2" in r.json()["detail"]


def test_upload_rejects_encrypted(make_client, embedder, tmp_path):
    src = tmp_path / "plain.pdf"
    make_pdf(src, 1)
    doc = fitz.open(str(src))
    locked = tmp_path / "locked.pdf"
    doc.save(str(locked), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
    doc.close()
    with make_client(embedder) as client:
        r = upload(client, locked)
        assert r.status_code == 422 and "password" in r.json()["detail"].lower()


def test_upload_filename_is_sanitised(make_client, embedder, tmp_path):
    pdf = tmp_path / "in.pdf"
    make_pdf(pdf, 1)
    with make_client(embedder) as client:
        r = upload(client, pdf, "..\\..\\evil/dir\\Report.pdf")
        assert r.json()["filename"] == "Report.pdf"


# ---- duplicates / retry -----------------------------------------------------------------------

def test_duplicate_hash_returns_existing_doc(make_client, embedder, tmp_path):
    pdf = tmp_path / "in.pdf"
    make_pdf(pdf, 2)
    with make_client(embedder) as client:
        first = upload(client, pdf, "a.pdf").json()
        wait_for(client, first["id"], DONE)
        second = upload(client, pdf, "renamed.pdf")
        assert second.status_code == 200
        assert second.json()["id"] == first["id"] and second.json()["duplicate"] is True
        assert second.json()["filename"] == "a.pdf"
        assert len(client.get("/documents").json()) == 1


def test_reupload_of_failed_doc_retries_it(make_client, settings, tmp_path):
    pdf = tmp_path / "in.pdf"
    make_pdf(pdf, 1, label="BOOM")
    emb = FakeEmbedder(fail_on="BOOM")
    with make_client(emb) as client:
        doc_id = upload(client, pdf).json()["id"]
        assert wait_for(client, doc_id, DONE)["status"] == st.FAILED
        emb.fail_on = None  # the cause was fixed
        r = upload(client, pdf)
        assert r.status_code == 202 and r.json()["id"] == doc_id
        assert wait_for(client, doc_id, DONE)["status"] == st.READY


# ---- read / delete ----------------------------------------------------------------------------

def test_list_get_and_404(make_client, embedder, tmp_path):
    pdf = tmp_path / "in.pdf"
    make_pdf(pdf, 1)
    with make_client(embedder) as client:
        doc_id = upload(client, pdf).json()["id"]
        wait_for(client, doc_id, DONE)
        listed = client.get("/documents").json()
        assert [d["id"] for d in listed] == [doc_id]
        assert "page_timings" not in listed[0]
        assert client.get(f"/documents/{doc_id}").json()["id"] == doc_id
        assert client.get("/documents/nope").status_code == 404


def test_delete_removes_row_file_and_vectors(make_client, embedder, settings, tmp_path):
    pdf = tmp_path / "in.pdf"
    make_pdf(pdf, 3)
    with make_client(embedder) as client:
        doc_id = upload(client, pdf).json()["id"]
        wait_for(client, doc_id, DONE)
        index = client.app.state.index
        assert index.count(doc_id) > 0
        assert client.delete(f"/documents/{doc_id}").status_code == 204
        assert client.get(f"/documents/{doc_id}").status_code == 404
        assert not (settings.upload_dir / f"{doc_id}.pdf").exists()
        assert index.count(doc_id) == 0
        assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_delete_while_processing_stops_the_job_and_leaves_no_vectors(make_client, settings, tmp_path):
    pdf = tmp_path / "in.pdf"
    make_pdf(pdf, 8)
    emb = FakeEmbedder()
    emb.gate, emb.gate_on_call = threading.Event(), 2
    with make_client(emb) as client:
        doc_id = upload(client, pdf).json()["id"]
        assert emb.entered_gate.wait(10)  # worker is inside its 2nd batch
        assert client.delete(f"/documents/{doc_id}").status_code == 204
        emb.gate.set()
        assert client.app.state.worker.wait_idle()
        assert client.app.state.index.count(doc_id) == 0
        assert client.get("/documents").json() == []
        assert not (settings.upload_dir / f"{doc_id}.pdf").exists()


# ---- status flow ------------------------------------------------------------------------------

def test_status_transitions_include_partial(settings, embedder, tmp_path):
    init_db(settings.sqlite_path)
    store = DocumentStore(settings.sqlite_path)
    index = VectorIndex(settings.chroma_dir, embedder.model_name)
    settings.upload_dir.mkdir(parents=True)
    make_pdf(settings.upload_dir / "d1.pdf", 5)
    store.insert("d1", "d1.pdf", "sha", 5)

    seen: list[tuple[str, int]] = []
    real_update = store.update

    def spy(doc_id, **fields):
        if "status" in fields:
            seen.append((fields["status"], fields.get("pages_done", -1)))
        real_update(doc_id, **fields)

    store.update = spy  # type: ignore[method-assign]
    IngestionWorker(settings, store, index, embedder).process("d1")

    statuses = [s for s, _ in seen]
    assert statuses[0] == st.PROCESSING and statuses[-1] == st.READY
    assert statuses.count(st.PARTIAL) == 2          # after pages 2 and 4; page 5 goes straight to READY
    assert [p for s, p in seen if s == st.PARTIAL] == [2, 4]
    assert st.FAILED not in statuses
    done = store.get("d1")
    assert done["pages_done"] == 5 and done["ingest_version"] == INGEST_VERSION


def test_partial_is_queryable_while_processing(make_client, tmp_path):
    pdf = tmp_path / "in.pdf"
    make_pdf(pdf, 6)
    emb = FakeEmbedder()
    emb.gate, emb.gate_on_call = threading.Event(), 2
    with make_client(emb) as client:
        doc_id = upload(client, pdf).json()["id"]
        assert emb.entered_gate.wait(10)
        doc = client.get(f"/documents/{doc_id}").json()
        assert doc["status"] == st.PARTIAL and doc["pages_done"] == 2 < doc["pages_total"]
        assert client.app.state.index.count(doc_id) == doc["chunks"] > 0  # first batch is searchable
        emb.gate.set()
        assert wait_for(client, doc_id, DONE)["status"] == st.READY


def test_page_kinds_metadata_and_batching(make_client, settings, tmp_path):
    settings.embedding.batch_size = 3
    emb = FakeEmbedder()
    table = tmp_path / "table.pdf"
    make_table_pdf(table)
    scan = tmp_path / "scan.pdf"
    make_scanned_pdf(scan, 2)
    with make_client(emb) as client:
        t = upload(client, table).json()["id"]
        s = upload(client, scan).json()["id"]
        t_doc, s_doc = wait_for(client, t, DONE), wait_for(client, s, DONE)

        assert (t_doc["n_text_pages"], t_doc["n_table_pages"], t_doc["n_ocr_pages"]) == (1, 1, 0)
        assert (s_doc["n_text_pages"], s_doc["n_ocr_pages"], s_doc["n_failed_pages"]) == (0, 2, 0)
        assert max(emb.batch_sizes) <= 3

        got = client.app.state.index.collection.get(where={"doc_id": s}, include=["metadatas", "documents"])
        assert {m["source_kind"] for m in got["metadatas"]} == {"ocr"}
        assert all("Scanned report text" in d for d in got["documents"])  # stored text has no header
        meta = got["metadatas"][0]
        assert meta["embedding_model"] == emb.model_name and meta["ingest_version"] == INGEST_VERSION
        assert {"doc_id", "filename", "page", "page_label", "chunk_idx", "char_len"} <= meta.keys()

        kinds = client.app.state.index.collection.get(where={"doc_id": t}, include=["metadatas"])["metadatas"]
        assert "table" in {m["source_kind"] for m in kinds}


def test_deterministic_ids_reruns_overwrite(settings, embedder):
    init_db(settings.sqlite_path)
    store = DocumentStore(settings.sqlite_path)
    index = VectorIndex(settings.chroma_dir, embedder.model_name)
    settings.upload_dir.mkdir(parents=True)
    make_pdf(settings.upload_dir / "d1.pdf", 4)
    store.insert("d1", "d1.pdf", "sha", 4)
    worker = IngestionWorker(settings, store, index, embedder)
    worker.process("d1")
    first = sorted(index.collection.get(where={"doc_id": "d1"}, include=[])["ids"])
    worker.process("d1")
    assert sorted(index.collection.get(where={"doc_id": "d1"}, include=[])["ids"]) == first
    assert first[0].startswith("d1:1:")


# ---- failures ---------------------------------------------------------------------------------

def test_worker_survives_a_failing_doc(make_client, tmp_path):
    bad, good = tmp_path / "bad.pdf", tmp_path / "good.pdf"
    make_pdf(bad, 3, label="BOOM")
    make_pdf(good, 3, label="fine")
    with make_client(FakeEmbedder(fail_on="BOOM")) as client:
        bad_id = upload(client, bad, "bad.pdf").json()["id"]
        good_id = upload(client, good, "good.pdf").json()["id"]
        bad_doc, good_doc = wait_for(client, bad_id, DONE), wait_for(client, good_id, DONE)

        assert bad_doc["status"] == st.FAILED
        assert "embedder exploded" in bad_doc["error"] and bad_doc["ingest_seconds"] is not None
        assert client.app.state.index.count(bad_id) == 0   # not left half-searchable
        assert good_doc["status"] == st.READY              # the worker kept going


def test_unexpected_error_in_process_does_not_kill_the_thread(make_client, embedder, tmp_path, monkeypatch):
    pdf1, pdf2 = tmp_path / "1.pdf", tmp_path / "2.pdf"
    make_pdf(pdf1, 1, label="one")
    make_pdf(pdf2, 1, label="two")
    with make_client(embedder) as client:
        worker = client.app.state.worker
        real = worker.process
        calls = []

        def flaky(doc_id):
            calls.append(doc_id)
            if len(calls) == 1:
                raise RuntimeError("a bug outside process()'s own error handling")
            real(doc_id)

        monkeypatch.setattr(worker, "process", flaky)
        a = upload(client, pdf1).json()["id"]
        worker.wait_idle()
        b = upload(client, pdf2).json()["id"]
        assert wait_for(client, b, DONE)["status"] == st.READY
        assert client.get(f"/documents/{a}").json()["status"] in (st.QUEUED,)  # never ran, thread alive


def test_all_scanned_pages_failing_ocr_marks_failed(make_client, embedder, tmp_path):
    scan = tmp_path / "scan.pdf"
    make_scanned_pdf(scan, 3)

    def broken_ocr(page, cfg):
        raise RuntimeError("tesseract is not installed")

    with make_client(embedder, ocr_fn=broken_ocr) as client:
        doc_id = upload(client, scan).json()["id"]
        doc = wait_for(client, doc_id, DONE)
        assert doc["status"] == st.FAILED
        assert "3 of 3 pages could not be read" in doc["error"] and "tesseract" in doc["error"]


def test_a_few_unreadable_pages_is_ready_with_a_warning(make_client, embedder, tmp_path):
    mixed = tmp_path / "mixed.pdf"
    make_pdf(mixed, 4)
    # Append one image-only page, so 1 of 5 pages needs OCR and fails.
    src = tmp_path / "scan.pdf"
    make_scanned_pdf(src, 1)
    doc = fitz.open(str(mixed))
    doc.insert_pdf(fitz.open(str(src)))
    both = tmp_path / "both.pdf"
    doc.save(str(both))
    doc.close()

    def broken_ocr(page, cfg):
        raise RuntimeError("ocr down")

    with make_client(embedder, ocr_fn=broken_ocr) as client:
        doc_id = upload(client, both).json()["id"]
        d = wait_for(client, doc_id, DONE)
        assert d["status"] == st.READY and d["n_failed_pages"] == 1 and d["n_text_pages"] == 4
        assert "1 of 5 pages could not be read" in d["error"]


def test_pdf_without_any_text_fails_readably(make_client, embedder, tmp_path):
    blank = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(blank))
    doc.close()
    with make_client(embedder) as client:
        doc_id = upload(client, blank).json()["id"]
        d = wait_for(client, doc_id, DONE)
        assert d["status"] == st.FAILED and "no text" in d["error"]


# ---- crash recovery ---------------------------------------------------------------------------

def test_requeue_on_startup(make_client, settings, embedder):
    init_db(settings.sqlite_path)
    store = DocumentStore(settings.sqlite_path)
    settings.upload_dir.mkdir(parents=True)
    for doc_id, status in [("qd", st.QUEUED), ("pr", st.PROCESSING), ("pa", st.PARTIAL), ("rd", st.READY)]:
        make_pdf(settings.upload_dir / f"{doc_id}.pdf", 3, label=doc_id)
        store.insert(doc_id, f"{doc_id}.pdf", doc_id, 3)
        store.update(doc_id, status=status, pages_done=2, chunks=2)

    # A previous run left a stale vector for the PARTIAL doc (e.g. from a different chunking config).
    stale = VectorIndex(settings.chroma_dir, embedder.model_name)
    stale.collection.upsert(ids=["pa:99:0"], embeddings=[[0.1] * 16], documents=["stale"],
                            metadatas=[{"doc_id": "pa"}])

    with make_client(embedder) as client:
        for doc_id in ("qd", "pr", "pa"):
            d = wait_for(client, doc_id, DONE)
            assert d["status"] == st.READY and d["pages_done"] == 3
        ids = client.app.state.index.collection.get(where={"doc_id": "pa"}, include=[])["ids"]
        assert "pa:99:0" not in ids and len(ids) == client.get("/documents/pa").json()["chunks"]
        assert client.get("/documents/rd").json()["chunks"] == 2   # READY docs are left alone


def test_requeued_doc_with_missing_file_fails_cleanly(make_client, settings, embedder):
    init_db(settings.sqlite_path)
    store = DocumentStore(settings.sqlite_path)
    store.insert("ghost", "ghost.pdf", "sha", 3)
    store.update("ghost", status=st.PROCESSING)
    with make_client(embedder) as client:
        d = wait_for(client, "ghost", DONE)
        assert d["status"] == st.FAILED and "missing" in d["error"]


# ---- embedding guard --------------------------------------------------------------------------

def test_embedding_model_mismatch_refuses_to_start(make_client, settings):
    VectorIndex(settings.chroma_dir, "model-a")
    VectorIndex(settings.chroma_dir, "model-a")  # same model: fine, reopening is allowed
    with pytest.raises(EmbeddingModelMismatch) as err:
        with make_client(FakeEmbedder("model-b")):
            pass
    msg = str(err.value)
    assert "model-a" in msg and "model-b" in msg and "re-upload" in msg


def test_index_uses_cosine_space(settings):
    index = VectorIndex(settings.chroma_dir, "m")
    assert index.collection.metadata["hnsw:space"] == "cosine"
    assert index.collection.metadata["embedding_model"] == "m"


# ---- schema migration -------------------------------------------------------------------------

def test_init_db_adds_new_columns_to_an_old_database(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL, "
                 "status TEXT NOT NULL DEFAULT 'QUEUED', pages_total INTEGER, pages_done INTEGER NOT NULL "
                 "DEFAULT 0, chunks INTEGER NOT NULL DEFAULT 0, ingest_seconds REAL, error TEXT, "
                 "created_at TEXT NOT NULL DEFAULT 'x', updated_at TEXT NOT NULL DEFAULT 'x')")
    conn.execute("INSERT INTO documents (id, filename, sha256) VALUES ('old', 'o.pdf', 's')")
    conn.commit()
    conn.close()
    init_db(path)
    init_db(path)  # idempotent
    doc = DocumentStore(path).get("old")
    assert set(ADDED_DOCUMENT_COLUMNS) <= doc.keys() and doc["filename"] == "o.pdf"

