# 03: Ingestion Part 2, Pipeline, Worker, Index

**Effort:** high · **Prereqs:** 02 · **Design refs:** §8, §10 (embedding guard), addendum §2 (PARTIAL status), §5 Prompt B

````text
Read docs/design.md (§8, §10) and docs/addendum.md, plus docs/progress.md. Reuse the parsing/chunking from step 02.

Task: wire the async ingestion pipeline end to end. No retrieval, router or LLM calls yet.

Build:
- POST /documents (multipart): validate (is a PDF, <= 25 MB, <= 400 pages, not encrypted; limits from config),
  compute SHA-256 (duplicate hash -> return the existing doc), save to data/uploads/, insert a documents row
  with status QUEUED, return 202 + doc_id.
- GET /documents (list with status, pages_done/pages_total, chunks, error) and GET /documents/{id}.
- Status flow: QUEUED -> PROCESSING -> PARTIAL (once the first batch is indexed; queryable) -> READY, or FAILED(reason).
- One background worker thread started with the app (FastAPI lifespan), fed by an in-process queue.
  On startup, re-queue any doc left in PROCESSING/PARTIAL (crash recovery).
  Errors on one doc mark it FAILED with a readable message and never kill the worker.
- Embeddings: BAAI/bge-small-en-v1.5 via fastembed, batches of ~32, embed the chunk's embed_text.
  Keep the embedder behind a tiny interface so tests can use a fake embedder.
- Chroma persistent collection (cosine). Upsert every ~20 pages (configurable) and update pages_done.
  Store chunk metadata from step 02 + embedding_model + ingest_version. Deterministic IDs mean re-runs overwrite.
- Embedding-model guard: store the model name in collection metadata; on startup refuse to serve
  (clear error) if config's model != index's model.
- Optional but handy: DELETE /documents/{id} (removes the row, file and vectors).
- Record ingest_seconds and per-page timing; log the counts of text/table/ocr pages.

Tests: upload validation (bad type, too big, encrypted), duplicate hash, status transitions incl. PARTIAL,
re-queue on startup, worker survives a failing doc, embedding guard mismatch. Use a fake embedder and
a temp Chroma/SQLite dir so tests are fast and offline.

Then measure: ingest one text PDF and one scanned PDF (rasterise a text PDF if no real scan is at hand)
and print seconds per page for each. Put the numbers in progress.md.

When done: run tests, append a "Step 03" entry to docs/progress.md, and stop.
````
