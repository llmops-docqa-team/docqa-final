# 02: Ingestion Part 1, Parsing & Chunking

**Effort:** high · **Prereqs:** 00 · **Design refs:** §8, §10, addendum §2, §4

Pure, testable functions only: PDF in, chunks out. No API, no DB, no embeddings (that's step 03).

````text
Read docs/design.md (§8, §10) and docs/addendum.md (§2, §4), plus docs/progress.md.

Task: build the parsing + chunking layer as pure functions with good unit tests.
No HTTP endpoints, no SQLite, no embeddings, no Chroma yet.

Build (module names are suggestions):
- app/ingestion/pdf_parse.py
  - Open a PDF with PyMuPDF and yield one record per page: pdf_page index (1-based), printed page label
    (from PDF page labels if present, else fall back to the index), text, image-area ratio, and source_kind.
  - Scanned-page detection per page: text under ~120 chars AND images covering >= ~60% of the page -> OCR.
    Thresholds come from config.
  - OCR fallback via pytesseract on a rendered page image (a sensible DPI like 200-300, configurable).
    If tesseract isn't installed, fail clearly for that page (don't crash the whole doc in a confusing way).
  - Strip repeated headers/footers: lines that appear on > ~50% of pages.
- app/ingestion/tables.py
  - Use PyMuPDF page.find_tables() to extract tables -> Markdown.
  - Each table becomes its own chunk. Long tables are split by rows with the header row repeated in every piece.
  - Remove table regions from the page's plain text if practical, to avoid duplicates (best effort; note it if skipped).
- app/ingestion/chunking.py
  - Recursive split WITHIN a page (paragraph -> sentence -> word), target ~400 tokens, ~60 overlap (from config).
    A cheap token estimate is fine; it just has to stay safely under bge-small's 512-token limit.
  - Never cross page boundaries.
  - Chunk record: id, doc_id, filename, page (pdf index), page_label, chunk_idx, source_kind (text/table/ocr),
    text (stored as-is), embed_text (text with a "{doc title} — page {label}" header prefix), char_len.
  - Deterministic IDs, e.g. "{doc_id}:{page}:{i}" (tables can use a "t" prefix on i). Re-running yields the same IDs.
- Test fixtures: generate small PDFs inside the tests with PyMuPDF (a text page, a page with a table,
  an image-only page made by rasterising a text page). Don't commit real reports.

Tests (at least): chunk size bound and overlap; no chunk crosses pages; deterministic IDs; page labels
fall back correctly; scanned detection true/false cases; table -> Markdown with repeated header on split;
header/footer stripping. Mark the OCR test to skip if tesseract isn't available locally.

Also add a small dev script (scripts/parse_preview.py <pdf>) that prints per-page kind, chunk counts and a few
sample chunks, useful for eyeballing real reports.

When done: run tests, append a "Step 02" entry to docs/progress.md (incl. any deviations), and stop.
````
