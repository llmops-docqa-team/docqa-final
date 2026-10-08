# Progress log

Hand-off log between sessions and teammates. Newest entries at the bottom.

## Step 00: Repo skeleton

**Built**
- Layout: `app/` (api, ingestion, retrieval, answering, routing, llm, storage, observability), `ui/`, `prompts/`, `eval/`, `scripts/`, `tests/`. Only `api`, `storage`, `observability` and `config.py` have code; the rest are empty packages.
- `config.yaml` holds all tunables (models, base URL, chunk size/overlap, fetch_k/top_k/theta, upload limits, upsert batch, OCR thresholds, paths). `app/config.py` is a typed Pydantic loader. Env vars read by name: `GROQ_API_KEY`, `GEMINI_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `DOCQA_CONFIG`. `.env.example` has no values.
- `prompts/{router,answer_doc,answer_general,judge}_v1.yaml`: stubs with a `version` field.
- SQLite (`app/storage/db.py`): `documents` and `requests` tables (design §15 fields), create-if-not-exists, run on API startup.
- structlog JSON logging and `RequestIDMiddleware` (honours/echoes `X-Request-ID`, logs one line per request).
- FastAPI `GET /health`; Streamlit page (`ui/app.py`) that calls it.
- Dockerfile (python:3.11-slim + tesseract-ocr), `docker-compose.yml` (api + ui, `./data` mounted), GitHub Actions (ruff + pytest on Python 3.11).
- Tests: `tests/test_smoke.py` (config, env overrides, tables, health + request ID). 4 passed, ruff clean.

**Run locally (no Docker)**
```
pip install -r requirements-dev.txt
python -m uvicorn app.main:app --port 8000
streamlit run ui/app.py            # second terminal; DOCQA_API_URL defaults to http://localhost:8000
python -m pytest -q
```

**Run with Docker**
```
cp .env.example .env               # fill keys if needed
docker compose up --build          # API :8000, UI :8501
```

**Deviations / notes**
- `requirements.txt` + `requirements-dev.txt` with major-version pins, no pyproject dependency table. Only skeleton deps are listed; later steps add PyMuPDF, fastembed, chromadb, openai, pytesseract, etc.
- Judge model in `config.yaml` is `gemini-2.5-flash` (design only says "Gemini Flash"); confirm before step 09.
- `theta: 0.5` is a placeholder until the step 10 sweep.
- Verified: `docker compose up --build` works. API healthy, UI serves on :8501, UI container reaches `http://api:8000/health`, tesseract 5.5.0 is in the image, `data/docqa.sqlite` is created on the mounted volume.
- Nothing was reused from Sifra-v2.

**Open TODOs**
- None.

## Step 01: Eval set tooling

**Built**
- `eval/schema.py`: Pydantic `EvalRow` (id, question, route, slice, type, answerable, gold_answer, gold_pages `{doc, page (printed label, int or str), pdf_page}`, MIXED gold fields, optional author/verified_by/notes). Unknown fields are rejected. Hard rules: route/slice must agree (GENERAL->general, MIXED->mixed, DOCUMENT->text/table/scanned), DOCUMENT needs `answerable`, MIXED needs all three gold fields.
- `eval/validate.py` (`python -m eval.validate [path] [--docs path] [--strict]`): schema errors with line numbers, bad JSON, duplicate ids, counts per route/slice/type/answerable. Warnings: placeholder (TODO) rows, answerable rows without gold_pages/gold_answer, unanswerable rows with gold_pages, gold pages without `verified_by` or `pdf_page`, unknown doc keys. Exit 1 on errors (or on warnings with `--strict`).
- `eval/docs.yaml`: doc keys `report_a` (text + tables), `report_b` (scanned), `demo_scan`, all with TODO filenames. PDFs are not committed.
- `eval/questions.jsonl`: 4 placeholder rows (`TODO-*`). Delete them when real rows exist.
- `eval/README.md`: how to write questions, record gold pages, and the two-person rule.
- `tests/test_eval_tooling.py`: 16 tests. Full suite 20 passed, ruff clean.

**Deviations / notes**
- Added `type: near_miss` and a `slice` on unanswerable rows (needed for per-slice reporting). Added `author`/`verified_by`/`notes` to support the two-person rule.
- Doc keys are `report_a/report_b/demo_scan` (addendum §3), not the `eig_fy25` style in design §14. Rename in `docs.yaml` once the corpus is chosen.

**Open TODOs (human work)**
- Pick the corpus and fill `eval/docs.yaml`; write ~85 questions; second person verifies gold pages; run `python -m eval.validate --strict` before freezing.

## Step 02: Ingestion part 1, parsing & chunking

**Built** (pure functions; no API, SQLite, embeddings or Chroma)
- `app/ingestion/pdf_parse.py`: `parse_pdf(path, ocr_cfg, parsing_cfg, ocr_fn=None)` yields one `PageRecord` per page (`pdf_page`, `page_label` from PDF page labels else the index, `text`, `image_area_ratio`, `source_kind` text|ocr, `tables`, `error`). Scanned test: `text < ocr.min_chars` and image union area `>= ocr.min_image_area_ratio`. OCR is pytesseract on a grayscale render at `ocr.dpi`; OCR is lazy (per page, so step 03 can show progress). Missing tesseract -> that page comes back with `error` set and empty text; other pages still parse. Password-protected or unreadable PDFs raise.
- Header/footer stripping: a cheap first pass finds edge lines (first/last `parsing.repeat_edge_lines`=3 lines of a page) that, after digits are normalised (`Page 3 of 300` == `Page 4 of 300`), appear on > 50% of text pages; they're dropped from every page. Skipped for docs under `parsing.repeat_min_pages`=3 text pages.
- `app/ingestion/tables.py`: `find_page_tables` (PyMuPDF `find_tables`, never raises), `clean_rows`, `table_to_markdown`, `table_to_markdown_chunks` (split by rows, header repeated, `|` escaped). Table text is removed from the page text by skipping text blocks whose centre is inside a table bbox.
- `app/ingestion/chunking.py`: `split_text` (paragraph -> sentence -> word -> hard char split), `chunk_page`, `chunk_document`, `Chunk` dataclass. IDs `{doc_id}:{page}:{i}` and `{doc_id}:{page}:t{i}` for tables; `embed_text` = `"{title} — page {label}\n{text}"`.
- `app/ingestion/tokens.py`: `estimate_tokens`, a regex word/punctuation count with extra for long words. Deliberately over-counts so a 400 chunk stays under bge-small's 512 limit.
- `scripts/parse_preview.py <pdf> [--pages 5-20] [--no-ocr] [--samples N]`: per-page kind/chars/image%/tables/chunks plus sample chunks.
- Config: new `parsing:` block in `config.yaml` (`repeat_line_ratio`, `repeat_min_pages`, `repeat_edge_lines`, `table_min_rows`, `table_min_cols`); `ParsingConfig` defaults so older configs still load. Added `PyMuPDF`, `pytesseract`, `Pillow` to `requirements.txt`.
- Tests: `tests/test_ingestion_parsing.py` (35 tests; PDFs generated in-test). Full suite 55 passed, 1 skipped (real-OCR test, tesseract not installed locally), ruff clean.

**Run**
```
python -m pytest -q tests/test_ingestion_parsing.py
python scripts/parse_preview.py "<path to pdf>" --pages 1-40 --no-ocr
```

**Deviations / notes**
- Header/footer candidates are limited to the first/last 3 lines of a page (not every line on the page), so a repeated body line like a recurring note isn't removed.
- Unicode is NFKC-normalised on parsed text (fixes ligatures such as `Oﬀerings`, which the real FY25 report has). So `text` is "as-is" apart from whitespace and ligature folding.
- Paragraphs: PyMuPDF text blocks become paragraphs, with lines inside a block joined by spaces; OCR output is split on blank lines the same way.
- Header/footer stripping on scanned pages only uses repeats found on the document's native-text pages (OCR is lazy, so we don't OCR everything first). A fully scanned PDF keeps its headers/footers.
- Overlap carries the last N words of the previous chunk (word granularity), not whole sentences.
- A single table row bigger than the chunk budget is emitted whole rather than cut mid-cell (may exceed 400 estimated tokens; embedding will truncate).
- Tried on the real EIG FY25 AR (first 60 pages, no OCR): ~270 chunks, 54 table chunks, ~0.4 s/page including `find_tables`, so a 300-page text report parses in about 2 minutes before embedding. Page 16 and others that are full-page images were routed to OCR as expected. `find_tables` is imperfect on that report: some narrative/forms text (e.g. the CSR table on p.39) gets merged into a table chunk. Not fixed; revisit if the table-slice eval is weak.
- **OCR not exercised against real tesseract on this machine** (not installed on this Windows box; it is in the Docker image). OCR code path is covered with a fake `ocr_fn` and a mocked "tesseract missing" error; `test_real_ocr_reads_rasterised_text` runs wherever tesseract is on PATH. Benchmark OCR on ~5 pages (addendum §2) before step 03 depends on it.

**Open TODOs**
- ~~Run the OCR benchmark~~ Done, see below.
- Step 03: decide whether a doc fails when any/all pages have `error` set (currently the parser only reports it).

**OCR benchmark (done, in the Docker image, tesseract 5.5.0, 200 dpi)**
- Input: 5 pages of the real EIG FY25 AR (pp. 22, 23, 35, 36, 39) rasterised into an image-only PDF at 150 dpi (a clean "scan"; real scans will be worse). Compared the OCR words with the original text layer.
- All 5 pages were routed to OCR with no errors. Word similarity vs the original: 0.97, 0.91, 0.99, 0.98 on text pages and **0.78 on the page with tables** (expected: scanned tables are an accepted limitation; cells get run together into one line).
- Speed: ~3-4 s/page after warm-up (first page 8 s), 20.7 s for 5 pages. That is ~15-20 min for 300 pages, a bit above the addendum's 5-15 min estimate. Partial querying (queryable while processing) matters even more; consider lowering `ocr.dpi` to 150 in step 10 if speed matters more than accuracy.
- Readable text overall. Weak spots: pages with sidebars/pull-quotes interleave columns and add junk ("Sn Listing ee ..."), and spelling slips on brand names ("Ellenbarie").
- Switched `import fitz` to `import pymupdf as fitz` (newer PyMuPDF warns that `fitz` is deprecated).

## Step 03: Ingestion part 2, upload API, worker, index

**Built**
- `POST /documents` (multipart): streams the upload to `data/uploads/.upload-*.tmp` while hashing (SHA-256) and counting bytes, then checks in order: PDF header (`%PDF-` in the first KB, else **415**), size > `upload.max_mb` (**413**, aborts mid-stream), duplicate hash, then opens it with PyMuPDF: unreadable, password-protected or > `upload.max_pages` pages -> **422** with a readable message. On success the file is stored as `data/uploads/{doc_id}.pdf`, a `documents` row is inserted as `QUEUED`, and the endpoint returns **202** with `doc_id`. Also `GET /documents`, `GET /documents/{id}` (adds per-page timings) and `DELETE /documents/{id}` (204; removes row, file and vectors; stops a job in flight).
- `app/ingestion/worker.py`: one daemon thread + `queue.Queue`, started in the FastAPI lifespan. Flow `QUEUED -> PROCESSING -> PARTIAL -> READY | FAILED(reason)`. Parses with `parse_pdf` (step 02), chunks per page, embeds in batches of `embedding.batch_size` (32), upserts to Chroma every `ingestion.upsert_every_pages` (20) pages and updates `pages_done`/`chunks`. PARTIAL = first batch is indexed (queryable); a short doc goes straight to READY. A job starts by deleting that doc's old vectors, so re-runs never leave strays. Any exception marks that doc FAILED (message <= 500 chars, vectors removed) and the loop continues.
- Crash recovery: on startup everything in `QUEUED/PROCESSING/PARTIAL` is reset and re-queued (`requeue_unfinished`).
- `app/ingestion/embedder.py`: tiny `Embedder` protocol (`model_name`, `embed_documents`, `embed_query`); `FastEmbedder` = bge-small-en-v1.5 via fastembed, loads lazily on first use, model files cached in `data/models` (`paths.model_cache_dir`). `embed_query` uses fastembed's `query_embed` (bge query prefix), ready for step 04.
- `app/ingestion/index.py`: `VectorIndex` over a Chroma persistent client, collection `chunks`, cosine. Chunk metadata: `doc_id, filename, page, page_label, chunk_idx, source_kind, char_len, embedding_model, ingest_version` (the stored text is the chunk without the title header). **Embedding guard:** the model name is stored in the collection metadata; startup raises `EmbeddingModelMismatch` (names both models and says how to fix it) if config differs, so the API refuses to start.
- `app/storage/documents.py`: `DocumentStore` (one short connection per call) and status constants (`QUERYABLE = (PARTIAL, READY)`, for step 04's filter). `init_db` now also adds the new `documents` columns to an existing database (`embedding_model, ingest_version, n_text_pages, n_table_pages, n_ocr_pages, n_failed_pages, embed_seconds, page_timings`), so an older `data/docqa.sqlite` keeps working.
- Config: `ingestion.max_failed_page_ratio` (0.5), `paths.model_cache_dir`. `requirements.txt`: `fastembed`, `chromadb`.
- Logging: an `ingest_done` line with pages, chunks, text/table/ocr/failed page counts, total and embed seconds, seconds/page.
- Tests: `tests/test_ingestion_pipeline.py` (26) + `tests/conftest.py` (fake embedder, PDF builders, app on temp dirs). Covers upload validation (non-PDF, empty, corrupt, too big, too many pages, encrypted, filename sanitising), duplicate hash and retry of a FAILED upload, list/get/404, delete (also mid-processing), status transitions incl. PARTIAL (also observed live from outside via a gated embedder), page-kind counts and chunk metadata (text/table/ocr), batch size cap, deterministic IDs on re-run, worker survives a failing doc and an unexpected error, all-OCR-fail -> FAILED, a few bad pages -> READY with warning, blank PDF -> FAILED, re-queue on startup (also clears stale vectors), missing file on re-queue, embedding-guard mismatch, old-DB migration. All offline. Full suite: **81 passed, 1 skipped** (the real-tesseract test), ruff clean; the pipeline tests ran 5x in a row with no flakes.

**Run**
```
python -m uvicorn app.main:app --port 8000
curl -F "file=@report.pdf;type=application/pdf" localhost:8000/documents      # 202 + doc_id
curl localhost:8000/documents                                                   # status, pages_done/pages_total
python -m pytest -q tests/test_ingestion_pipeline.py
python scripts/ingest_bench.py report.pdf [--pages 22-41] [--rasterise]        # real model, prints s/page
```

**Measured** (`scripts/ingest_bench.py`: real worker + fastembed + Chroma; model load excluded)
| | Text PDF | Scanned PDF |
|---|---|---|
| Input | real EIG FY25 AR, all **86 pages** (shorter than the ~300 the design assumed) | FY25 AR pp. 22-41 (20 pages) rasterised to an image-only PDF at 150 dpi (a clean "scan") |
| Where | Windows laptop CPU, no tesseract | Docker image (tesseract 5.5.0, `ocr.dpi` 200) |
| Pages | 84 text (42 with tables); 2 image-only pages **failed OCR** (no tesseract here) and the doc still finished READY with a warning | 20 OCR, 0 failed |
| Chunks | 466 (5.4/page) | 90 (4.5/page) |
| **Total** | 126.5 s = **1.47 s/page** | 87.8 s = **4.39 s/page** |
| Parse + chunk (+OCR) | 42.1 s = 0.49 s/page (median 0.28) | 69.3 s = 3.47 s/page (median 3.19, max 5.5) |
| Embed | 83.8 s = 180 ms/chunk | 18.2 s = 202 ms/chunk |
- Extrapolated to 300 pages: text ~7 min (about two thirds is embedding), scanned ~22 min. Both are above the design (1-3 min) and addendum (5-15 min) estimates, so partial querying matters, and so does the demo plan below.
- Real-server check (uvicorn, real model): uploaded a 40-page slice (212 chunks) and hard-killed the process while the doc was PARTIAL (20/40, 85 chunks). On restart the doc was re-queued and finished READY with 212 chunks, and Chroma held exactly 212 unique IDs (no duplicates). Changing `embedding.model` in a copy of the config made startup fail with the guard's message.

**Deviations / notes**
- `doc_id` = first 16 hex chars of the SHA-256, so it is deterministic and two racing identical uploads collide on the primary key instead of creating two docs.
- A duplicate upload returns **200** with the existing doc and `duplicate: true`. Exception: if the existing doc is `FAILED`, the same file is re-queued (**202**), so a user can retry after fixing the cause (e.g. installing tesseract).
- Page-error policy (the step 02 open TODO): a page with a parser `error` (OCR unavailable/failed) is skipped. If more than `ingestion.max_failed_page_ratio` of the pages failed -> `FAILED("N of M pages could not be read (...)")`; zero chunks -> `FAILED("no text could be extracted from this PDF")`; otherwise **READY with a warning in `documents.error`**. So `error` can be non-null on a READY doc, and the UI should show it as a notice.
- Recovery re-queues `QUEUED` docs too (the prompt names PROCESSING/PARTIAL; a doc queued just before a crash would otherwise sit forever). A re-run starts over; it does not resume.
- `pages_done` counts pages parsed and flushed in a batch, so with `upsert_every_pages: 20` a 10-15 page demo scan shows 0 -> done with no visible progress. For the step 12 demo set `ingestion.upsert_every_pages` to ~3-5.
- "Encrypted" = `needs_pass`; a PDF encrypted with an empty user password opens fine and is accepted.
- Windows: PyMuPDF keeps the file open while the worker parses, so `DELETE` during processing cannot unlink it. The endpoint ignores that error; the worker closes the parse generator and removes the file when it notices the cancel.
- `embed_seconds` is embedding only (the Chroma upsert is not included and was well under 1 s in total in the measured runs). Per-page time = parse + OCR + chunking; page 1 also carries the header/footer pre-pass.
- Test PDFs need per-page distinct words: the step 02 header/footer stripper (rightly) removes lines that repeat on > 50% of pages once digits are normalised, so near-identical pages vanish.
- Local runs used Python 3.12 (chromadb 1.5.9, fastembed 0.8.1); CI is 3.11. fastembed prints "Local file sizes do not match the metadata" in Docker when the model cache was downloaded on Windows; harmless. `docker compose` also prints an orphan-container warning for `docqa-mongodb-1`, which is not part of this project.

**Open TODOs**
- Step 04: filter retrieval by `QUERYABLE` statuses and add the coverage note ("Searched pages 1-N of M") when a PARTIAL doc is involved.
- Step 10: embedding is the bottleneck (~180-200 ms/chunk). Try fastembed `threads`, `ocr.dpi` 150 (OCR is ~3.5 s/page), and check whether fewer/smaller chunks cost recall.
- Step 07 UI: show `pages_done/pages_total`, the PARTIAL state, FAILED reasons and the READY-with-warning notice.

## Step 04: Retrieval, retrieval eval, MLflow, CI gate

**Built**
- `app/retrieval/retriever.py`: `Retriever(index, embedder, store, retrieval_cfg).retrieve(question, top_k=None, doc_ids=None) -> RetrievalResult`. Embeds the question with the bge query prefix and searches **only PARTIAL/READY documents** (doc ids come from SQLite and go to Chroma as a `doc_id $in` filter; `doc_ids` can narrow the search but never revives a non-queryable doc). `top_k` defaults to `retrieval.fetch_k` (8); step 05 takes the first `retrieval.top_k` (5). Each `RetrievedChunk` has chunk id, doc_id, filename, page (pdf index), page_label, source_kind, text, score (cosine similarity = 1 - distance) and rank. The result also carries `searched` / `not_ready` doc statuses (for the "still processing" / "failed" messages in step 05), `top_score`, `t_embed_ms` / `t_retrieve_ms` (design §15 fields) and **`coverage_note()`**: "X.pdf: searched pages 1-20 of 312; the rest is still processing." when a searched doc is PARTIAL (the step 03 TODO; step 05 shows it when it abstains). No queryable docs gives an empty result without embedding the question.
- `POST /debug/retrieve` `{question, top_k?, doc_ids?}` returns the result as JSON (`app/api/debug.py`; no auth, like the rest of the API). `app.state.retriever` is built in the lifespan.
- `eval/retrieval_eval.py` (`python -m eval.retrieval_eval`): Recall@1/3/5/8 (strict), Recall@5 ±1, MRR (strict and ±1), overall and per slice (text/table/scanned). Only answerable DOCUMENT rows are scored, only against READY docs (docs.yaml filename -> READY document); everything else is skipped with a reason. Prints a table and writes `eval/results/retrieval_latest.json` (git-ignored; holds every question's rank, the misses and the top pages retrieved). Also reports mean top score of answerable vs unanswerable questions (that gap is what θ has to separate) and retrieval latency p50/p95. Pure metric functions are in `eval/retrieval_metrics.py`.
- **MLflow** (`--mlflow`, `eval/tracking.py`): params (chunk size, overlap, top_k, embedding model, mode), all metrics per slice, tags `git_hash` (+`+dirty`) and `eval_set_hash`, the result JSON as an artifact; local file store in `./mlruns`. Optional: imported only with the flag; `requirements-eval.txt` installs `mlflow-skinny`.
- **CI gate**: `python -m eval.retrieval_eval --fixture --check`, run in `.github/workflows/ci.yml` after pytest, with the fastembed model cached (`actions/cache` on `data/models`) and the result JSON uploaded as an artifact. Corpus = two **synthetic, generated** near-twin annual reports (`eval/ci_fixture.py`, "Halden Power Cables" FY25/FY26: 17 pages each, roman front matter so label != pdf page, running footers, 4 table pages each, FY25 figures repeated as comparatives in FY26) + `eval/fixtures/ci_questions.jsonl` (49 answerable, 4 near-miss unanswerable; text 28 / table 21) + `eval/baselines/ci_retrieval.json`. Generated rather than committed because `*.pdf` is git-ignored and there is no licence question; two near-identical reports force the retriever to pick the right year as well as the right page. Real embedder, real worker, real Chroma; no LLM, no network except the one-time model download. Fails if Recall@5 drops more than 3 points or MRR more than 0.05 below baseline; exit 1 = regression, 2 = setup error.
- Tests: `tests/test_retrieval.py` (15) and `tests/test_retrieval_eval.py` (16). Non-queryable docs filtered, exact cosine ranking, `top_k`/`doc_ids`, coverage note, query prefix, the debug endpoint, metric functions on hand-made data (strict, ±1, label fallback, Recall@k, MRR, per slice), evaluation plumbing with a scripted retriever, `resolve_docs`, fixture consistency (every gold answer is printed on its gold page and labels match; the baseline hash matches the question file), the whole fixture pipeline end to end with an offline bag-of-words embedder, the gate (pass / fail / MRR / override / warning), MLflow (skipped if not installed). Full suite **112 passed, 1 skipped**, ruff clean.

**Baseline** (committed in `eval/baselines/ci_retrieval.json`; identical on Windows/Python 3.12 and in the Docker image, Linux/Python 3.11)

| slice | n | R@1 | R@3 | R@5 | R@8 | R@5 ±1 | MRR |
|---|---|---|---|---|---|---|---|
| overall | 49 | 75.5% | 95.9% | **100%** | 100% | 100% | **0.857** |
| text | 28 | 89.3% | 100% | 100% | 100% | 100% | 0.946 |
| table | 21 | 57.1% | 90.5% | 100% | 100% | 100% | 0.738 |

Top retrieval score: answerable 0.741 vs unanswerable 0.711 (max 0.765). Retrieval p50 is ~12-15 ms (query embedding included). Sabotage check: `--top-k 1` makes the gate fail (R@5 75.5%, exit 1).
- **Real corpus: no numbers yet.** `eval/questions.jsonl` still holds placeholders and `docs.yaml` has TODO filenames, so the run exits 2 with a clear message. Once the questions and filenames are in: upload the reports in the app, then `python -m eval.retrieval_eval --mlflow`.
- Sanity check on the real EIG FY25 AR (86 pages, 466 chunks, ingested in 142 s; 2 pages not OCR'd because tesseract is not installed on this machine): "consolidated revenue from operations FY2025" retrieves the P&L (p56) and the revenue tables (p35, p74) at the top. Scores for real questions are 0.55-0.71.

**Deviations / notes**
- **Bug fixed from step 03: the bge query prefix was never applied.** fastembed's `query_embed` for `bge-small-en-v1.5` is identical to `embed` (cosine 1.0, checked), so step 03's comment was wrong. `FastEmbedder.embed_query` now prepends the instruction itself; the string is `embedding.query_prefix` in `config.yaml` (queries only, so **no re-index needed**; set it to `""` to turn it off). `embedder_from_settings()` builds the embedder everywhere (app, bench, eval).
- Effect of the prefix on the tiny CI corpus: R@5 unchanged (100%), but R@1 is 75.5% vs 83.7% and MRR 0.857 vs 0.903 without it (about 3-4 questions, within noise at n=49). bge v1.5's own guidance says the prefix is optional. **Step 10 should test prefix on/off on the real corpus** (one config line).
- **θ = 0.5 is meaningless with this embedder.** Every retrieved chunk scores above 0.5, even for the unanswerable "FY2030 forecast" question (0.66 on the real report; 0.71 mean over the CI corpus's unanswerables). Gate 1 (score) will barely filter anything until θ is recalibrated in step 10 (probably ~0.65-0.75); gate 2 (the LLM's INSUFFICIENT) carries the load meanwhile.
- The CI corpus is small and easy, so Recall@5 saturates at 100% and only gross breakage moves it. That is why MRR is gated too (a looser 0.05): my addition to the "Recall@5 drops > 3 pts" rule. One question is 2 points of Recall@5, so a single flipped question does not fail CI. Numbers for the README must come from the real corpus, not this one.
- MLflow 3.x refuses its file store unless `MLFLOW_ALLOW_FILE_STORE=true` ("maintenance mode"), and the SQL backend needs full `mlflow` (not skinny). I kept the design's local file store; `eval/tracking.py` sets the env var for the run. To follow MLflow's recommendation instead, switch to `sqlite:///mlruns/mlflow.db` and install `mlflow`.
- Real-corpus mode opens the same `data/` SQLite + Chroma as the app: stop the API while it runs.
- Placeholder `TODO-*` rows are skipped automatically (their docs are not READY), so nothing needs deleting to run the eval once real rows exist.
- Not done (by design): hybrid/BM25 and reranker (step 10 experiments); router/answer evals (steps 05-09).

**Open TODOs**
- Human: real eval questions + `docs.yaml` filenames (blocks the real Recall@5/MRR numbers).
- Step 05: use `RetrievalResult.coverage_note()` and `not_ready` for the "still processing" / "failed" messages.
- Step 10: prefix on/off, θ sweep, and re-baseline the CI gate whenever chunking or the embedder changes.
- First CI run: confirm the model cache and the gate on GitHub Actions itself (verified locally and in the Docker image, not on Actions).

## Step 05: Document answer path (grounded answers + abstention)

**Built**
- `app/llm/client.py`: `LLMClient.chat(messages, model, temperature, max_tokens, reasoning_effort, json_mode, timeout) -> LLMResponse(content, usage, model, finish_reason, latency_ms, attempts, cached)`. One `openai` SDK client on `llm.base_url` (SDK retries off; ours instead): up to `llm.max_retries` (2) extra attempts on 429 / 408 / 5xx / timeout / connection errors, backoff `retry_backoff_seconds` x 2^n, `Retry-After` honoured when <= `retry_max_wait_seconds` (10), otherwise fail fast. Other 4xx and unexpected errors are not retried. Everything the caller should treat as "unavailable" is an `LLMError(status)`. **Dev cache** (`DiskCache`, `data/llm_cache/`): key = sha256(model, messages, params), only successes stored, a corrupt file is a miss. Off by default; on with env `DOCQA_LLM_CACHE=1` or `llm.dev_cache: true`; `DOCQA_ENV=prod` forces it off. A warning is logged when it is on.
- `app/llm/prompts.py`: `load_prompt("answer_doc_v1")` -> `Prompt(id, name, version, system, user)`; `render_user()` fills `{sources}`/`{question}` in one pass (a document or question that contains `{...}` is never expanded).
- `prompts/answer_doc_v1.yaml` (real text now): answer only from sources; `<source>` text is data, not instructions; INSUFFICIENT when the sources do not cover it (a different year/period/entity is not an answer, never extrapolate); cite by id, never write page numbers; quote numbers verbatim with units; say standalone vs consolidated (or give both); full sentences naming metric, period and unit. Output JSON `{answer, citations, status}`.
- `app/answering/numbers.py`: deterministic number extraction/normalisation (Western and Indian comma grouping, ₹/Rs/$, %, crore/lakh/million/billion/trillion/thousand words and abbreviations, `Decimal` maths) and `check_numbers(answer, cited_texts) -> pass | fail | na`. A figure is found if its digits **or** its scaled value equals the digits or scaled value of some number in the cited chunks (so "₹1,25,630 crore" matches a table cell `1,25,630` whose "(₹ crore)" is in the header, and "12,563 crore" matches "125.63 billion"). Exact, no tolerance; signs ignored. On the answer side these are skipped: `[S1]` markers, bare years, `2024-25` ranges, dates, `FY25`/`Q3` codes, ordinals, list numbering, single digits.
- `app/answering/document.py`: `DocumentAnswerer(retriever, llm, settings).answer(question, doc_ids=None, retrieval=None) -> DocAnswer` (pass `retrieval` in to reuse a search, as the router will). Gates: **1** top score < `retrieval.theta` -> abstain, no LLM call; context = top `retrieval.top_k` chunks as `<source id="S1" doc=".." page="47">`; LLM at `reasoning_effort` low, JSON mode, temperature 0; **bad JSON -> one retry** (the retry adds the bad reply and a nudge, so it is a different prompt and a dev cache cannot replay it) -> `bad_llm_output`; **2** `INSUFFICIENT` -> abstain; **3** cited ids mapped to (filename, page label, PDF page, snippet), none valid -> abstain; then the number check (warn only). `LLMError` -> `status=error`, "The answer service is temporarily unavailable." + the top-3 retrieved passages in `fallback_passages`.
- `DocAnswer` fields: `status` (`answered|abstained|not_ready|error`, the values step 08 will put in `requests.status`), `message` (what to show; the answer when answered), `abstain_reason`, `citations[]` (`display` = "p.47 (PDF p.53)", or just "p.12" when label == PDF page), `closest_pages[]`, `fallback_passages[]`, `coverage_note`, `notices[]`, `number_check`, `number_warning`, `unmatched_numbers`, `citations_valid`, `dropped_citations`, `top_score`, `n_sources`, `searched`, `timings` (`embed_ms, retrieve_ms, llm_ms, total_ms`), `tokens` (`prompt, completion, total`, summed over the retry), `llm_calls`, `model`, `prompt_version`. Abstain reasons: `low_score, insufficient, no_valid_citations, bad_llm_output, no_documents, documents_not_ready, no_results, llm_unavailable`. One `answer_doc` log line per call (ids, scores, counts, timings; no question or document text).
- Abstention text: "I couldn't find this in your documents (searched: X.pdf). I won't guess. Closest sections: X.pdf p.47 (PDF p.53), ..." plus `RetrievalResult.coverage_note()` when a searched doc is PARTIAL, plus a line for any doc that is still processing or FAILED. If nothing is queryable: `not_ready` with the "still processing (140/312 pages)" / "couldn't be processed: reason" messages; nothing uploaded: `abstained/no_documents`.
- `POST /debug/answer_doc {question, doc_ids?}` (`app/api/debug.py`); `create_app(..., llm=None)` is the test injection point; `app.state.llm`, `app.state.doc_answerer`. Startup logs a warning when `GROQ_API_KEY` is missing and the base URL is Groq.
- Config: `llm.max_retries / retry_backoff_seconds / retry_max_wait_seconds / dev_cache / cache_dir`, new `answer:` block (`max_tokens` 1500, `snippet_chars` 300, `closest_pages` 3, `fallback_passages` 3), all with defaults. `requirements.txt`: `openai`. `.env.example`: commented `DOCQA_LLM_CACHE`.
- Tests (LLM mocked): `tests/test_numbers.py` (normaliser incl. "12,563", "₹1,25,630 crore", "4.5%", "12.5 million", grouping, noise skipping, the check), `tests/test_llm_client.py` (retries, Retry-After, usage, timeouts, cache hit/key/corruption, prod guard, prompt loader), `tests/test_answer_document.py` (every gate, citation mapping and normalisation, bad JSON then retry, number-check wiring, coverage note, not-ready/failed/no-docs, LLM-error fallback, `<source>` tag defusing, page display), `tests/test_answer_api.py` (endpoint end to end with the real app). Full suite **223 passed, 1 skipped**, ruff clean. One unreproduced failure of `test_retrieval_eval.py::test_eval_end_to_end_on_the_fixture_with_an_offline_embedder` in a single full run (it passed alone, 6x in a row, and in 3 more full runs); probably timing under load, and nothing in it touches step 05 code.

**Run**
```
python -m uvicorn app.main:app --port 8000          # GROQ_API_KEY must be in the environment
curl -X POST localhost:8000/debug/answer_doc -H "content-type: application/json" -d '{"question":"What was consolidated revenue in FY25?"}'
python -m pytest -q tests/test_numbers.py tests/test_llm_client.py tests/test_answer_document.py tests/test_answer_api.py
```

**Hand trial** (real Groq `gpt-oss-120b`, real EIG FY25 AR, 86 pages, 466 chunks, theta 0.5)

| Question | Result |
|---|---|
| Consolidated revenue from operations FY2025 | answered, "3,124.83 million rupees", cites p.56, p.35, p.74, number check pass, 2.6 s |
| Profit after tax FY2025 | answered "832.89" (correct: PBT 1,078.25 - tax 245.36), number check pass, but **no unit** (it is not in the retrieved chunk). Prompt rule 7 was then tightened to ask for a full sentence; the reply is now "Profit after tax for FY2025 was 832.89." |
| Chairman and Managing Director | answered, cites the related-party table p.79 |
| Employees | answered "281 permanent and 85 contractual, totaling 366", number check **fail** on 366: the model added them. The check does what it is for (flags a figure that is not in the source), but a correct sum will also warn |
| Forecast revenue FY2030 / revenue FY2022 / revenue FY2030 | all abstained via gate 2 (`insufficient`) with closest pages p.83/p.56/p.74; top scores 0.67-0.69, so **gate 1 let them through**, as predicted in step 04 |
| Capital of France | abstained at gate 1 (`low_score`, top score 0.495), only because theta 0.5 sits just above this question's score |

- Latency (Groq, one call): LLM 0.5-2.5 s, total 0.6-3.6 s per question; about 1,500-2,400 prompt tokens each.

**Deviations / notes**
- **Groq free-tier TPM is the real constraint:** 8,000 tokens/min per model, so about 3 answers per minute at ~2,200 tokens each. Four questions in a row hit a 429 with `Retry-After` 5.9 s; with the first cap (5 s) that surfaced as "temporarily unavailable" (so the fallback path was seen working live), so `retry_max_wait_seconds` is now 10. Eval runs (steps 09-10) must use the dev cache (`DOCQA_LLM_CACHE=1`) and be paced; an 85-question run is ~190K tokens, most of one day's 200K cap per model.
- Gate 3 wording: the build prompt says every cited id must exist, design §11 says "at least one valid". Implemented the design: unknown ids are **dropped and recorded** (`dropped_citations`, `citations_valid=false`), and we abstain only when none are left.
- `answer_doc_v1.yaml` was edited in place after the hand trial (rule 7), not bumped to v2, because v1 had never been used by a released step. Bump the version from here on.
- `status` also has `not_ready` (every searched doc is still processing or failed), matching the `requests.status` values in design §15.
- A correct derived figure (sums, differences) trips the number check; MVP only warns. How often it fires on the real eval set is a step 10 item.
- `timings.total_ms` includes retrieval only when the answerer ran it (not when `retrieval=` was passed in).
- Not tried live: the partial-coverage note (unit tests and step 04's `coverage_note()` cover it) and the Ollama base URL.

**Open TODOs**
- Step 06: the router should call `DocumentAnswerer.answer(question, retrieval=...)`; GENERAL/MIXED answers and the `/query` endpoint.
- Step 08: log `DocAnswer.timings/tokens/status/abstain_reason/citations_valid/number_check` into `requests`.
- Step 10: calibrate theta (0.5 filters almost nothing; on the real report top scores are 0.63-0.73 for answerable and 0.49-0.69 for unanswerable questions), and measure the number-check false-warning rate.
- Step 07 UI: show `notices`, `number_warning`, `coverage_note`, `closest_pages`, and `fallback_passages` on error.

## Step 06: Router, general path, MIXED, `/query`

**Built**
- `prompts/router_v1.yaml`: system prompt with the three routes, six rules and 8 few-shot examples (the hard ones from design §9: "What is EBITDA?" -> GENERAL, "EBITDA margin" -> DOCUMENT, "What was the revenue?" -> DOCUMENT, a vague "tell me about it" -> DOCUMENT, a company that is not uploaded -> GENERAL, two MIXED splits). The user message lists the uploaded documents with their state (`EIG AR FY25.pdf (READY)`, `(processing 140/312 pages)`, `(failed)`, `(queued)`). File names are sanitised (control chars and `<>` removed, cut to `query.doc_title_chars`) and capped at `query.router_max_docs`.
- `app/routing/router.py`: `Router.route(question, docs) -> RouteOutcome`. gpt-oss-20b, JSON mode, temperature 0, `reasoning_effort` low, `query.router_max_tokens` 600. Reply validated by `RouterDecision` (route in DOCUMENT|GENERAL|MIXED; MIXED needs both sub-questions). **Bad JSON -> one retry (the retry carries the bad reply and a nudge) -> fall back to DOCUMENT with the whole question**, `router_ok=false`, `fallback_reason` `bad_json`, and a `router_fallback` warning log line. An `LLMError` from the router falls back the same way (`llm_unavailable`). **No documents at all -> no LLM call**, route GENERAL, `router.skipped=true`.
- `app/answering/general.py`: `GeneralAnswerer.answer(question) -> GeneralAnswer`. `prompts/answer_general_v1.yaml` tells the model to say when it is unsure of exact figures, never to imply it read the user's documents, and to ask for specifics on a vague question. `GENERAL_LABEL` ("General knowledge — not from your documents; may be out of date.") is attached by code on every result, including errors. LLM failure or an empty reply gives `status=error` (never an exception).
- `app/answering/query.py`: `QueryService.run(question, trace_id)`. Reads the document list, routes, runs the document path, the general path or **both in parallel (two threads, each with its own copy of the logging context)**, and composes the response. Each path is isolated: an unexpected crash in one becomes an `error` section (`abstain_reason: internal_error`, no internals shown) and the other part still answers.
- `POST /query {question}` (`app/api/query.py`): the question is stripped; empty or longer than `query.max_question_chars` (500) -> **422** with a readable message, before any LLM call. `trace_id` is the request ID (`X-Request-ID`, echoed by the middleware).
- `/debug/*` (the step 04/05 endpoints) are **kept but hidden (404) unless `api.debug_endpoints: true` or env `DOCQA_DEBUG=1`**. They have no auth and `/debug/answer_doc` spends LLM quota. The test fixture `settings` turns the flag on.
- Small refactor: `app/llm/json_reply.py::load_json_object` (fence/prose-tolerant JSON parse) is shared by the router and the step-05 answer parser.
- Config: new `query:` block (`max_question_chars`, `router_max_tokens`, `general_max_tokens`, `router_max_docs`, `doc_title_chars`) and `api:` block, both with defaults.

**Response shape** (the same for every route, so the UI needs no per-route logic)
```
{trace_id, route, router_ok,
 router: {ok, skipped, fallback_reason, model, prompt_version, document_question, general_question},
 sections: [                       # document first, then general; one entry per path that ran
   {kind: "document"|"general", status, answer, question, citations[], label, abstain_reason, hint,
    closest_pages[], fallback_passages[], coverage_note, warnings[], number_check, top_score}],
 timings: {router_ms, embed_ms, retrieve_ms, document_llm_ms, document_ms, general_ms, total_ms},
 tokens: {router, document, general, total}     # each {prompt, completion, total}
}
```
- `answer` is always the text to show: the answer when `status=answered`, otherwise the abstention / not-ready / error message (design §11 wording from step 05). Document statuses: `answered|abstained|not_ready|error`; general: `answered|error`. `label` is set on general sections only.
- `warnings` (document sections that answered): other documents still processing or failed, "X.pdf is still processing (20/312 pages): this answer only uses the pages indexed so far", the number-check badge text ("⚠ number not found verbatim in source: 366"), dropped citations. A general section with no documents uploaded gets "No documents are uploaded yet, so this answer is general knowledge. Upload a PDF to ask about it."
- `hint` ("Try asking about something more specific: name the metric, the period and the document.") is set on weak-evidence abstentions (`low_score`, `insufficient`, `no_results`, `no_valid_citations`). That is how vague questions are handled: they proceed, and abstain with the hint.

**Tests** (LLM mocked; `tests/fakes.py::FakeLLM` replies by role, so concurrent calls do not depend on order)
- `tests/test_router.py`: parsing (fences, normalisation, rejects), each route, single routes use the user's own words, JSON mode + router model, hard examples are in the prompt, retry once, fallback on bad JSON twice (plus warning log with no question text) and on `LLMError`, the no-docs shortcut makes zero calls, document-list formatting/sanitising/capping, general label always present, the uncertainty rule is in the general prompt, general failure modes.
- `tests/test_query_api.py` (real app): no docs, DOCUMENT, GENERAL, **MIXED really concurrent (a 2-party barrier inside the fake LLM only releases if both paths are in flight)**, MIXED where the document part abstains while the general part answers, general part failing, a crashed path, router fallback visible in the response, processing doc "(140/312 pages)", FAILED doc reason, MIXED with a processing doc, PARTIAL-doc coverage note, vague-question hint, answered-with-partial warnings, length limit 422 (also empty/whitespace/missing, and read from config), debug endpoints 404/200 by flag and env var.
- Full suite **265 passed, 1 skipped**, ruff clean.

**Run**
```
python -m uvicorn app.main:app --port 8000          # GROQ_API_KEY must be in the environment
curl -X POST localhost:8000/query -H "content-type: application/json" -d '{"question":"What was consolidated revenue from operations in FY2025?"}'
python -m pytest -q tests/test_router.py tests/test_query_api.py
```

**Hand trial** (real Groq: gpt-oss-20b router, gpt-oss-120b answers; real EIG FY25 AR, 86 pages; theta 0.5; dev cache off; 30 s between questions)

| Question | Route (router_ok) | Result | total |
|---|---|---|---|
| Consolidated revenue from operations in FY2025? | DOCUMENT (ok) | answered "3,124.83 million", cites p.56 and p.35, number check pass | 3.9 s |
| What is EBITDA? | GENERAL (ok) | answered, label attached, no retrieval | 1.2 s |
| Profit before tax in FY2025, and who is the Prime Minister of India? | MIXED (ok), split into "What was the profit before tax in FY2025?" / "Who is the Prime Minister of India?" | document: answered "1,078.25", cites p.76; general: answered, label. The two ran in parallel (1.4 s total vs 0.8 + 0.5 s) | 1.4 s |
| **Near-miss:** Ellenbarrie's revenue forecast for FY2030? | DOCUMENT (ok): not sent to GENERAL although the file is called "EIG AR FY25.pdf" | abstained (`insufficient`; top score 0.715, so gate 1 let it through as predicted), closest p.33/p.23/p.17, hint shown | 4.9 s |
| Zenith Foods revenue FY2025 (not uploaded) | GENERAL (ok) | the model said it cannot see documents and has no figure; label attached | 9.0 s |
| EBITDA margin in FY2025 (no company named) | DOCUMENT (ok) | answered "35.12 %", cites p.24, number check pass | 2.0 s |
| "tell me about it" | DOCUMENT (ok) | abstained (`insufficient`; top score 0.527), hint shown | 1.9 s |

- All 7 routed as intended, 0 router fallbacks. A router call is ~1,000-1,200 tokens (the few-shot prompt is most of it) and 0.5-1.6 s. It runs on gpt-oss-20b, which has its own Groq quota, so it does not eat the 120b's 8K TPM; a document answer is still ~2,200-2,400 tokens, so the earlier pacing advice stands.
- Server log checked: no key and no question text; one `query` line per request (route, router_ok, statuses, timings, token totals).

**Deviations / notes**
- **The router does not retrieve**, so the "pass `retrieval=` into `DocumentAnswerer.answer`" hook from step 05 is not used. `/query` calls `answer(sub_question)` and the document path retrieves itself.
- For DOCUMENT and GENERAL the original question is passed on unchanged; the router's sub-questions are used **only for MIXED**. This keeps a rewrite from dropping "FY2025" or a company name.
- A MIXED reply with one empty part counts as invalid JSON (retry, then fall back to DOCUMENT) instead of guessing a split.
- Router prompt rule 3 is deliberately tilted to DOCUMENT when unsure about a company name (file names are often initials, e.g. "EIG" for Ellenbarrie), because DOCUMENT abstains safely while GENERAL could state invented figures. The design §11 policy (a document-style question about a company that is not uploaded is answered GENERAL with the label) still holds when the question names a clearly different company (Zenith Foods above).
- The few-shot prompt uses made-up document names ("Acme Annual Report FY25.pdf"), not the real reports.
- `router_v1` and `answer_general_v1` were stubs, so they are v1 here; bump the version from the next edit on.
- The response has no single per-request `status`. Step 08 should derive `requests.status` from the sections (e.g. the document section's status, else the general one's).
- Windows console: a script that prints answers containing non-breaking hyphens needs `PYTHONIOENCODING=utf-8`; the API itself returns UTF-8 JSON.

**Open TODOs**
- Step 07 UI: render `sections` (document: answer, citation chips -> snippet, closest pages, fallback passages, warnings, hint; general: label). Treat `router_ok=false` as a metric, not a user error. Keep the 500-char limit in the input box.
- Step 08: write one `requests` row per `/query` from `trace_id, route, router_ok, router.model/prompt_version, timings, tokens, sections[].status/abstain_reason/number_check/top_score`; `router.fallback_reason` feeds the router-fallback metric (addendum §4).
- Step 09: router benchmark (3x3 confusion matrix on the labelled set). The hand trial is 7 questions only. Check how often the DOCUMENT-when-unsure tilt sends a truly GENERAL company question to the document path.
- Step 10: theta. Top scores of 0.7+ on FY2030-style near-misses still pass gate 1; gate 2 caught them every time so far.

## Step 07: Streamlit UI

**Built**
- `ui/app.py` (entry): page config, the documents **sidebar** (shown on every page), and `st.navigation` over `ui/views/ask.py` and `ui/views/metrics.py` (placeholder, step 08 fills it). `ui/` is now a package; `app.py` puts the repo root on `sys.path` so `from ui import ...` works under `streamlit run ui/app.py`.
- `ui/api_client.py`: `ApiClient` (`health, list_documents, upload, ask, send_feedback`). Every failure is an `ApiError` whose text is safe to show: API down -> "Can't reach the DocQA service at <url>. Is it running?", timeout, 5xx -> a generic line (never the body), 4xx -> the API's own readable `detail` (415 "Not a PDF", 413, 422 too long, duplicate etc.). Timeouts: query 90 s, upload 120 s, quick calls 8 s.
- `ui/formatting.py`: pure helpers (status labels, progress fraction, FAILED reason / READY warning text, citation label `Report.pdf · p.47 (PDF p.53)`, section titles, `$` escaping, which source lists to show, 👍/👎 mapping). `ui/components.py`: the Streamlit rendering.
- **Documents panel:** upload (multiple PDFs allowed) -> `POST /documents`; per-document card with the badge `⏳ Queued / ⚙️ Processing n/N / 🟡 Partial n/N · searchable / ✅ Ready / ⚠️ Ready (with a warning) / ❌ Failed` + a progress bar for PROCESSING/PARTIAL, the failure reason under FAILED and the step 03 warning under READY. It is an `st.fragment` that re-polls `GET /documents` every 2 s **only while something is in progress**; when that flips, one full rerun redefines the fragment (Streamlit fixes `run_every` at definition). Duplicate uploads say "already uploaded"; upload errors show the API's message.
- **Ask page:** `st.chat_input` (max 500 chars, `DOCQA_MAX_QUESTION_CHARS` overrides) -> `POST /query` -> renders `sections`:
  - DOCUMENT: the answer, one expander per citation (`📎 Report.pdf · p.47 (PDF p.53)`) holding the snippet, and the warnings (including `⚠ number not found verbatim in source: ...`, partial-coverage and dropped-citation notes).
  - GENERAL: the label in an info box **above** the answer.
  - MIXED: `📄 From your documents` and `🌐 General knowledge`, each with its own status.
  - Abstention: the reason text, the hint, the closest pages as expandable snippets, and the coverage note (not repeated if the message already contains it). `not_ready` is an info box. LLM-down shows the error plus "Most relevant passages found" (the step 05 fallback).
  - A "Routed as document · 2.3 s" caption. `router_ok=false` is not shown (a metric, not a user error).
  - History is shown on screen only; `ask()` sends the one question string.
- **👍/👎** (`st.feedback("thumbs")`) under every answer -> **`POST /feedback {trace_id, value: 1|-1}`** (`app/api/feedback.py`, `app/storage/requests.py::RequestStore`). Once rated, the widget becomes a one-line "you rated this" note. 422 for any other value or an empty/over-128-char trace id.
- Tests: `tests/test_feedback_api.py` (8: stored, existing row kept, overwrite, validation, via a real `/query` trace id, SQL-injection-shaped id, log line), `tests/test_ui_helpers.py` (formatting; client with a fake session for every error class; client against the real app: upload, duplicate, list, ask, feedback, API rejections), `tests/test_ui_smoke.py` (headless `AppTest`: every status, every reply kind, API down with a fake and with the real client on a dead port, unreadable replies, no history sent, metrics placeholder). Full suite **324 passed, 1 skipped**, ruff clean.

**Run / click through**
```
python -m uvicorn app.main:app --port 8000        # GROQ_API_KEY in the environment
streamlit run ui/app.py                           # http://localhost:8501 ; DOCQA_API_URL defaults to :8000
docker compose up --build                         # same, UI on :8501
```
1. Sidebar: pick a PDF -> **Upload**. The card goes Queued -> Processing n/N (bar) -> Partial (searchable) -> Ready without touching anything. Upload the same file again: "already uploaded". Upload a text file renamed `.pdf`: the "Not a PDF" message.
2. Ask "What was consolidated revenue from operations in FY2025?" -> answer; open the `📎` chips for the snippets; click 👍.
3. Ask "What is EBITDA?" -> the general-knowledge label above the answer.
4. Ask "Profit before tax in FY2025, and who is the Prime Minister of India?" -> two titled sections.
5. Ask "What was Ellenbarrie's revenue in FY2030?" -> abstention, hint, closest pages. (Needs the unanswerable to get past gate 2; it did in the trial below.)
6. Stop the API and ask again: a red "Can't reach the DocQA service" message, and the sidebar shows the same with a Retry button.

**Checked live** (real API + real Groq + the real EIG FY25 AR, driven through the real UI code with `AppTest`, not a browser): the revenue question answered "3,124.83 million" with chips p.56 / p.35 / p.83 in 3.6 s; the FY2030 question abstained with the hint and closest-page chips in 1.3 s; with no Groq key the same questions showed "The answer service is temporarily unavailable." plus the retrieved passages. The sidebar showed the real READY-with-warning notice (2 pages not OCR'd, no tesseract here).

**Deviations / notes**
- **`/feedback` upserts.** Nothing writes `requests` rows at `/query` time yet (step 08), so a missing `trace_id` row is created holding only `feedback`. **Step 08's per-request write must be an upsert that leaves `feedback` alone** (or insert first and let this update it). Until then the table only holds feedback-only rows.
- Citation "chips" are `st.expander`s, not `st.popover`s: long file names and up to 5 chips don't fit in a row, and an expander shows the snippet inline.
- The 500-char limit is duplicated in the UI (constant, env override); the API remains the authority and its 422 message is shown if they ever differ.
- The UI keeps polling only while a document is in progress; if the API is down the sidebar stops polling and shows Retry.
- A rated answer can't be re-rated in the UI (the endpoint overwrites, so it could be allowed later).
- Widget keys use a per-message id, not the trace id (a caller-supplied `X-Request-ID` could repeat, and Streamlit crashes on duplicate keys; found by a test).
- Long document warnings are cut to 200 chars in the sidebar.
- Not exercised: a real browser (the file-uploader and 👍/👎 clicks, the 2 s fragment polling, `AppTest` can't drive them; the helper logic behind them is unit-tested), and the Docker UI image was not rebuilt (the compose command is unchanged and `ui/` is copied whole).
- `tests/test_retrieval_eval.py::test_eval_end_to_end_on_the_fixture_with_an_offline_embedder` failed again in 2 of 4 full runs this session, **including on a clean stash of the tree without my changes**, and passes alone. It is the same intermittent failure noted in step 05; it is not caused by step 07 and still needs a look (timing or shared state under a full run).
- `.env` has an unquoted value containing spaces, so `source .env` fails in bash (line 4). The API doesn't read `.env` itself; pass `GROQ_API_KEY` explicitly or use docker compose (`env_file`).

**Open TODOs**
- Step 08: metrics page content; write the `requests` row with an upsert (see above); derive `requests.status` from the sections.
- Click through once in a real browser before the demo (upload, polling, thumbs).
- Investigate the intermittent `test_retrieval_eval` failure.

## Step 08: Request log & metrics page

**Built**
- **One `requests` row per `/query`**, written by `QueryService` after the response is built (`app/observability/request_log.py::build_record`, pure). Fields: every design §15 column plus `app_version` (git SHA; env `DOCQA_GIT_SHA` wins, for Docker), `router_fallback` (reason), `answer_chars`, `cited_chunks` (JSON chunk ids), and empty `judge_correct` / `judge_grounded` slots for step 09. `prompt_versions` / `model_ids` are JSON per part (router / document / general). Cost = each part's tokens x `observability.pricing_per_mtok` (list-price equivalent; the rates are my recollection of Groq's gpt-oss prices, **please verify**).
- **No content in the log:** the question's length only, chunk ids of the citations, codes and numbers. A test checks that question, answer and document text are absent from the row.
- **Logging never fails a request:** build and write are wrapped; failure logs `request_log_failed` (error class + short detail, no text). A crash before any response writes an `error` row (`error` = exception class name) and re-raises.
- **Writers share the row safely:** `RequestStore.log` is an upsert that never touches `feedback` (columns are an allow-list); `/feedback` only touches `feedback`. Either can come first. Metrics ignore feedback-only rows (`status IS NULL`).
- **Schema:** `ADDED_REQUEST_COLUMNS` are added to an existing DB by `init_db` (same trick as step 03); new table `upload_failures(ts, status_code, reason)` for uploads refused at the door (415/413/422), written by `POST /documents` with a reason code (`not_a_pdf, too_large, encrypted, too_many_pages, empty_pdf, unreadable, rejected`; no file names).
- **`app/observability/metrics.py`:** percentiles (linear interpolation), rates with their denominators, `build_dashboard(rows, documents, rejections, cfg)`, drift (daily median top score; route mix week over week as total-variation distance between the latest week and the calendar week before), colour `indicator()`, read-only loaders (`mode=ro`, missing file/table gives no rows, never creates the DB), `since_iso` for the time-range filter.
- **Metrics page** (`ui/views/metrics.py`, a thin view): time range (hour / 24 h / 7 d / 30 d / all), Refresh, and the five categories. Operational: p95 tile, error rate, tokens and cost per request, p50/p95/p99 table per stage, ingestion s/page per document. Input: route mix, question-length histogram, upload failures by reason (refusals plus documents that ended FAILED). Output: abstention, citation-invalid, number-check-fail and router-fallback rates, abstain reasons, answer length. Quality: 👍/👎, down share, share of answers rated, judge slot ("No judged answers yet"). Drift: daily median top score, weekly route mix.
- **Colours** come from `observability.thresholds` in `config.yaml` (strict `>`, `lower_is_worse` for the retrieval score); the rule is printed under each number. Fewer than `min_requests_for_indicator` (5) requests gives a grey dot, not a colour. No alerting.
- `docker-compose.yml`: the UI container now mounts `./data` read-only (the page reads SQLite directly); the API gets `DOCQA_GIT_SHA: ${GIT_SHA:-}`.
- Tests: `tests/test_metrics.py` (45: percentiles, every rate and its denominator, ingestion, drift, thresholds, read-only loaders, range filter), `tests/test_request_log.py` (25: a row per route/status, privacy, broken log, crash row, feedback vs log, upload failures, old-DB migration), `tests/test_metrics_page.py` (8: `AppTest`; empty, missing DB, five categories, red/amber/green/grey, range filter). The step 07 placeholder test was replaced. Full suite **401 passed, 1 skipped**, ruff clean.

**Run**
```
python -m uvicorn app.main:app --port 8000      # GROQ_API_KEY in the environment
streamlit run ui/app.py                         # Metrics page in the sidebar
GIT_SHA=$(git rev-parse --short HEAD) docker compose up --build
python -m pytest -q tests/test_metrics.py tests/test_request_log.py tests/test_metrics_page.py
```

**Checked live** (real API + real Groq + the real EIG FY25 AR; 12 questions 25 s apart, 2 refused uploads, 3 thumbs; page rendered with `AppTest` against `data/docqa.sqlite`, not a browser). Routes 8 DOCUMENT / 3 GENERAL / 1 MIXED, 0 router fallbacks. Page showed p95 total 6.92 s (red), p50 2.47 s; abstention 25% (green); one **error row, a real Groq 503 "over capacity"** (red error rate 8.3%, which is 1 of 12); cost-equivalent ~$0.0003/request; ingestion 1.71 s/page; upload failures 2 x `not_a_pdf`; 👍 2 / 👎 1. Server log: no question text, no key.

**Deviations / notes**
- **`status` is the worst of the sections** (error > not_ready > abstained > answered): a MIXED question whose general half failed counts as `error`, one whose document half abstained counts as `abstained`. Say so if you would rather track the halves separately.
- A stage that did not run is **NULL, not 0** (no router call, gate 1 abstention has no LLM time, no document path). Consequence: a reading that rounds to 0.0 ms is stored as NULL too (only seen with the instant fake embedder in tests).
- `t_llm_ms` is the slowest answer-LLM call (MIXED runs both at once), not their sum.
- "Error/timeout rate" is one number: `status = error` (LLM unavailable after retries, which includes timeouts, or an internal error). Timeouts are not split out because the answer objects do not carry the HTTP status.
- Router fallback rate's denominator is requests where the router actually ran (or failed trying); the no-documents shortcut is excluded.
- Drift indicator for the retrieval score uses the **latest day's** median; thresholds (0.55 / 0.45) are guesses from the 0.55-0.71 scores seen so far, to recalibrate after step 10 sets theta.
- Streamlit's `use_container_width` is deprecated in the installed 1.55, so the tables use the default width.
- Seen live, not changed: a 503 from Groq surfaced as "LLM unavailable after 1 attempt(s)"; with `max_retries: 2` I expected more attempts, so check the retry path for 5xx with a long `Retry-After` in step 11 (fallback work).
- Not exercised: a real browser (charts, the range selector, Refresh) and the Docker UI image with the read-only mount (`docker compose config` is valid; image not rebuilt).

**Open TODOs**
- Step 09: write `judge_correct` / `judge_grounded` (0/1) into `requests`; the page already reads them.
- Step 10: recalibrate the colour thresholds, especially `median_top_score` once theta is chosen.
- Step 11: Langfuse reads the same trace ids; revisit the 5xx retry behaviour above.
- Click through the Metrics page once in a real browser before the demo.
- One full run, right after I stopped the live uvicorn, failed `test_retrieval_eval...offline_embedder` (the known intermittent one from steps 05/07) and `test_ui_helpers::test_upload_list_ask_and_rate_through_the_client`; both pass alone and the suite was green 4 times in a row afterwards (401 passed). I could not reproduce the second; treat it as the same load-related flake and watch for it.

## Eval set: first real rows (from the sifra-v2 EIG FY25 golden cases)

**Built**
- `eval/questions.jsonl` now holds **72 real rows** (D001-D072), converted from `sifra-v2/docs/eig-fy25-ar-golden-cases.md` (85 cases). The 4 `TODO-*` placeholders are gone. `eval/docs.yaml`: `report_a` = `EIG AR FY25.pdf` (the file already ingested); `report_b` / `demo_scan` still TODO.
- Kept: 62 direct numeric (D001-D062), 7 multi-year FY2024 comparatives read from the FY25 report (D063-D069), 3 committee questions (D070-D072). **Dropped: all 13 derived cases** (ratios, growth, EBITDA, free cash flow), as agreed, since the system does not compute.
- Rows: `route DOCUMENT`, `answerable true`; numbers are `slice table, type numeric`, committees `slice text, type narrative`. Gold answers read like `Rs 3,124.83 million (standalone, FY2025)` (negatives in the report's own `(692.21)` form; shares and EPS have their own units). `author` = "sifra-v2 golden set"; `notes` carries the original sifra case id.
- **Mechanical check:** every gold figure (and committee name) was looked up in the PDF text of its gold page(s); 72 of 72 found. This is not the two-person check: `verified_by` is empty on every row, so the validator still gives 72 warnings. The report has no PDF page labels, so printed page = PDF page (`page` == `pdf_page`).

**Changes to the wording of 3 questions (everything else is verbatim)**
- Financing cash flow: "net cash **used** in financing" became "net cash **from** financing", because the gold figure is a positive 519.20.
- Two committee questions mentioned "knowledge graph" (a Sifra Neo4j notion); they now simply ask whether the committee existed in FY2025. The answer is "Yes" plus the committee name, so they are narrative rows.
- Multi-year gold pages are the statement page the comparative column sits on (p.56), not every page that repeats the figure.

**First real retrieval numbers** (`python -m eval.retrieval_eval`, real index of the FY25 report, 72 questions, git `fbf694e+dirty`)

| slice | n | R@1 | R@3 | R@5 | R@8 | R@5 +/-1 | MRR |
|---|---|---|---|---|---|---|---|
| overall | 72 | 41.7% | 66.7% | **77.8%** | 83.3% | 91.7% | **0.561** |
| table | 69 | 43.5% | 69.6% | 79.7% | 85.5% | 94.2% | 0.581 |
| text | 3 | 0% | 0% | 33.3% | 33.3% | 33.3% | 0.083 |

- 16 misses at 5. Most are the three-page standalone statements (pp.56-58) confusing each other: a p.57 cash-flow question retrieves p.58 or p.56 (hence +/-1 at 91.7%), plus a few where the statement page is outranked by narrative pages (p.32, p.53). Two committee questions miss p.37. These are real inputs for the step 10 experiments (BM25 / hybrid, chunking of statements, query prefix).

**Limits of this set (it is not the ~85-row design target yet)**
- All DOCUMENT and answerable; **no near-miss unanswerables, no GENERAL, no MIXED**, no scanned slice. Router accuracy, abstention and the router benchmark in step 09 need those rows.
- 51 of 72 gold pages are p.56, so results lean on one page. The text slice is 3 rows. Treat per-slice numbers for `text` as anecdotes.
- A full answer run over 72 rows is ~160K tokens on the 120b model against a 200K/day free cap: use `--slice` / `--limit` and the dev cache.

**Open TODOs (human)**
- Second person verifies the gold pages and fills `verified_by`.
- Write near-miss, GENERAL and MIXED rows (and a few narrative/text rows) before step 09 is meaningful for the router and abstention.
- Add `GEMINI_API_KEY` to `.env` for the judge.

**Update: 12 abstention rows added (A01-A12, written by Claude, `author: claude`)**
- `type near_miss`, `answerable false`, route DOCUMENT, no gold pages; the eval set is now **84 rows** (72 answerable, 12 unanswerable; 74 table / 10 text). Each row's `notes` says why the report cannot answer it.
- Wrong period (FY2030 revenue forecast; FY2022 revenue; total assets at 31 Mar 2022; ROE FY2023; FY2022 operating cash flow), a period the report has no results for (profit for the quarter ended 31 Dec 2024), and things the report never discloses (credit rating, order book, market share, Scope 1 emissions, share price, FY2027 capex guidance). Each sits next to real content (e.g. 31 Dec 2024 appears in a bank-return table; "market leadership" is described without a figure), so retrieval finds plausible pages and the LLM gate has to say no.
- Checked by script: every "must be absent" phrase/pattern has zero hits in the full PDF text. FY2023 figures were left out on purpose (the p.24 three-year chart answers them). No "different company" rows: per the router rule, those are GENERAL, not DOCUMENT abstentions. Still not independently verified by a second person.
- **Finding for step 10: gate 1 cannot separate these.** Mean top retrieval score is **0.794 for answerable vs 0.801 for unanswerable** (max 0.860), so no theta works on this corpus; the LLM's INSUFFICIENT carries all abstention. Budget a real look at hybrid retrieval / a different gate rather than a theta sweep alone.
- `GEMINI_API_KEY` is now in `.env` (git-ignored) for the step 09 judge; not yet used or tested.

**Update: GENERAL and MIXED rows added; the set is now 117 rows (all routes covered)**
- **18 GENERAL (G01-G18):** 10 definition traps for terms the report itself uses (working capital, EBITDA, depreciation vs amortisation, deferred tax, bonus issue, EPS, Audit Committee, air separation unit, standalone vs consolidated, CSR), a few more definitions (balance sheet, IPO, inflation), 3 plain facts (capital of France, Hamlet, boiling point of liquid nitrogen), and 2 router-tilt rows (G17 "What does Linde India Limited do?", G18 "Zenith Foods Limited's revenue in FY2025", a fictional company; the right outcome is GENERAL and no invented figure). `gold_answer` is a short reference for the judge.
- **15 MIXED (M01-M15):** one sentence with an FY25 document fact and an unrelated general question, in varied word orders ("..., and what is the capital of France?", "Who wrote Pride and Prejudice, and what was ...?"). Besides the three required gold fields, each row has `gold_answer` + `gold_pages` for its document part (value checked against the gold page text by script); M15 is the CSR-Committee question plus "what is CSR in general".
- Totals: DOCUMENT 84 (72 answerable + 12 near-miss), GENERAL 18, MIXED 15; slices table 74 / text 10 / general 18 / mixed 15. `python -m eval.validate --strict` passes with 0 errors and 0 warnings. Still **no scanned slice** (needs tesseract, i.e. Docker, and a scanned report).
- All G/M/A rows were written by Claude (`author: claude`); general-knowledge answers are not independently checked. M-row general parts include one time-sensitive fact at most (none are about current office-holders).

**Verification is now a recommendation, not a gate (your call)**
- `eval.validate` no longer warns per row for a missing `verified_by`; it prints one line, `info: N row(s) with gold pages have no verified_by yet (recommended ..., not required)`, and `--strict` ignores it. `eval/README.md` section renamed "Verification (recommended, not a gate)". MIXED rows may carry gold pages (for their document part) without a warning. Tests updated/added (`tests/test_eval_tooling.py`).

**Fixed: the intermittent `test_eval_end_to_end_on_the_fixture_with_an_offline_embedder` failure (open since step 05)**
- Cause: the test's `BagOfWordsEmbedder` bucketed words with `hash(w) % 1024`; Python salts string hashes per process, so on about 6% of processes (3 of 53 `PYTHONHASHSEED`s tried: 32, 44, 58) unlucky word collisions dropped recall and the asserted thresholds failed. Not a retrieval or step 04-07 bug.
- Fix: `zlib.crc32` instead of `hash()` (deterministic). Seeds 32/44/58 plus 100-130 now pass. The earlier `test_ui_helpers::test_upload_list_ask_and_rate_through_the_client` failure (one run) was not reproduced and is probably unrelated.

## Step 09: Full eval runner (router benchmark, judge, abstention, latency)

**Built**
- `python -m eval.run --config config.yaml [--slice ...] [--limit N] [--no-llm-judge]` (plus `--router-only`, `--report-only`, `--run`, `--fresh`, `--rejudge`, `--no-cache`, `--no-pace`, `--markdown`, `--no-mlflow`). One question = retrieval (step 04 metrics) + the real `QueryService` (new `run_detailed`, the same path as `/query`; the request log is off so the Metrics page is untouched) + the two baseline routers. One record per question goes into `eval/results/runs/<run>.jsonl`; the report is rebuilt from that file (`eval/aggregate.py`, pure functions) and written to `eval/results/latest.json`, the console, an MLflow run (params, metrics, git hash, eval-set hash, prompt versions; skip with `--no-mlflow`) and optionally Markdown.
- **Resumable and quota-aware.** Re-running skips finished questions; `--limit N` means N *unfinished* questions; a question that hit an LLM outage is retried; a judge outage leaves verdicts pending and a later run fills them without re-running the pipeline; an edited row is re-run; the run file stores a fingerprint (models, prompt versions, theta, top-k, chunking, embedding model, indexed docs) and resuming under a different setup is refused. Dev LLM cache on; calls paced per model to `eval.tpm_budget` (6,500 tokens/min, the Groq cap is 8,000); judge spaced by `eval.judge_min_interval_seconds`. Latency is reported only for questions with no cached call.
- **Metrics:** router accuracy + 3x3 confusion + DOCUMENT recall for keyword rules / retrieval-score threshold / LLM router (`eval/router_baselines.py`); Recall@1/3/5/8 and MRR (strict, ±1); numeric answers by exact number match (`app/answering/numbers.py`), other answers by the judge; groundedness; citation accuracy (strict, ±1, all cited pages gold); abstention precision/recall, false-answer and wrong-abstention rates; number-check failure rate; latency p50/p95/p99 per stage; tokens and list-price cost; all by slice (text/table/scanned, plus general and mixed blocks). Unjudged rows are counted and left out of rates, never counted wrong; LLM-outage rows are excluded and listed.
- **Judge:** `prompts/judge_v1.yaml`, Gemini (`gemini-2.5-flash`) through its OpenAI-compatible endpoint, behind a `Judge` interface (`eval/judge.py`). One call returns `correct` (vs the gold answer) and `grounded` (vs the cited text). **Gemini's hidden thinking tokens count against `max_tokens` and truncated the JSON verdict** (seen live), so the judge runs with `llm.judge_reasoning_effort: none`. Numeric rows are judged too, which gives a free judge-vs-number-match agreement figure.
- **Calibration:** `python -m eval.calibrate export --n 20` (half the answers the judge called wrong; the judge's verdict is hidden from the sheet) and `score` (% agree and Cohen's kappa: judge vs A, vs B, vs both, A vs B; warns below 80%).
- **Online judge:** `scripts/judge_recent.py --last N` writes `requests.judge_correct/judge_grounded` for the Metrics page. See the first deviation below.
- Tests: about 200 new (`tests/test_eval_*.py`, `tests/test_online_judge.py`, `tests/evalrecs.py`): metrics on hand-made data, baselines, judge parsing/pacing/failures (mocked), record and resume rules, the aggregator on a hand-computed 11-record scenario (mutation-checked), the runner end to end on a real ingested index with a scripted LLM and judge, the CLI, calibration, the content log. Full suite green, ruff clean.

**Run**
```
python -m eval.run --router-only --no-llm-judge --run main     # small model's quota only (what was run, below)
python -m eval.run --run main --slice general --slice mixed    # then answers, in chunks (see "Not done")
python -m eval.run --run main --report-only --markdown         # rebuild the report from the run file
python -m eval.calibrate export --n 20   /   python -m eval.calibrate score
python scripts/judge_recent.py --last 50
```

**Results so far: router benchmark and retrieval only** (run `main`, git `0201c2f+dirty`, eval set `4ae0ab0c55ad`, all 117 rows, 0 LLM outages, 1,110 s, router `gpt-oss-20b` prompt v1)

| Router | Accuracy | DOCUMENT/GENERAL rows only | DOCUMENT recall | MIXED recall |
|---|---|---|---|---|
| Keyword rules | 95.7% (112/117) | 99.0% (101/102) | 100% (84/84) | 73.3% (11/15) |
| Retrieval-score threshold (best tau 0.724 on this set, optimistic) | 84.6% (99/117) | 97.1% (99/102) | 98.8% (83/84) | 0% (0/15) |
| **LLM router** | **99.2% (116/117)** | 99.0% (101/102) | 98.8% (83/84) | **100% (15/15)** |

- The LLM router's one miss is A10 (closing share price on the NSE on 31 March 2025, an unanswerable question) sent to GENERAL, so it would answer from general knowledge, with the label, instead of abstaining. 0 router fallbacks (bad JSON).
- **Honest reading:** on this set the keyword rules already reach 99% on DOCUMENT/GENERAL questions (their one miss is G18, the fictional company with "FY2025" in it). The LLM router's real advantage is MIXED (100% vs 73% for rules; the score baseline cannot split at all) and the fictional-company case. The set is probably easy for rules: nearly every document question names a fiscal year or the company, and most GENERAL questions are "What is X?". A hybrid (rules first, LLM only when unsure) is a possible cost saving; a harder question set would tell us more.
- Retrieval (identical to the earlier hand run, 72 answerable rows): Recall@5 **77.8%**, ±1 **91.7%**, MRR **0.561**; table slice (n=69) R@5 79.7%; the text slice has only 3 rows, so treat it as anecdotal.

**Not done (on purpose, as instructed): the full pipeline**
- No answers were generated, so there are **no numbers yet** for answer correctness, groundedness, citation accuracy, abstention, number-check failures, latency or tokens. The Gemini judge was exercised live on 3 questions only (a smoke run), and **judge calibration has not been done** (it needs judged answers, then two people labelling 20).
- Budget for whenever it is run: about 99 document answers at ~2,300 tokens each plus the general answers is ~260K tokens against the 120b model's ~200K/day free cap, so plan two days and use `--slice`/`--limit`. Router replies are already cached, so the router part costs nothing again. Use `--no-cache` for a latency run.

**Deviations / notes**
- **The online judge needs text the request log deliberately does not keep.** Added an opt-in `observability.log_content` (default **false**) that stores question, answers and the cited chunks' full text in a new `request_content` table (separate from `requests`, purge it independently). `DocAnswer.cited_texts` carries the chunk text (never in an API response). Without a gold answer, online `correct` means "answers the question and agrees with the cited sources", which is not the same construct as offline correctness, and only the document part of an answered request is judged.
- `QueryService.run_detailed()` is new (`run` wraps it); it returns the raw per-path results for the eval.
- The score-threshold baseline's tau is tuned on the eval set (the baseline's best case). The keyword rules were written from the design's description, not tuned on the set.
- Numeric match is exact (like the number check): same value, or same digits when one side names no scale; two different scale words with equal digits do not match; signs and `%` are ignored. It is used for `type: numeric` rows, and for a MIXED row's document half when its gold answer is a figure. A correct derived figure not printed in the source would fail it, as in step 05.
- A question misrouted away from the document path counts as `other` in the abstention block, not as a false answer.
- `.env` is now read by the eval tools (`eval/env.py`, no new dependency; real environment variables win). The API still reads real environment variables only.
- Fixed a racy assertion in `tests/test_ui_helpers.py::test_upload_list_ask_and_rate_through_the_client` (it failed about 1 run in 3 alone): the 202 can legitimately report `PROCESSING` because the endpoint reads the row after handing the job to the worker.
- A background run was killed once by Claude Code's low-memory guard; the run file kept 21 records and the rerun resumed from there.

**Open TODOs**
- Run the answer pass (general and mixed first, then table and text in chunks), judge it, calibrate the judge with two labellers, and paste the Markdown tables into the README (step 12).
- Harder router questions (paraphrases, a company named without a year, GENERAL questions with report vocabulary) before trusting 99%.
- A10-style near-misses (share price, rating) that the router may send to GENERAL: decide whether the router prompt needs an example.
- Step 10 inputs: Recall@5 77.8% (statement pages 56-58 confuse each other); theta cannot separate answerable from unanswerable (step 08 note); the keyword baseline suggests a rules-first hybrid is worth testing.
- Still no scanned slice (needs Docker/tesseract and a scanned report).

## Step 09 follow-up: why the system abstained on D001-D021, and the fix

**Finding (not a model problem, a retrieval problem the page-level metric hid).** The first answer pass declined 13 of the first 15 document questions as `insufficient`. Inspecting what reached the model:
- On the EIG report's statement pages (landscape, two printed pages per PDF page) the table chunks held only numbers. The statement's name ("Standalone Balance Sheet ... (All amount are in ` million)") sat in a separate *text* chunk next to the signatures, so a table chunk never said what it was, nor whether it was standalone.
- Dense retrieval (bge-small) ranked the signature text chunk first and the table chunk holding the figure 26th-58th of 475. The strict page metric still counted it a hit (right page), which is why Recall@5 read 77.8%. A new check, **"the gold figure is inside a top-5 chunk on a gold page"**, was **1 of 69** numeric rows. The model was shown no figure, so declining was correct behaviour.

**Fix (all measured offline first, no LLM calls).**
1. *Table titles* (`app/ingestion/pdf_parse.py: table_titles`, `tables.py`, `chunking.py`): the text blocks printed just above a table, in its own column (up to 3 blocks, 300 chars, repeated page furniture ignored), are put on the first line of every piece of that table. They are embedded and shown to the model ("standalone", the unit). `INGEST_VERSION` 1 -> 2; `scripts/reingest.py` re-runs ingestion for outdated documents (no LLM key needed).
2. *Hybrid retrieval* (`app/retrieval/bm25.py`, `retriever.py`, `retrieval.mode/bm25_weight/pool/rrf_k` in config): in-memory BM25 (about 60 lines, no new dependency) fused with the dense list by weighted reciprocal-rank fusion. A chunk's `score` stays its cosine similarity and `top_score` is the best cosine among the returned chunks, so the theta gate keeps its meaning. `mode: dense` restores the old behaviour for experiments.

| 69 numeric rows, top-5 | gold page | gold figure in context |
|---|---|---|
| dense (before) | 55 | 1 |
| BM25 only | 69 | 55 |
| RRF 1:1, pool 50 | 61 | 32 |
| **RRF, BM25 weight 3, pool 50 (adopted)** | **68** | **52** |

The 3 narrative rows: gold page in the top 5 for 1/3 (dense) vs 3/3 (hybrid). This clears the design's adoption rule (>= 5 points of Recall@5) by a wide margin, so it was pulled forward from step 10. **Caveat:** the eval questions reuse the statements' own line-item words, which favours BM25; weight 3 was chosen on this same set, and paraphrased questions are not tested yet. Pure BM25 scored slightly higher here; hybrid was kept as the safer default.

**Result (run `fix1`, D001-D021, 21 rows, fresh index): 19/21 correct, 0 number-check failures, all 19 answers cited a gold page (strict).** Before: 2 of 15 answered. Tokens: 73K (3.5K per question including the router).
- D007 (net cash from financing) still abstains: its table chunk ranks 6th, one place outside the top 5. Ranking-tuning question for step 10 (weight, top-k, reranker).
- D002 (closing borrowings, movement table) still abstains: the right chunk is 2nd and contains 2,452.96, but its title ("(c) Movement in borrowings and lease liabilities") never says "standalone" and the question does; the answer prompt's rule 3 makes the model decline. Conservative by design.
- Recall on this slice: R@1 47.6%, R@5 95.2%, R@8 100% (21 rows only).

**Also changed**
- `eval.judge_max_calls` (default **10**) / `--judge-limit N` (0 = no limit): live judge calls per invocation, because Gemini's free tier allows about 20 a day. Cached replays are free; answers beyond the budget keep their verdicts pending and a later run fills them. In `fix1` the judge got 1 verdict (D001, agrees with the number match) before Gemini returned 429 "quota exceeded" (the daily quota was not yet back), after which the runner stopped asking, as designed.
- The run fingerprint now includes the retrieval mode, BM25 weight, pool and `INGEST_VERSION`, so old runs are refused rather than mixed (the old `main` D001-D021 records came from the broken index; use new run names).
- Known report quirk: the latency block reports "0 questions with no cached call" because the router call is always replayed from the cache; a latency run needs `--no-cache`.
- Tests: 631 pass (new: `tests/test_bm25.py`, hybrid cases in `test_retrieval.py`, table-title cases in `test_ingestion_parsing.py`, judge-budget cases in `test_eval_run.py`).

**Still open:** answer pass for D022-D072 (about 50 questions, roughly 175K tokens at 3.5K each: spread over days, `--slice table --limit N --run fix1`), general/mixed rows under the new fingerprint, judge verdicts (10 at a time), calibration. A retrieval check on the figure-in-context measure should become part of `eval.retrieval_eval` before step 10.

## Step 11: Langfuse tracing, load test, Ollama fallback

All three are optional, off when their env vars are missing, and cannot block or fail a request. How-to: `docs/optional-features.md`.

**Built**
- **Langfuse tracing** (`app/observability/tracing.py`, Langfuse Python SDK v4 used directly, `langfuse>=4,<5` added to `requirements.txt`). On only when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are both set (`LANGFUSE_BASE_URL` picks the server; `LANGFUSE_TRACING_ENABLED=false` switches it off). Span tree per request: `query` > `router` (generation) / `document` > `retrieve` + `generate` (generation) / `general` > `generate`. Prompt name + version, model ids, token usage, backend, status, abstain reason, top score, retrieved chunk ids and scores, and cited chunk ids are on the spans. The no-op `Tracer` is the default; every SDK call is wrapped (a `tracing_failed` warning, never an exception); the SDK exports in a background thread; the app flushes on shutdown. The trace id is derived from the request id (`create_trace_id(seed=...)`), so late scores land on the right trace from any process.
- **Scores:** `/feedback` sends `user_feedback` (1 / -1); `scripts/judge_recent.py` sends `judge_correct` / `judge_grounded` (boolean) for each answer it judges.
- **Content policy:** by default no question, answer or document text leaves the machine (design §15: no document text in traces). `tracing.capture_content: true` adds the question, the answer and the 300-char cited snippets, never whole chunks. One place enforces it (`LangfuseTracer.clean` drops `input`/`output`).
- **LLM backend + fallback** (`app/llm/client.py`, `app/config.py`): `llm.backend` / `LLM_BACKEND` = `groq` | `ollama`; the model names live in `llm.ollama` in `config.yaml`. `OLLAMA_BASE_URL` turns the automatic fallback on: after the client's own retries, a Groq timeout / 429 / 5xx / connection error repeats the call once on Ollama (the router model maps to `ollama.router_model`, everything else to `ollama.answer_model`), and Groq is skipped for `llm.fallback_cooldown_seconds` (30) so later requests do not each wait out the retries. A rejected request (400/401) never falls back. `LLMError.retryable` marks the difference. Fallback answers are not written to the dev cache; the Gemini judge client never gets a fallback.
- **Request log:** new columns `llm_backend` (`groq`, `ollama`, `groq+ollama`; NULL = no LLM call), `llm_fallbacks`, `llm_rate_limited` (429s seen), added by `init_db` to existing databases. They come from a per-request context (`app/llm/request_stats.py`, shared by the two parallel paths of a MIXED question) and are also returned as an `llm` block in the `/query` response. The Metrics page shows a caption: requests that saw a 429 / fell back.
- **Load test** (`scripts/load_test.py`, asyncio + httpx): 1 / 3 / 5 concurrent users send a seeded mix (50% document, 30% general, 20% mixed) of `eval/questions.jsonl`; it reports req/min, p50/p95, error rate, 429 count, tokens and a verdict per level, where it first breaks, and writes `eval/results/load_test.json` (that folder is git-ignored). Small by design: per level at most `--max-requests 6` and `--duration 30` s, a `--token-budget 60000` for the whole run, 65 s cool-down between levels.
- Tests: **61 new** (`tests/test_tracing.py` 20, `tests/test_llm_fallback.py` 22, `tests/test_load_test.py` 18, 1 in `test_metrics.py`) on a fake Langfuse client that keeps parent/child links through threads, scripted SDKs, and a scripted API. The tracing tests include "no keys: no-op", "a client that raises everywhere does not fail the request", "content dropped by default", the MIXED tree, and score ids. I broke the code 8 ways (no `retryable` check, no cooldown, caching fallback answers, content not dropped, swallowing the caller's exception, no feedback score, env var not enabling the fallback, per-level token budget) and each was caught. Full suite **692 passed, 1 skipped**, ruff clean. `tests/conftest.py` now removes `LANGFUSE_*`, `OLLAMA_BASE_URL` and `LLM_BACKEND` from the environment for every test.

**Run**
```
# tracing: keys in .env, then (Docker Compose loads .env into the API container)
docker compose up -d --build
# fallback: in .env   OLLAMA_BASE_URL=http://host.docker.internal:11434/v1   and   ollama pull gpt-oss:20b  (or llama3.2:3b)
# load test (small; spends Groq quota)
python scripts/load_test.py --levels 1,3,5 --max-requests 5 --token-budget 40000
python scripts/judge_recent.py --last 20        # also puts judge scores on the Langfuse traces
```

**Checked live**
- *Langfuse:* rebuilt and restarted the Docker stack; startup logged `tracing_on`. Two real questions (general, document) plus a 👍 on the document one, read back from Langfuse: the span trees are as above with the right parent for every span (checked from raw parent ids), `openai/gpt-oss-120b` / `-20b`, token usage, prompt `router_v1` / `answer_doc_v1` + `v1`, `backend: groq`, retrieved ids, and a `user_feedback = 1.0` score. No question or answer text on the trace (the default policy). Not run live: judge scores (the Gemini quota is exhausted; they use the same `score()` call as the 👍) and `capture_content: true` (unit-tested only).
- *Fallback:* no Ollama is installed here, so **not run against a real Ollama**. Checked over real HTTP with the real `openai` SDK against two local stand-in servers (one always answering 429 with `Retry-After`, or hanging past the timeout; one answering): both cases fell back with the right model names, and the second call skipped the primary. Whether `gpt-oss:20b` / `llama3.2:3b` accept `reasoning_effort` and `response_format: json_object` through Ollama's OpenAI endpoint is **unverified**.
- *Load test* (real API + real Groq + the EIG FY25 AR; 5 requests per level, the same 5 questions at every level because the plan is seeded and as long as the cap; 40.6K tokens in total):

| Users | Requests | req/min | p50 ms | p95 ms | Error rate | Degraded | 429s | Tokens | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 32.7* | 1384 | 2990 | 0.0% | 0 | 0 | 15,016 | ok |
| 3 | 5 | 112.3* | 1369 | 1530 | 20.0% | 1 | 1 | 12,778 | broken |
| 5 | 5 | 184.6* | 1451 | 1591 | 20.0% | 1 | 1 | 12,762 | broken |

  \* a burst of 1.6-9 s, so req/min is extrapolated and **not sustainable** (the three levels ran at 98K / 284K / 479K tokens per minute against Groq's 8,000 per model). Reading: **latency is fine (p50 about 1.4 s) until Groq's token-per-minute cap, not our code, is the limit.** The first 429 came at 3 users. The failed request in each of the 3- and 5-user levels is the same event: `gpt-oss-120b` TPM limit 8,000, "Used 7302, Requested 2294, try again in 11.97 s" (and 13.4 s), longer than `retry_max_wait_seconds` (10), so the client failed fast after one attempt and the answer part came back `error` / `llm_unavailable`. With 1 failure in 5, "broken" (>= 5% errors) is a single request, so read it as "the free tier cannot serve even three people asking within the same minute", not as a precise threshold. One question costs about 1.4K (general) to 3.5K (document) tokens, so the 8K cap is about 2-3 document questions a minute and the 200K/day cap about 60 of them. This also explains the step 08 note (a 5xx "after 1 attempt(s)" despite `max_retries: 2`): a Retry-After above the 10 s cap fails fast, and that is now the case the fallback covers.

**Deviations / notes**
- **v4 of the Langfuse SDK has no `@observe` plus `start_as_current_span` pair; it has `start_as_current_observation(as_type=...)`.** I used explicit, wrapped context managers (the same idea, but every call can be guarded so tracing can never fail a request) rather than decorators.
- **Langfuse's legacy trace / score read APIs return HTTP 410 for organisations created after 2026-09-16**; reading traces back needs `api.observations.get_many` and `api.scores_v3.get_many_v3`. Writing is unaffected.
- The offline eval runner is **not** traced (the tracer is a no-op until the API's startup or `judge_recent.py` installs one), and offline judge verdicts are not sent to Langfuse; only online ones are.
- **`eval/results/` is git-ignored**, so `load_test.json` stays local; the table above is the record. Its two derived fields (`tokens_per_min`, `extrapolated`) were added to the saved file from its own numbers after I added them to the script, so a rerun would produce the same shape.
- The fallback's cooldown is process-local state (a plain float): fine for one uvicorn worker.
- `docker-compose.yml`: `extra_hosts: host.docker.internal:host-gateway` on the API service (Docker Desktop has the name already; Linux needs the line).
- The load test's 429 count is "Groq 429 responses seen by the request's calls, retried or not", taken from the `/query` response's new `llm` block, because the API itself answers 200 even when the LLM failed.

**Open TODOs**
- Pull a model and run the fallback against a real Ollama; check the two request parameters above and the first-call latency.
- A longer load test (needs a day with quota, or a paid Groq tier) for a sustained req/min; consider `retry_max_wait_seconds` and a paced queue in front of the LLM instead of failing at 12 s waits.
- Run `judge_recent.py` with `observability.log_content: true` once the Gemini quota is back, and look at the judge scores on the traces.
- Metrics page: no coloured indicator for the backend mix yet (a caption only).

---

## Fixes after the first live test (2026-10-07)

**What changed**
- **Text pages first, OCR pages last** (`pdf_parse.parse_pdf`). Pages are classified up front; every page with a text layer is parsed and yielded in one fast sweep, then the scanned pages are OCR'd. Found live: asked during ingestion, "total equity" was answered from the share-capital line on p.35 (₹261.87M) because the balance sheet on p.78 was not indexed yet. With text first, p.78 is in the index within seconds.
- **₹ drawn as a backtick** (`tables.fix_rupee`). The EIG report's font maps the rupee sign to `` ` ``, so answers read "\` 3,124.83 million" and a stray backtick can open a Markdown code span. Restored to ₹ before an amount or a unit word; real code spans are untouched. 105 occurrences in the EIG FY25 AR.
- **`INGEST_VERSION` 2 → 3**: re-index with `scripts/reingest.py`.
- **Coverage wording**: "searched pages 1-40 of 86" became "searched 40 of 86 pages" (the indexed pages are no longer always the first N). Page timings are stored in page order.
- **Stronger caveat on answers from a still-processing document**: "may be incomplete or wrong. Ask again when the document is Ready.", shown above the answer instead of below it.
- **UI restyle** (chat-panel layout): dark navy theme with a purple accent (`.streamlit/config.toml`), DocQA logo in the top bar (`ui/assets/`), the conversation inside one rounded panel with "You" bubbles on the right and "AI" bubbles on the left, pill suggestion chips with Material icons, a rounded input with a separate Send button (a form, so Enter also sends), a Remove option per document (⋯ menu), and quoted source snippets. Styling is CSS on Streamlit's stable hooks (`data-testid`, `st-key-<key>` classes) in `ui/components.py`; restart Streamlit after editing that file (it is imported, so a save does not reload it).

**Tests**: 699 passed (7 new: text-before-OCR order, ₹ fix, caveat placement, example chips, You/AI labels, empty question not sent, remove document). The UI tests now type into the composer form instead of `st.chat_input`. `ruff check .` clean.

**Open TODOs**
- Re-run the async test (ask while processing) on the EIG report and record time-to-searchable for p.78.

**Follow-up (same day): keyword search's #1 always reaches the model** (`retrieval.bm25_keep_top: 1`). Found live on a Mac: "How much did the company spend on purchases of property, plant and equipment in FY2025?" was refused on a fully indexed report. Keyword search ranked the cash-flow table (p.57, ₹692.21M) **#1**, but the meaning-based search ranked it about 28th on Linux and below its 50-chunk pool on the Mac, so fusion (which rewards chunks both lists find) let accounting-policy prose (p.59, p.65) push it out of the top 8. Weighting keyword search higher did not fix it. Now the keyword winner is moved into the last of the `top_k` places when fusion leaves it out. On the EIG report, reproducing the Mac ranking: missing → position 5 (top_k 5). 702 tests pass (3 new).
- **CI retrieval gate:** `eval.retrieval_eval --fixture --check` reports MRR 0.752 vs the 0.857 baseline **here**, but the original code (before any of today's changes) gives the identical 0.752 in this environment, and this change moves it by 0.000. It looks like a library-version difference; check it on GitHub CI.

**Follow-up (2026-10-08)**
- **Page heading on every chunk** (`pdf_parse.page_heading`, `chunking.chunk_page`; `INGEST_VERSION` 4). A page's biggest text near the top ("Standalone Balance Sheet as at March 31, 2025") is put in front of every chunk from that page. Found live: Fortis "total equity" was refused because the chunk holding "Total equity (A) 9,07,400.25" had none of the words "standalone", "balance sheet" or "March 31, 2025". Keyword rank of the right chunk: Fortis total equity not in top 50 → 2; HPCL total equity not in top 50 → 5. Checked live: Fortis and HPCL total equity both answered correctly.
- **Comfy dark UI**: warm cocoa-brown surfaces, one honey accent, Nunito, rounded cards with soft shadows; your question in a bubble on the right, the answer on a card with numbered sources ([1], [2]) that open to the quoted page; composer docked at the bottom. New logo: an open ring binder with one lined page lifted out of it.
- **Tesseract on Windows without PATH**: `pdf_parse.tesseract_cmd()` uses `TESSERACT_CMD` if set, else (Windows only, when `tesseract` is not on PATH) the default install folder. Linux/macOS unchanged.
- `upload.max_pages` 400 → 500 in config.yaml (the HPCL and Fortis FY25 reports are 436 and 427 pages).
- Multi-report note: with several reports loaded, a question that does not name the company searches all of them ("total equity ... standalone balance sheet" returned Fortis's figure while HPCL was meant). Name the company.

**Tests**: 709 passed. `ruff check .` clean.

**Faster ingestion (2026-10-08)**: measured first, on the Fortis FY25 AR (427 pages): embedding with bge-small was ~94% of ingestion (~2.4 s/page; 613 ms/chunk on a 2-core box), parsing ~6% (table detection 145 ms/page).
- **Static embeddings** (`embedding.backend: model2vec`, `minishlab/potion-retrieval-32M`): all 1,576 Fortis chunks embed in 0.45 s (~2,000x faster). Static vectors are weaker at meaning, so keyword search leads: `retrieval.bm25_weight` 3 → 20. EIG eval (72 answerable questions): bge-small at weight 3 R@1 65.3% / R@5 98.6% / MRR 0.781 → static at weight 20 R@1 72.2% / R@5 98.6% / MRR 0.819. The score scale changed (lowest answerable top score 0.37 vs 0.72), so `theta` 0.5 → 0.25. bge-small stays available: `backend: fastembed` (then `bm25_weight: 3`, `theta: 0.5`).
- **Parallel parsing**: text pages are parsed in worker processes (`parsing.workers`, 0 = cores-1, max 8; documents under 24 pages stay in-process), results in page order so PARTIAL still works.
- **Keyword search sees the document title** (as the embeddings already did), and **ordinals fold to numbers** ("31st March" matches "31 March"). With both Fortis and HPCL loaded, the passage holding the figure now reaches the model for 5 of 6 statement questions; the miss is Fortis "standalone profit" (the figure is in the P&L table chunk, keyword rank ~25, behind the same page's prose chunk).
- **Dropped after measuring**: a "skip table detection on pages without drawings" gate (every annual-report page has drawings: 0 of 427 Fortis pages skipped) and a number-density gate (saved 11-22% of table time while skipping real tables).
- `top_k` 5 → 8 (with 5, a label-less table chunk could push out the labelled figure on long reports).
- Timings on a 2-core box: Fortis 427 pages ~17 min → 68-75 s; HPCL 436 pages 83 s; EIG 86 pages 389 s → 35 s. More cores parse faster.
- **CI baseline rewritten** for the new model (fixture MRR 0.699; bge-small gave 0.752 on the same machine; Recall@5 100%).
- Changing the embedding model means re-indexing: start with an empty `data/` and re-upload.

**Split statement tables (2026-10-08)**: "What was Fortis's standalone profit for the year ended March 31, 2025?" was refused with both reports loaded. Found: the P&L was cut into two 400-token pieces; the half holding "Profit for the year 6,378.44" ranked #16, the half without it #4, so the model saw the right page but the wrong half and (correctly) refused. Researched first (financial-report chunking paper: tables as whole units; LangChain table benchmark; LlamaIndex auto-merging "small-to-big"; row-level serialisation; table summaries). Changes:
- **Missing outer table columns** (`tables._open_edges`): statements often rule every row across all columns but draw no border on the outer edge, so `find_tables` stopped at the last vertical line and dropped the outer column. Fortis p.195/p.299 lost the FY2024 column; the standalone balance sheet (p.194) lost its Particulars column, so its rows had no labels. A side is closed (and the page redetected) only when words sit beyond the table on a third of its rows and the table's own horizontal rules run on over them; off-page crop marks are ignored. 98 Fortis pages and 14 HPCL pages gain a column (row labels or prior-year figures); EIG 0. Cost ~17 s CPU on Fortis, spread over the parse workers.
- **Small-to-big for tables** (`retrieval.table_siblings`, `retriever.fit_context`): when a piece of a split table reaches the model, the table's other pieces (same page, same repeated title and header row) follow it. All passages are capped at `answer.context_max_tokens` (5,000; Groq free tier is 8,000 TPM).
- **₹ in table headers**: "(` in Lakhs)" and "Basic (in `)" now restored too.
- **Tried and dropped**: keeping whole tables as one chunk (800-token table budget). Long chunks rank lower in both searches: statement figures reaching the model fell to 11/15 (vs 12/15 with the column fix alone, 14/15 with small pieces + siblings).
- Measured with Fortis + HPCL loaded, 15 statement questions (incl. FY2024 figures, consolidated, EPS, total assets): **14/15 figures reach the model**. EIG eval unchanged (R@1 72.2%, R@5 98.6%, MRR 0.819); CI gate passes (MRR 0.676).
- Remaining miss: HPCL standalone total equity (#12). With labels restored, Fortis's balance-sheet rows now outrank HPCL's for a question naming HPCL (cross-report bleed).
- `INGEST_VERSION` 4 → 5: re-upload documents. 717 tests pass (4 new).

## Improvements 01: trap set + number check (2026-10-08)

- **Trap eval set**: `eval/questions_traps.jsonl` (T001-T010, DOCUMENT, answerable; definition conflicts, IPO total vs company proceeds, computed figures, a negative fact, a crore-vs-million trap) with its own file, so the 117-question set and its run files are untouched. `eval/docs.yaml` gains `report_fy24` and `report_fy26` (`report_a` stays FY25). Printed labels: FY26 = PDF page - 2 (checked on 34, 164, 228, 229, 231, 235), FY24/FY25 as in the existing set. Try it: `python -m eval.validate eval/questions_traps.jsonl`, `python -m eval.retrieval_eval --questions eval/questions_traps.jsonl` (free); the paid run (`eval.run --run traps-<name>`) was not done.
- **Gold-page notes**: "EBITDA 1,162.17" for FY26 is not printed anywhere in the PDF (it is PBT less other income plus finance cost and D&A, which sit on PDF p.34), so T001/T002 are "computed figure" traps. T009: the Corporate Information page (PDF p.2, "Dividend Declared: None") is left out of the gold pages because it has no printed page label to confirm; p.36 and p.86 are kept.
- **Number check** (`app/answering/numbers.py`): `FY 25`, `FY 2026`, `FY25-26`, `Q3 FY26`, `Q 3`, `H 1` are period codes, not figures. A figure missing after the exact lookup now passes as **computed** when it is a difference of two cited figures (exact, either order) or, for a percentage, a share `a/b*100` or growth/decline `(a-b)/b*100` rounded to its own printed decimals. Operands are the mantissas as printed (bare years excluded). `NumberCheck.computed` carries e.g. `"482.06 = 1097.36 − 615.30"`; it is also in the API section (`computed_numbers`) and the UI shows "Computed from cited figures: ..." under the answer. A wrong figure (`482.60`) still fails.
- **Caps**: no computed pass above 400 distinct numbers in the cited text (spec). Added a tighter one for ratios, `MAX_RATIO_POOL = 120`: percentages match only to their printed decimals, so with ~400 random numbers 5 of 7 invented percentages "computed" by chance. Differences are exact and keep the 400 cap. Search is sorted + bisect, about 4 ms for a 400-number pool.
- **Open**: prompt 02 can reuse the operands (they are in the `computed` strings; parsing them back is a regex away, or return them structured if that gets awkward). `tests/test_ui_smoke.py` has 20 failures that also occur without my changes (checked with the changes stashed), and `ruff format --check ui/formatting.py` already flagged the file before; neither was touched.

**Tests**: `tests/test_numbers.py` + `tests/test_eval_tooling.py` (+ document/query/answer API tests) 148 passed; `ruff check` clean on changed files.

## Improvements 02: show in PDF (2026-10-08)

- **What it does**: each citation expander has a **Show in PDF** button. Click it and the cited page appears, cropped around the answer's figures, with the whole printed row shaded yellow (so the label "Finance Cost" is covered) and the figure itself boxed in red. Free (PyMuPDF only, no LLM) and nothing is rendered on the `/query` path.
- **Answer time** (`app/answering/highlight.py::highlight_terms`, called from `DocumentAnswerer`): every citation gets `highlight_terms`, the answer's figures that occur in *that* chunk, written as the chunk prints them (`94.90`, `1,097.36`). A computed figure (prompt 01) contributes its operands instead (parsed from the `computed` string, the "× 100" dropped). The citation with the most terms gets `primary: true`; the UI opens its expander first. Both fields are in the API's citation objects. Plain integers under 3 digits are skipped (they would match half the page).
- **Click time**: `GET /documents/{doc_id}/pages/{pdf_page}/highlight?term=…&ctx=…` returns `image/png` with `X-Highlight-Matches`. The file is the stored upload of the document, never a path from the request. 404 unknown document, 422 page out of range or a bad term (≤ 12 terms, ≤ 32 chars, `[0-9A-Za-z,.%()-]`). `lru_cache` of 64 renders; the UI also caches with `st.cache_data`. `ctx` (the citation snippet) breaks ties when a figure occurs in several rows: the row sharing the most words with the chunk wins.
- **Details**: a hit counts only if it is a whole printed word (`94.90` does not match inside `1,094.90`). The PDF is opened per request and the marks are drawn on the in-memory page; it is never saved.
- **Captions**: "Rows holding the answer's figures are highlighted." / "Figure not found on this page; showing the page." / "Scanned page: highlighting not available." (the last when the citation's `source_kind` is `ocr`).
- **Measured** on the real EIG FY26 report (PDF p.34, `94.90`): 43 ms for the render in-process (70 KB PNG), including opening the PDF. Not measured through a browser.
- **Open (TODO)**: OCR word boxes for scanned pages (Tesseract `image_to_data`), so the highlight also works there. Still no highlighting for general-knowledge answers (no citations).
- **Tests**: `tests/test_highlight.py` (12): render finds one match and crops, unknown term gives the plain page, whole-word matching, tie-break by row words, page out of range, terms (as printed, absent, computed operands), primary citation, endpoint 200/404/422, client + captions. With `test_answer_document`, `test_query_api`, `test_answer_api`, `test_ui_helpers`, `test_ingestion_pipeline`, `test_numbers`: 202 passed. `ruff check app ui tests/test_highlight.py` clean. `tests/test_ui_smoke.py` still has the same 20 failures as before this change (see Improvements 01); the new button is not covered by a UI test.
