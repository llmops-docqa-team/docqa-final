# Addendum v2 — Decisions Locked + Kickoff Prompts

This overrides the main design report wherever they differ.

## 1. Decisions locked (5 Oct)

| Topic | Decision |
|---|---|
| Hosted API | **Allowed.** Groq free plan (gpt-oss-120b answers, gpt-oss-20b router). Ollama stays only as a demo fallback |
| Who implements | **Vinay + Purab only.** Nobody else touches code |
| Inputs | PDFs only: **text PDFs and scanned PDFs**, **tables included** |
| Eval corpus | A fixed set of reports used **only for testing** (see §3). The product accepts any PDF |

## 2. What the scanned-PDF + tables decision changes

| Area | Was | Now |
|---|---|---|
| Scope | Text-layer PDFs only; scans rejected | **Per-page OCR fallback** for scanned pages (Sifra's page check: under 120 chars of text and ≥ 60% image area → OCR) |
| OCR engine | — | **Tesseract** (`tesseract-ocr` in the Docker image, `pytesseract`), run **locally on CPU**, so it uses no API quota. **Benchmark on 5 pages first**; swap only if unreadable |
| Tables | Nice-to-have | **MVP.** Extract with PyMuPDF `find_tables()` → Markdown; each table is its own chunk (long tables split by rows, **header row repeated**). About 50 lines, not Sifra's 2,000-line finance chunker |
| Scanned tables | — | **Accepted limitation.** OCR on tables is weak. We measure it separately and say so in the viva |
| Ingestion time | ~1–3 min per report | Text report ~1–3 min. **Scanned report: estimated 5–15 min for 300 pages (to be measured)** |
| Partial querying | Not in MVP | **Now in MVP, in a minimal form.** Reason: with OCR, waiting 10+ minutes is real. We already upsert every ~20 pages. Status `PARTIAL` means queryable. Rule: if the answer abstains and the document isn't finished, say *"Searched pages 1–140 of 312; the rest is still processing."* |
| Evaluation | One set | Report **three slices separately**: text pages, table questions, scanned pages. Only fully `READY` documents are used for scoring, so results stay deterministic |
| Demo | Upload a small PDF | Upload a small **scanned** 10–15-page PDF live (OCR progress is visible). The full reports are pre-indexed |
| Chunk metadata | page | Add `source_kind` (`text` / `table` / `ocr`) and page label |

## 3. Eval corpus (for you to pick, ~30 min)

1. **Report A:** a real, text-layer annual report **with financial tables** (~15 table questions + narrative questions).
2. **Report B:** a **scanned** report, or **40–60 pages** of one. If you can't find a real scan, take a text report and **rasterise its pages into an image-only PDF** (this mimics a clean scan, so real-world accuracy would be lower; say that).
3. **A tiny 10–15-page scanned PDF** for the live-upload demo.

Original data stays private to the team; the public repo holds only questions, gold pages and gold answers, not the PDFs themselves. Check the licence before committing any PDF.

## 4. Updated plan items (from Sifra review + the above)
- Citations show `p.47 (PDF p.53)`; Recall@5 reported strict **and** ±1 page.
- Router fallback is logged as a warning and shown as a metric.
- Number check uses a deterministic parser (no fuzzy tolerance).
- Chunks stay ~400 tokens with 60 overlap, **inside one page**. Do **not** copy Sifra's 700/200 (bge-small reads only 512 tokens).

## 5. Kickoff prompts for Claude Code (run inside the **new project repo**)

**Setup (once):** put the main design report in `docs/design.md` and this file in `docs/addendum.md`. Add Sifra as a *read-only reference* with `/add-dir <path-to-sifra-v2>`. Never copy anything from `.env` or config.

### Prompt A: Step 0 (skeleton)
```
Read docs/design.md and docs/addendum.md fully. Addendum v2 overrides the design where they differ.
Implement ONLY Step 0 of section 23: repo skeleton, config.yaml, prompts/ folder (empty versioned YAML stubs),
Dockerfile (Python 3.11, include tesseract-ocr), docker-compose with FastAPI + Streamlit,
SQLite init (documents, requests tables), structlog logging with a request-ID middleware, pytest with 1 smoke test,
and a GitHub Actions workflow that runs pytest.
Rules: plain Python, no LangChain/LlamaIndex, no extra services, no features outside Step 0. Never read or print secrets or .env;
read env vars by name only. Sifra-v2 is a read-only reference for patterns: take only what the design lists, rewrite in your own code,
and list each thing you reused with its source path.
When done: show how to run it, run the tests, and stop. Do not start Step 2.
```

### Prompt B: Step 2 (ingestion), after Step 0 works
```
Read docs/design.md (sections 8, 10) and docs/addendum.md. Implement ONLY ingestion:
POST /documents (validate, hash, save, 202), SQLite status (QUEUED/PROCESSING/PARTIAL/READY/FAILED with pages_done),
one background worker thread (re-queue PROCESSING docs on startup),
per-page parsing with PyMuPDF keeping PDF page index + printed page label,
per-page scanned detection (<120 chars text and >=60% image area) -> Tesseract OCR fallback,
tables via find_tables() -> Markdown chunks with header row repeated,
text chunks ~400 tokens / 60 overlap, never crossing a page, heading prefix added before embedding (stored text unchanged),
deterministic chunk IDs, bge-small-en-v1.5 via fastembed, Chroma upsert every ~20 pages, embedding-model guard on startup,
GET /documents for status. Add unit tests for chunking, page labels, scanned detection and chunk IDs.
No retrieval, router or LLM calls yet. Measure and print ingestion seconds per page on a text PDF and on a scanned PDF.
```

### Not for Claude Code
**Step 1 (the eval set) is human work.** Claude Code can help format it, but **you and Purab must write and verify the gold answers and pages yourselves**, or the evaluation means nothing.

## 6. Who does what (suggestion)
- **Code:** Vinay implements end to end (ingestion, query path, metrics, CI). Purab helps on request, e.g. reviewing, testing, or taking a self-contained piece such as OCR/table extraction or the metrics page.
- **Non-code work for the other four:** their call, not mine. The eval questions are the biggest piece of non-code work.
- **The rubric still scores the presentation.** All six presenting must understand the architecture, because individual Q&A performance moves marks and sharing the talk equitably is graded. Budget one walkthrough session for the whole team before rehearsal.
