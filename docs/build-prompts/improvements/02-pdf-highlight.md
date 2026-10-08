# 02 · "Show in PDF": the cited page with the answer highlighted

**Model:** Sonnet 5.5, effort high · **Branch:** `improve/answer-quality-and-highlight`

## Read first

`docs/build-prompts/improvements/README.md`, `app/api/documents.py`, `app/answering/document.py`
(`make_citation`, `Citation`), `app/answering/numbers.py` (after prompt 01: `NumberCheck.computed`),
`ui/components.py` (citation expanders around line 400), `ui/api_client.py`, the last entries of `docs/progress.md`.

## Goal

A user reads an answer, opens a citation, clicks **Show in PDF**, and sees that page of the PDF, cropped
around the answer's figures, with the **whole printed row** lightly highlighted (so the row label like
"Finance Cost" is covered) and the **figure itself** boxed. It must be free (PyMuPDF only, no LLM) and must not
slow down `/query`: the picture is rendered only when the button is clicked.

A prototype on the real FY26 report (Financial Highlights, PDF page 34, figure `94.90`) took about 50 ms in
total: open 10 ms, `page.search_for` 10 ms, widen to row 4 ms, draw + render a cropped PNG about 20 ms (73 KB).

## Today → after

- **Today:** the citation expander shows a text snippet of the chunk (`Report.pdf · p.32`). The user has to
  open the PDF and find the line themselves.
- **After:** the expander also has a **Show in PDF** button. Clicking it shows a cropped page image with the
  row highlighted. If the figure can't be found on the page (or it is a scanned page with no text layer), show
  the plain page with a short caption saying so.

## Design (adjust if the code suggests something simpler)

1. **Which figures to highlight, decided at answer time (cheap).** When citations are built, give each one a
   `highlight_terms` list: the answer's figures that literally occur in *that* chunk's text, as printed in the
   chunk (e.g. `"94.90"`, `"1,097.36"`). For a figure that prompt 01 marked as **computed**, use its operands
   instead (e.g. `482.06` → `1,097.36` and `615.30`). This is string work on data already in memory; no extra
   I/O. The citation with the most terms is the "main" one. Mark it (e.g. `primary: true`) so the UI can open
   it first.
2. **A small render function** (new module, e.g. `app/answering/highlight.py` or under `app/ingestion/`,
   wherever fits best): `render_highlight(pdf_path, pdf_page, terms, *, zoom=1.6, margin=150) -> (png_bytes, n_matches)`.
   - For each term: `page.search_for(term)`. If a term matches several places, prefer the matches whose row
     shares words with the cited chunk text (pass the chunk text in, or the first ~200 chars), else keep all.
   - Widen each hit to its printed row: words from `page.get_text("words")` whose vertical centre is within
     the hit's height; union their rects.
   - Draw: `add_highlight_annot(row_rect)` for the row and a thin red `add_rect_annot(hit)` for the figure
     (draw on the in-memory page only; never save the PDF).
   - Clip to the union of rows ± margin (full width of the page), render with `get_pixmap(matrix=Matrix(zoom, zoom), clip=...)`, return PNG bytes.
   - No match → render the whole page (or its top part), `n_matches = 0`.
3. **One endpoint** in `app/api/documents.py`: `GET /documents/{doc_id}/pages/{pdf_page}/highlight?term=…&term=…`
   → `image/png`, with the match count in a header (e.g. `X-Highlight-Matches`). The PDF path comes **only** from
   the stored document (upload dir + its id), never from the request. 404 for an unknown document, 422 for a
   page out of range. Cap terms (e.g. ≤ 12, each ≤ 32 chars, digits/`,.%()-` and letters only). A sync `def`
   endpoint is fine (FastAPI runs it in a thread). A tiny in-process `lru_cache` on (doc, page, terms) is enough.
4. **UI** (`ui/components.py`, `ui/api_client.py`): a **Show in PDF** button inside each citation expander;
   on click, fetch the PNG (cache it with `st.cache_data`) and `st.image` it with a one-line caption ("Rows
   holding the answer's figures are highlighted" / "Figure not found on this page; showing the page" / "Scanned
   page: highlighting not available"). Keep the current snippet. Don't redesign the expander.

## Out of scope

OCR word boxes for scanned pages (Tesseract `image_to_data`). Leave a TODO in `docs/progress.md`. A full PDF
viewer or pdf.js component. Highlighting for general-knowledge answers (no citations).

## Tests (just for this change)

Build a one-page PDF in the test with PyMuPDF (`page.insert_text`) holding two rows, e.g.
"Finance Cost 94.90 171.40" and "Other Income 500.49 359.49".
- `render_highlight` finds `94.90` (1 match) and returns PNG bytes (starts with the PNG signature).
- An unknown term → 0 matches, still returns an image.
- The endpoint: 200 + `image/png` for a real doc/page (use the existing test app/fakes for documents), 404 for an
  unknown doc, 422 for a bad page.
- `highlight_terms`: a figure present in the chunk is included; a computed figure contributes its operands.

Run the new/changed test files and `ruff check` on changed files only.

## Try it

Start the API and UI (`start_api.bat`, `start_ui.bat`), ask "What was finance cost in FY2026?" (or any table
question), open the citation, click **Show in PDF**. Note the time it takes in `docs/progress.md`.

## Done when

- Clicking **Show in PDF** shows the cropped, highlighted page for text PDFs; a clear caption otherwise.
- `/query` latency unchanged (no rendering on the answer path).
- Tests above pass; `docs/progress.md` entry ("Improvements 02: show in PDF"); one commit, no push.
