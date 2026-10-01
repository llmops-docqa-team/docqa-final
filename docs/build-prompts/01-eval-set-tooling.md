# 01: Eval Set Tooling

**Effort:** medium · **Prereqs:** 00 · **Design refs:** §14, addendum §2 (three slices), §3, §5 "Not for Claude Code"

> **Human work first.** The questions, gold answers and gold pages must be written and checked by people (you plus a second checker).
> Claude Code only builds the *format + validator*, so mistakes in the JSONL get caught early.
> Target: ~85 questions = 40 answerable doc (incl. ~15 table questions) + 12 unanswerable near-misses + 18 general + 15 mixed.
> Tag each one with the slice: text / table / scanned.

````text
Read docs/design.md §14 and docs/addendum.md §2-§3, plus docs/progress.md.

Task: build tooling for the evaluation set. Do NOT write the actual questions or gold answers;
those are human work. A few clearly-fake example rows for tests are fine.

Build:
- eval/schema.py: a Pydantic model for one row of eval/questions.jsonl. Fields, roughly:
  id, question, route (DOCUMENT/GENERAL/MIXED), answerable (bool, doc questions), gold_answer,
  gold_pages (list of {doc, page, pdf_page?}), slice (text/table/scanned/general/mixed),
  type (numeric/narrative/explain/definition/...), and for MIXED: gold_document_question,
  gold_general_question, gold_general_answer. Extend the schema if something's obviously missing.
- eval/validate.py (runnable as `python -m eval.validate`): loads the JSONL and reports
  schema errors, duplicate IDs, counts per route/slice/answerable, and warnings
  (e.g. answerable DOCUMENT with no gold_pages, unknown doc keys).
- eval/docs.yaml: maps doc keys (e.g. "report_a") to filenames + a note. PDFs themselves are NOT committed.
- eval/questions.jsonl with 3-5 obviously placeholder rows marked "TODO", so the pipeline can be tested.
- eval/README.md: short guide for teammates on how to write a good question (near-miss unanswerables,
  definition-trap generals, table questions), how to record gold pages (printed label AND PDF page index),
  and the two-person verification rule.
- Unit tests for the schema and validator.

When done: run tests, append a "Step 01" entry to docs/progress.md, and stop.
````
