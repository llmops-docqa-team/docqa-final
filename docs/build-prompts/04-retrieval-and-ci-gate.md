# 04: Retrieval, Retrieval Eval, MLflow, CI Gate

**Effort:** high · **Prereqs:** 03, 01 · **Design refs:** §10, §13 (MLflow, CI gate), §14, addendum §2, §4

````text
Read docs/design.md (§10, §13, §14) and docs/addendum.md, plus docs/progress.md.

Task: dense retrieval + a retrieval-only evaluation that is free (no LLM calls) and runs in CI.

Build:
- app/retrieval/: retrieve(question, top_k) -> ranked chunks with scores and metadata.
  - Embed the query with the bge query-instruction prefix.
  - Search only documents whose status is PARTIAL or READY (filter via metadata or a doc_id list).
  - Fetch top-8 by default (config); the answer step will pass the top-5 to the LLM.
  - Return enough info for later steps: chunk id, doc filename, page, page_label, source_kind, text, score.
- A debug endpoint or script (e.g. POST /debug/retrieve or scripts/retrieve.py) to inspect results by hand.
- eval/retrieval_eval.py (python -m eval.retrieval_eval): for answerable DOCUMENT questions in
  eval/questions.jsonl, compute Recall@5 (strict page match AND ±1 page), MRR, broken down by slice
  (text/table/scanned). Use only fully READY documents. Print a table and write eval/results/retrieval_latest.json.
- MLflow logging (local file store) for eval runs: params (chunk size, overlap, top_k, embedding model),
  metrics, git hash, eval-set hash. Make MLflow optional with a flag so CI doesn't need it.
- CI gate. The real reports are private and not in the repo, so CI can't use them. Pick a practical option
  and document it, e.g.:
    a small committed fixture corpus (synthetic or licence-OK PDF, a few pages) + a handful of fixture questions,
    with a committed baseline (eval/baselines/ci_retrieval.json). CI ingests it with the real embedder,
    and fails if Recall@5 drops more than ~3 points below baseline.
  Cache the fastembed model in CI if possible. If you find a better approach, use it and explain why.

Tests: retrieval filters out non-ready docs; ranking order; metric functions (Recall@k, MRR, ±1 page) on hand-made data.

When done: run tests and the retrieval eval on whatever data is available, put the baseline numbers in
docs/progress.md under "Step 04", and stop.
````
