# 09: Full Eval Runner (Router Benchmark, Judge, Abstention, Latency)

**Effort:** high · **Prereqs:** 06, and 01 with real questions written · **Design refs:** §9 (benchmark), §12 (judge), §14, addendum §2 (three slices)

> Groq's free tier has daily token caps (design §5, §12). Use the dev cache from step 05, and make the runner resumable / runnable per slice.

````text
Read docs/design.md (§9, §12, §14) and docs/addendum.md, plus docs/progress.md.

Task: one command that evaluates the whole system on eval/questions.jsonl and logs the results to MLflow.

Build `python -m eval.run --config config.yaml [--slice ...] [--limit N] [--no-llm-judge]`:
- Runs every question through the same code path as /query (call the functions directly, not over HTTP,
  unless that's simpler), with the dev cache on. Only READY docs are scored.
- Router: accuracy + 3x3 confusion matrix, DOCUMENT recall. Benchmark 3 variants:
  (a) keyword rules baseline, (b) retrieval-score threshold baseline, (c) the LLM router.
  Write the two baselines as small, honest implementations in eval/ (or app/routing/baselines.py).
- Retrieval: reuse step 04's metrics.
- Answers (answerable DOCUMENT questions): numeric questions use a normalised number match (reuse the step-05
  normaliser); the others use an LLM judge, 0/1 vs the gold answer. Also groundedness ("every claim supported
  by the cited sources?") via the judge, and citation accuracy (cited page in gold pages, strict and ±1).
- Abstention: precision/recall of ABSTAIN, false-answer rate on unanswerables, wrong-abstention rate on answerables.
- Number-check failure rate, latency p50/p95/p99 per stage, tokens and cost-equivalent.
- Every answer metric broken down by slice: text / table / scanned.
- Judge: prompts/judge_v1.yaml, Gemini Flash via its API (a different model family from the generator).
  Keep the judge behind a small interface so it can be swapped.
- Judge calibration helper: export 20 sampled answers to a CSV for two humans to label, then a command that
  computes the judge-human agreement (% agree and Cohen's kappa).
- Online judge script (scripts/judge_recent.py): scores the last N logged answers from the requests table
  and stores the scores so the metrics page Quality section can show them.
- Outputs: a console summary table, eval/results/latest.json, an MLflow run (params, metrics, git hash,
  eval-set hash, prompt versions), and optionally a Markdown table ready to paste into the README.
- Make the runner resumable (skip questions already done in this run's output) so you can split it across days.

Tests: baseline routers on hand-made cases; metric functions; the resume logic; judge-response parsing (mocked).

When done: run it on whatever real questions exist (even partially), put the headline numbers in
docs/progress.md under "Step 09", and stop.
````
