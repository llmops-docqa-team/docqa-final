# 10: Experiments → Final Config

**Effort:** high · **Prereqs:** 09 · **Design refs:** §10 (BM25/reranker rules), §11 (θ calibration), §14 (planned experiments), §19 (trade-offs)

````text
Read docs/design.md (§10, §11, §14 "Planned experiments", §19) and docs/progress.md.

Task: run the planned experiments, each as an MLflow run, and pick the final config with evidence.

Experiments:
1. Chunk size 250 / 400 / 600 tokens (keep overlap proportional) -> Recall@5 (needs a re-index per setting:
   use a separate Chroma collection per config so nothing gets clobbered).
2. Dense vs dense + BM25 with RRF (rank_bm25, small and self-contained) -> Recall@5.
   Adopt hybrid only if Recall@5 improves by >= 5 points (design rule); otherwise keep the code behind a config flag, off.
3. top-k 3 / 5 / 8 passed to the LLM -> answer correctness vs tokens.
4. θ sweep -> abstention trade-off curve (false-answer rate vs wrong-abstention rate). Choose θ so that
   the false-answer rate on unanswerables <= 10%. Save the curve as a PNG.
5. Router variants: already done in step 09; just collect the numbers here.
6. Optional: cross-encoder reranker on/off -> correctness and latency. Same adoption rule.

Retrieval-only experiments (1, 2) cost no LLM quota; do those first. For 3, 4 and 6 use the dev cache
and keep the runs small enough for the free tier.

Outputs:
- scripts/run_experiments.py (or a few small scripts): reproducible, config-driven.
- Update config.yaml with the chosen values and update the CI retrieval baseline if retrieval changed.
- docs/experiments.md: a table per experiment, the chosen value, and one "We chose X over Y because Z (numbers)"
  line each, ready for the trade-offs slide.

Tests: RRF fusion function; the θ-sweep curve computation on synthetic data.

When done: run tests, append a "Step 10" entry to docs/progress.md, and stop.
````
