# Build Prompts: Step-by-Step Implementation

A series of copy-paste prompts for **Claude Code (Sonnet 5)** that build the Document Q&A system end to end.
Source of truth: `docs/design.md` + `docs/addendum.md` (the addendum wins where they differ).

## How to use

1. Open Claude Code in the project repo root.
2. Set the effort level shown for the prompt (`/model` → Sonnet 5, then pick medium/high).
3. Paste the prompt from the file (the block inside the fence).
4. Review the diff, run the tests yourself, commit, then move on.
5. One prompt per session is ideal. Start a fresh session (`/clear`) between steps; every prompt re-reads the docs and `docs/progress.md`, so no context is lost.

## Order

| # | File | Builds | Effort | Depends on |
|---|---|---|---|---|
| 00 | `00-skeleton.md` | Repo skeleton, config, Docker, SQLite, logging, CI | medium | — |
| 01 | `01-eval-set-tooling.md` | Eval-set schema + validator (the questions themselves are **human work**) | medium | 00 |
| 02 | `02-ingestion-parsing.md` | PDF parsing, OCR fallback, tables, chunking (pure functions) | high | 00 |
| 03 | `03-ingestion-pipeline.md` | Upload API, worker thread, embeddings, Chroma, status | high | 02 |
| 04 | `04-retrieval-and-ci-gate.md` | Retrieval, Recall@5/MRR eval, MLflow, CI gate | high | 03, 01 |
| 05 | `05-document-answer-path.md` | LLM client, grounded answers, citations, abstention, number check | high | 04 |
| 06 | `06-router-general-mixed.md` | Router, general path, MIXED, `/query` endpoint | high | 05 |
| 07 | `07-streamlit-ui.md` | Upload, status, chat, citation snippets, feedback buttons | medium | 06 |
| 08 | `08-observability-metrics.md` | Request log wiring, metrics page, drift signals | medium | 07 |
| 09 | `09-full-eval-runner.md` | Router benchmark, judge + calibration, abstention, latency | high | 06, 01 |
| 10 | `10-experiments.md` | Chunk size, BM25, top-k, θ sweep → final config | high | 09 |
| 11 | `11-langfuse-loadtest-fallback.md` | Langfuse, load test, Ollama fallback | medium | 08 |
| 12 | `12-readme-seed-demo.md` | README with numbers, seed script, demo prep | medium | all |

After step 12: [`improvements/`](improvements/README.md) holds four follow-up prompts (trap eval set, number-check
fixes, "Show in PDF" highlighting, retrieval synonyms/topic boost, answer prompt v2) from the 2026-10-08 live test.

Steps 00 → 03 are strictly in order. After that, 07/08 and 09 can run in parallel if two people are working.
Step 01's tooling is quick. **Start writing the eval questions on day 1**, in parallel with 02/03.

## Working agreements (light on purpose)

These are defaults, not laws. If something is better done differently, do it and write down why.

- **Tests come with each step.** Every prompt asks for unit tests for the logic it adds. Network/LLM calls are mocked in unit tests. `pytest` must be green before committing.
- **Don't break what's there.** If a step needs to change earlier code, that's fine. Keep the existing tests passing, or update them with a reason.
- **Config over constants.** Tunables (models, θ, chunk size, top-k, limits) live in `config.yaml`. LLM prompts live in `prompts/*.yaml` with a `version`.
- **No secrets in code or logs.** Read env vars by name (`GROQ_API_KEY`, etc.). `.env` is git-ignored.
- **Deviations are allowed.** If the implementation differs from the design (a different library, a different number, an extra file), note it in `docs/progress.md` under that step. That keeps teammates in sync without blocking anyone.
- **Libraries:** prefer what the design lists. Adding a small, well-known dependency is fine if it clearly saves effort. Avoid big frameworks (LangChain/LlamaIndex) and extra services (Redis/Celery/K8s) unless the team agrees.

## `docs/progress.md`

Each prompt ends by appending a short entry here: what was built, how to run it, deviations, and open TODOs.
It's the hand-off log between sessions and between teammates. Read it before starting the next step.
