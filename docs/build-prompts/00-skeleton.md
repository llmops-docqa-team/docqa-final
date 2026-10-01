# 00: Repo Skeleton

**Effort:** medium · **Prereqs:** none · **Design refs:** §16, §23 step 0, addendum §5 Prompt A

````text
Read docs/design.md and docs/addendum.md (the addendum overrides the design where they differ),
and docs/build-prompts/README.md for working agreements.

Task: build Step 0, the project skeleton. No product features yet.

Build:
- A Python 3.11 project with a sensible package layout, e.g.:
    app/        (FastAPI backend: api/, ingestion/, retrieval/, answering/, routing/, llm/, storage/, observability/)
    ui/         (Streamlit app)
    prompts/    (versioned LLM prompt YAMLs: router_v1.yaml, answer_doc_v1.yaml, answer_general_v1.yaml, judge_v1.yaml as stubs with a `version` field)
    eval/       (eval set + runners later)
    scripts/    (seed / utility scripts later)
    tests/
  Adjust the layout if you have a better idea; keep it easy to navigate.
- Dependency management (pyproject.toml or requirements files; your call). Pin major versions.
- config.yaml holding the tunables from the design (models, base URLs, chunk size/overlap, top_k, theta,
  upload limits, ingestion batch size, OCR thresholds). Plus a small typed loader (Pydantic settings or similar)
  that also reads env vars like GROQ_API_KEY, GEMINI_API_KEY, LLM_BASE_URL by name. Add .env.example (no real values).
- SQLite init: `documents` table (id, filename, sha256, status, pages_total, pages_done, chunks, ingest_seconds,
  error, created_at, updated_at) and `requests` table with the fields listed in design §15 (feedback and
  per-stage timings included). A simple migration or "create if not exists" approach is fine.
- Structured logging (structlog or std logging with JSON) plus a request-ID middleware in FastAPI.
- FastAPI app with GET /health. A Streamlit app that calls /health and shows the result.
- Dockerfile (Python 3.11-slim, installs tesseract-ocr) and docker-compose running the API + Streamlit
  with a mounted data/ volume.
- pytest with a smoke test (health endpoint, config loads, DB tables created).
- GitHub Actions workflow that installs deps and runs pytest. Add ruff if it's quick.
- .gitignore (data/, .env, mlruns/, caches, PDFs).

Out of scope: ingestion, retrieval, LLM calls.

When done:
- Run the tests and show the result. Show how to run locally (without Docker) and with docker compose.
- Create docs/progress.md if missing and append a "Step 00" entry: what exists, how to run, any deviations from the design.
- Stop. Don't start the next step.
````
