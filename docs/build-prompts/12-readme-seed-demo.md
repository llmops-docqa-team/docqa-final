# 12: README, Seed Script, Demo Prep

**Effort:** medium · **Prereqs:** everything else · **Design refs:** §18 (demo + fallbacks), §19, §20, deliverables checklist

The slides, recording and rehearsal are team work. This prompt covers the code and docs side.

````text
Read docs/design.md (§18, §19, §20), docs/addendum.md, docs/progress.md and docs/experiments.md.

Task: make the project easy to run, reproduce and demo.

Build:
- scripts/seed.py: ingest the eval reports from a local folder (path from env/config; the PDFs aren't in git)
  so the demo starts with pre-indexed documents. Make it idempotent (same hash -> skip). Optionally run it on
  container start via a compose flag.
- A demo checklist script or doc (docs/demo.md): the exact demo questions in order (general while uploading,
  document with citation, MIXED, near-miss abstention, metrics page), their expected outputs (they should be
  in the eval set), and the fallback steps (switch to Ollama, the screen recording).
- README.md, written for a reviewer who has 5 minutes:
  - Problem + one-sentence business objective, and scope in/out.
  - The architecture diagram (from design §7, updated to match what was actually built, incl. OCR/tables/PARTIAL).
  - Setup: local + docker compose, env vars, seeding, running tests, running the eval.
  - Results table with REAL numbers from eval/results and docs/experiments.md (router accuracy, Recall@5
    strict/±1, answer correctness by slice, citation accuracy, abstention precision/recall, latency p50/p95/p99,
    ingestion s/page for text vs scanned, load-test throughput, judge-human agreement).
  - Key trade-offs ("We chose X over Y because Z") and known limitations, honestly (e.g. scanned tables).
  - A short "what breaks first at 10×" section.
- Clean-up pass: remove dead debug code (or keep it behind a flag), make sure `docker compose up` + seed works
  from a fresh clone, that all tests pass, and that CI is green.

When done: append a final "Step 12" entry to docs/progress.md listing anything left undone, and stop.
````
