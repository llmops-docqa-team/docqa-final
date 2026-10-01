# 08: Request Log & Metrics Page

**Effort:** medium · **Prereqs:** 07 · **Design refs:** §13, §15, addendum §4 (router fallback metric)

````text
Read docs/design.md (§13, §15) and docs/addendum.md, plus docs/progress.md.

Task: make every query measurable, and show it on a metrics page.

Build:
- Write one row per /query to the SQLite `requests` table with the fields in design §15: trace_id, ts,
  question_len, route, router_ok, prompt_versions, model_ids, per-stage timings (router/embed/retrieve/llm/total),
  top_score, n_sources, status (answered/abstained/not_ready/error), abstain_reason, citations_valid,
  number_check, tokens_in/out, cost_usd_equiv (list-price-equivalent, rates in config), error, feedback.
  Add/adjust columns as needed (the step-00 schema was a first guess). Also store the git SHA / app version.
  Logging must never fail the request: catch and log errors.
- Don't store full document text in the log; chunk IDs and short snippets at most.
- Streamlit Metrics page reading the SQLite file directly, covering design §15's five categories:
  - Operational: p50/p95/p99 total and per stage, error/timeout rate, tokens + cost per request, ingestion s/page.
  - Input: question-length distribution, route mix, upload failures by reason.
  - Output: abstention rate, citation-invalid rate, number-check failure rate, router fallback rate, answer length.
  - Quality: 👍/👎 rate; judge scores once step 09 writes them (leave a slot).
  - Drift: daily median top retrieval score; route mix week over week.
  Colour indicators from config thresholds (e.g. p95 > 6 s red, abstention > 40% amber). No alerting system.
  A time-range filter is nice to have.
- Put the metric computations in plain functions (e.g. app/observability/metrics.py) so they're testable,
  with the Streamlit page as a thin view.

Tests: percentile and rate functions on synthetic rows; logging failure doesn't break /query; feedback update.

When done: run a dozen queries, check the page, append a "Step 08" entry to docs/progress.md, and stop.
````
