# Eval set

`questions.jsonl` is the labelled set behind every number in the README (router accuracy, Recall@5,
correctness, abstention). **Humans write and verify it.** The tooling only checks the format.

```
python -m eval.validate            # report errors and warnings
python -m eval.validate --strict   # warnings also fail (use before freezing the set)
```

`questions_traps.jsonl` is a separate 10-question "trap" set (T001-T010; definition conflicts, IPO total vs company proceeds, computed figures) on the FY24/FY25/FY26 reports. Retrieval only (free): `python -m eval.retrieval_eval --questions eval/questions_traps.jsonl`; full run (uses Groq quota): `python -m eval.run --questions eval/questions_traps.jsonl --run traps-<name>`.

Rows whose id starts with `TODO-` are placeholders; delete them once real rows exist.

## Targets (~85 rows)

| Group | # | route | slice |
|---|---|---|---|
| Answerable doc questions | 40 (~15 tables) | DOCUMENT, `answerable: true` | `text` / `table` / `scanned` |
| Unanswerable near-misses | 12 | DOCUMENT, `answerable: false` | `text` / `table` / `scanned` |
| General | 18 | GENERAL | `general` |
| Mixed | 15 | MIXED | `mixed` |

Every doc row gets a slice, so results are reported separately for text, table and scanned.

## One row

```json
{"id":"D07","question":"What was the consolidated revenue from operations in FY2025?",
 "route":"DOCUMENT","slice":"table","type":"numeric","answerable":true,
 "gold_answer":"Rs 1,234 crore (consolidated)","gold_pages":[{"doc":"report_a","page":47,"pdf_page":53}],
 "author":"vinay","verified_by":"purab"}
```

(One line per row in the real file.) Fields: `id` (unique), `question` (max 500 chars), `route`, `slice`,
`type` (`numeric|narrative|explain|definition|near_miss|other`), `answerable` (required for DOCUMENT),
`gold_answer`, `gold_pages`, optional `author`, `verified_by`, `notes`. MIXED rows also need
`gold_document_question`, `gold_general_question` and `gold_general_answer`; their `gold_answer` / `gold_pages`
(optional) describe the document part. Unknown fields are
rejected, so typos get caught.

## Writing good questions

- **Near-miss unanswerables.** Ask for something the retrieved pages *look* relevant to but do not
  contain: a wrong year (FY2030), a metric the report does not disclose, a different company.
  Off-topic questions are too easy, since the score gate catches them. Near-misses test the LLM gate.
  Use `type: near_miss`, `answerable: false`, and no `gold_pages`.
- **Definition-trap generals.** Terms that appear in the report but are asked generically
  ("What is working capital?", "What is EBITDA?"). Route is GENERAL; a naive router sends them to DOCUMENT.
- **Table questions.** A number from a financial statement. Say standalone vs consolidated and which
  year. Put the exact figure and unit in `gold_answer`. Scanned-table questions go in the `scanned` slice.
- **Mixed.** One sentence with a document part and an unrelated general part. Fill all three gold fields.
- Self-contained, single-turn, one fact per question. No "and its profit?" follow-ups.
- If one report repeats another's figures (e.g. FY26 contains FY25 comparatives), the gold answer must
  say which report and which year.

## Recording gold pages

Record **both**: `page` is the printed label on the page (what a citation shows) and `pdf_page` is the
1-based index in the PDF file. They differ because of cover pages and roman-numeral front matter. List
every page that legitimately contains the answer. Doc keys come from `docs.yaml`. PDFs are never
committed; check the licence first.

## Verification (recommended, not a gate)

Ideally a **second person** opens the PDF, confirms the answer and the gold pages independently, then sets
`verified_by`. It makes the numbers more credible, but nothing waits for it: the validator only prints an
`info:` count of rows without `verified_by` (and `--strict` ignores it). Run evals and experiments on the
set as it is, and say in the README which rows were second-checked.

## Retrieval eval and the CI gate (step 04)

`eval/retrieval_eval.py` measures retrieval only (no LLM calls, so it is free and deterministic):
Recall@1/3/5/8 and MRR per slice, scoring answerable DOCUMENT questions against **READY** documents.
A hit is a retrieved chunk in the right document on a gold page. **Strict** = same page, **±1** =
neighbouring pages also count. Pages are compared by `pdf_page` (the printed `page` label is used only
when a gold page has no `pdf_page`, which is why you should record both).

```
python -m eval.retrieval_eval                 # your corpus: READY docs in data/ whose filenames match docs.yaml
python -m eval.retrieval_eval --mlflow        # also log params/metrics/git hash/eval-set hash to ./mlruns
python -m eval.retrieval_eval --fixture --check   # what CI runs
mlflow ui --backend-store-uri ./mlruns        # needs: pip install -r requirements-eval.txt
```

Upload the reports through the app first (the eval reads the same SQLite/Chroma the app uses; stop the API
while it runs). Questions whose documents are missing or not READY are skipped and listed in
`eval/results/retrieval_latest.json`, together with every miss and the top retrieved pages.

**CI gate.** The real reports are private, so CI runs the same code on a **synthetic corpus**: two
near-twin fictional annual reports ("Halden Power Cables", FY25/FY26) generated by `eval/ci_fixture.py`
(deterministic, not committed), with 49 answerable + 4 unanswerable questions in
`eval/fixtures/ci_questions.jsonl`. It ingests them through the real worker with the real embedder, and fails
if Recall@5 drops more than 3 points, or MRR more than 0.05, below `eval/baselines/ci_retrieval.json`.
Any change to chunking, the embedding model/prefix, parsing or retrieval is therefore gated.
Intentional change? Re-run `python -m eval.retrieval_eval --fixture --write-baseline` and commit the file with
a note on why. Changing the fixture questions needs the same (a unit test checks the baseline matches).

## Full eval (step 09)

`python -m eval.run` sends every question through the real `/query` code path (router, document and general
paths) and reports the router benchmark, retrieval, answers by slice, abstention, number-check failures,
latency and tokens. Outputs: a console summary, `eval/results/latest.json`, an MLflow run (params, metrics, git
hash, eval-set hash, prompt versions), and with `--markdown` README-ready tables.

```
python -m eval.run --config config.yaml                 # everything not done yet, then the report
python -m eval.run --slice table --limit 20             # the next 20 unfinished table questions
python -m eval.run --slice general --slice mixed
python -m eval.run --router-only                        # router + baselines only (cheap: small model's quota)
python -m eval.run --no-llm-judge                       # no Gemini calls: number match, routing, retrieval
python -m eval.run --report-only --markdown             # rebuild the report from the run file, no services
python -m eval.run --no-cache                           # live LLM calls: needed for latency numbers
```

- **Resumable.** Each question is appended to `eval/results/runs/<run>.jsonl` (`--run NAME`, default
  `default`). Run the same command again and finished questions are skipped; the report always covers
  everything in the run file. `--limit N` means "do N *unfinished* questions", so repeating it walks through
  the set. A question whose LLM was unavailable (Groq 429/503) is marked and **retried** on the next run; a
  judge outage leaves its verdicts pending and a later run fills them without re-running the pipeline.
- **Never mixes setups.** The run file records models, prompt versions, theta, top-k, chunking, embedding model
  and the indexed documents. Resuming with a different setup is refused (`--fresh` or another `--run`).
  An edited question is re-run; a different judge model or prompt re-judges.
- **Free-tier friendly.** The dev LLM cache is on (a question is paid for once), calls are paced to
  `eval.tpm_budget` tokens/min per model, and the judge is spaced to `eval.judge_min_interval_seconds`.
  Latency is only reported for questions with no cached call, so use `--no-cache` for a latency run.
- **How things are scored.** Numeric questions: normalised number match (exact; years and `FY25` codes ignored;
  a MIXED row's document half too when its gold answer is a figure). Everything else: the Gemini judge
  (`prompts/judge_v1.yaml`: 0/1 correct vs the gold answer, and 0/1 grounded in the cited text). A numeric row
  is also judged, which gives a free agreement check between the two methods. Citation accuracy: a cited
  page is a gold page (strict) or within one page. Abstention: positive class = ABSTAIN; a question the router
  sent away from the document path counts as `other`, not as a false answer. Rows that need the judge and
  have no verdict yet are counted as `unjudged` and left out of the rate, never counted wrong.
- **Router baselines.** (a) keyword rules (`eval/router_baselines.py`, written without looking at the set) and
  (b) a retrieval-score threshold. The threshold is tuned on the eval set itself (the best cut-off), so it is
  the baseline's best case; it cannot answer MIXED. The report says so when a baseline reaches 90% on
  DOCUMENT/GENERAL questions.
- **Judge calibration.** `python -m eval.calibrate export --n 20` writes a CSV for two people to label (half
  the answers the judge called wrong, and no judge verdict shown) plus a hidden key; fill `human_a_*` /
  `human_b_*` with 1/0, then `python -m eval.calibrate score` reports % agreement and Cohen's kappa for
  judge vs each person, judge vs both, and person vs person. Below 80%: fix the judge prompt first.
- **Online judge.** `python scripts/judge_recent.py --last 50` scores recent answered requests and stores
  them in `requests.judge_correct/judge_grounded` for the Metrics page. The request log holds no text, so this
  needs `observability.log_content: true` (config.yaml) when the requests are asked; without a gold answer,
  "correct" means "answers the question and agrees with its cited sources".

The eval opens the app's own `data/` index like `eval.retrieval_eval`: stop the API while it runs.
