# Improvement prompts (after the 2026-10-08 live test)

Four prompts that improve answer quality and add "show me where in the PDF". They come from a live test of
10 analyst-style questions on the EIG FY24 / FY25 / FY26 reports, and from ideas borrowed (in small, local
form) from the sifra-v2 codebase. Run them on the branch `improve/answer-quality-and-highlight`.

| # | File | What it does | Effort | Depends on |
|---|---|---|---|---|
| 01 | `01-trap-eval-and-number-check.md` | 10 "trap" eval questions with verified gold answers; fix number-check false alarms; recognise computed figures | Sonnet 5.5 · high | — |
| 02 | `02-pdf-highlight.md` | "Show in PDF" on a citation: the cited page, cropped, with the answer's row highlighted | Sonnet 5.5 · high | 01 (soft) |
| 03 | `03-retrieval-synonyms-and-topic-boost.md` | Line-item synonyms in the query enhancer; a small, measured topic boost in retrieval | Sonnet 5.5 · high | 01 |
| 04 | `04-answer-prompt-v2.md` | Answer prompt v2: several values for one metric, attribution check behind a flag | Sonnet 5.5 · high | 01 |

02, 03 and 04 are independent of each other. 01 goes first because it adds the questions the others are
measured on.

## How to run one

Start a fresh Claude Code session in the repo root, on the branch above, and type:

> Implement `docs/build-prompts/improvements/01-trap-eval-and-number-check.md`. Read that file and this
> folder's README first, follow it, and stop when its "Done when" list is met.

Change `01-…` to the next file for each new session. One prompt per session works best.

## Ground rules (light on purpose)

- **Scope is this project only.** Plain Python, the existing stack (FastAPI, Streamlit, PyMuPDF, Chroma, SQLite,
  Groq). No new services, no paid APIs, no big new dependencies.
- **Free and fast.** Nothing here may add an LLM call to the answer path, and nothing may add noticeable
  latency to `/query`. Prompt 04's optional check makes answers slightly longer, which is why it sits behind
  a config flag that is off by default.
- **Tests: a few, for the change only.** Add unit tests for the new logic (happy path, one or two edge cases).
  Mock LLM calls. Run the test files you touched plus `ruff check` on changed files. **Do not** run or fix the
  whole 699-test suite per prompt; we run everything once after all four prompts.
- **If something in a prompt turns out wrong or not worth it, don't force it.** Do the sensible thing, and
  write one line about it in `docs/progress.md`.
- **Finish each prompt** with a short entry in `docs/progress.md` (what changed, how to try it, what's open) and
  one commit on the branch (no push).

## What the live test showed (context for all four prompts)

| # | Question (trap) | Result |
|---|---|---|
| T001 | EBITDA margin FY26 vs FY25 | correct |
| T002 | PBT +25% on revenue +9%: operations vs other income vs finance cost | correct, but a false ⚠ on computed figures |
| T003 | FY24 basic EPS and face value (5-for-1 split) | correct |
| T004 | Total equity FY26 vs FY25, and the driver | **wrong driver**: said the IPO "raised ₹8,525.25 Mn of equity"; that is the whole issue incl. ₹4,525.25 Mn offer for sale that went to selling shareholders |
| T005 | What the company itself got from the IPO | correct (₹3,731.36 Mn net) |
| T006 | FY25 ROE (two definitions in the reports) | both numbers, no explanation of why they differ |
| T007 | EBITDA growth FY24 to FY25 | correct, false ⚠ on the computed difference |
| T008 | Project Engineering segment revenue change and share | correct, false ⚠ on "FY 25"/"FY 26" and computed figures |
| T009 | FY26 dividend per share (report says none) | abstained (LLM said INSUFFICIENT: the dividend passage was not in the 8 retrieved) |
| T010 | FY26 finance cost and why it fell | abstained (the Financial Highlights row was not retrieved) |

Notes: `retrieval.theta` is 0.25 and `top_k` is 8 (see `config.yaml`); T009/T010 were **not** score-gate
refusals: the model saw 8 passages without the answer and correctly said INSUFFICIENT. The fix there is
retrieval, not the threshold.

## Deliberately left out

- **Unit legend on table chunks** (sifra's `unit_scale.detect_scale`). Table chunks already carry a title that
  usually includes "(All amount are in ₹ million)", and no unit error appeared in the test. Revisit only if a
  crore-vs-million mix-up shows up (e.g. when comparing BPCL, which reports in crore, with EIG in million).
  It would also need a re-index.
- **A reranker model.** Too heavy for the target laptop and not needed for these fixes.
