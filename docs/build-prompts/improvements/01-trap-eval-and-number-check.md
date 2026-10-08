# 01 · Trap eval questions + number-check fixes

**Model:** Sonnet 5.5, effort high · **Branch:** `improve/answer-quality-and-highlight`

## Read first

`docs/build-prompts/improvements/README.md` (context and ground rules), `eval/README.md` (row format),
`app/answering/numbers.py` and `tests/test_numbers.py`, and the last few entries of `docs/progress.md`.

## Part A: a small "trap" eval set

Create `eval/questions_traps.jsonl` with the 10 rows below (ids `T001`–`T010`, one JSON object per line, same
schema as `eval/questions.jsonl`). Keep them in their **own file** so the existing 117-question set and its run
files are untouched. Add two doc keys to `eval/docs.yaml`: `report_fy24` → `EIG AR FY24.pdf` and
`report_fy26` → `EIG AR FY26.pdf` (`report_a` is already FY25).

The gold values were checked against the PDF text. `pdf_page` values are below. Printed page labels: FY24 and
FY25 labels equal the PDF page; FY26 labels are PDF page − 2 (e.g. PDF 34 = printed 32). Open the PDFs in
`data/uploads/` (the uploaded copies; filenames via `GET /documents` or SQLite) to confirm a label if unsure,
and drop a page you cannot confirm rather than guessing. Use `author: "trap-set 2026-10-08"`.

| id | question | slice / type | gold_answer (put the key figures in) | gold pdf_pages |
|---|---|---|---|---|
| T001 | What was the EBITDA margin in FY2026 compared with FY2025, and did profitability from operations expand or contract? | table / numeric | 34.0% in FY2026 vs 35.1% in FY2025 (EBITDA ₹1,162.17 Mn on revenue ₹3,415.82 Mn vs ₹1,097.36 Mn on ₹3,124.83 Mn): contracted. EBITDA here excludes other income. | FY26: 23, 34 |
| T002 | Profit before tax rose about 25% in FY2026 while revenue grew about 9%. How much of the PBT increase came from operations versus other income and lower finance cost? | table / explain | PBT +₹274.69 Mn (₹1,078.25 → ₹1,352.94 Mn). Other income +₹141.00 Mn (359.49 → 500.49), finance cost −₹76.50 Mn (171.40 → 94.90), EBITDA +₹64.81 Mn (1,097.36 → 1,162.17) less D&A +₹7.62 Mn (207.20 → 214.82). | FY26: 34 |
| T003 | What was basic earnings per share for FY2024, and on what face value is it stated? | table / numeric | ₹3.46 (basic and diluted), on face value ₹2 per share after the 5-for-1 split approved in April 2024 (EPS adjusted under Ind AS 33). | FY24: 61, 92 |
| T004 | What was total equity at March 31 2026 versus March 31 2025, and what drove the change? | table / explain | ₹9,771.41 Mn vs ₹4,933.59 Mn (+₹4,837.82 Mn), mainly IPO fresh-issue net proceeds ₹3,731.36 Mn (gross ₹4,000 Mn) plus FY26 profit ₹1,044.00 Mn. NOT the ₹8,525.25 Mn total issue size: ₹4,525.25 Mn was an offer for sale paid to selling shareholders. | FY26: 164, 228 |
| T005 | How much did the company itself receive from its IPO, as opposed to the total size of the issue? | text / numeric | Net ₹3,731.36 Mn (fresh issue ₹4,000 Mn less ₹268.64 Mn issue expenses). Total issue ₹8,525.25 Mn = fresh issue ₹4,000 Mn + offer for sale ₹4,525.25 Mn (to selling shareholders). | FY26: 228, 235 |
| T006 | What was return on equity for FY2025? | table / numeric | Two published values: 18.44% (Schedule III ratio, PAT ÷ average shareholders' equity; FY25 report and FY26 report comparatives) and 16.88% (Key Financial Ratios table, FY25 report). A good answer gives both and says where each comes from. | FY25: 83, 33; FY26: 231 |
| T007 | By how much did EBITDA grow from FY2024 to FY2025, and what happened to the EBITDA margin? | table / numeric | +₹482.06 Mn (₹615.30 → ₹1,097.36 Mn, about 78%); margin 22.83% → 35.12%. | FY25: 33, 35; FY24: 16 |
| T008 | How did revenue from the Project Engineering segment change in FY2026 compared with FY2025, and what share of total revenue is it? | table / numeric | ₹200.28 Mn → ₹75.40 Mn (−₹124.88 Mn, about −62%); about 2.2% of FY2026 revenue (₹3,415.82 Mn). | FY26: 229 |
| T009 | What dividend per share did the company pay for FY2026? | text / narrative | None: the Board did not recommend any dividend for the year ended March 31 2026 (to conserve resources after the IPO). Dashboard: "Dividend Declared: None". | FY26: 2, 36, 86 |
| T010 | What was finance cost in FY2026 and why did it decline compared with FY2025? | table / explain | ₹94.90 Mn vs ₹171.40 Mn. IPO proceeds were used to repay borrowings: ₹2,100 Mn under note 53, the same amount the chairman's letter calls "approximately ₹210 Cr" (a unit trap: crore vs million). | FY26: 34, 235, 14 |

All 10 are `route: DOCUMENT`, `answerable: true`. Put the trap in `notes` (e.g. "definition conflict",
"IPO total vs company proceeds", "restated base", "computed figures", "negative fact"). Validate with
`python -m eval.validate eval/questions_traps.jsonl`. If the schema rejects something (e.g. a type value), adapt
the row, not the schema.

Add one line to `eval/README.md` saying the trap set exists and how to run it:
`python -m eval.retrieval_eval --questions eval/questions_traps.jsonl` (free) and
`python -m eval.run --questions eval/questions_traps.jsonl --run traps-<name>` (uses Groq quota).
Don't run the paid one in this prompt.

## Part B: number-check false alarms (`app/answering/numbers.py`)

**Today:** `extract_numbers(..., for_answer=True)` skips a number with a letter right before it (`FY25`), but
`FY 25` / `FY 2026` / `Q 3` (with a space) slip through, so "Return on Equity for FY 25 was 18.44%" fails with
"number not found: 25". And any figure the model *computed* (a difference, a share, a growth rate) fails
because it is not printed in the source.

**Change 1, noise:** treat `FY 25`, `FY 2026`, `FY 2025-26`, `Q3 FY26`, `H1` with optional space as period
codes, not figures (add to `_NOISE` or the `for_answer` skips, whichever reads cleaner).

**Change 2, computed figures:** after the exact lookup, a figure that is still missing passes as **computed**
if it can be produced from two numbers in the cited text:
- difference `a − b` (either order, sign ignored), exact `Decimal` arithmetic; or
- ratio / share / growth: `a / b × 100` or `(a − b) / b × 100`, matching when rounded to the figure's own
  printed decimals (e.g. `2.2%`, `62.4%`, `78%`).

Compare like with like: use the mantissas as printed (the answer's "₹482.06 million" vs table cells "1,097.36"
and "615.30"). Keep it cheap: the pool is small (one answer's cited chunks), and a pairwise loop is fine; cap it
(e.g. skip the computed pass if the pool has more than ~400 numbers) so a huge table cannot slow `/query`.

Extend `NumberCheck` with something like `computed: tuple[str, ...]` (figure → how it was made, e.g.
`"482.06 = 1097.36 − 615.30"`). Status stays `pass` when every figure is either found or computed; the warning
fires only for figures that are neither. Keep the existing behaviour for everything else. Where the result is
serialised for the API/UI, include the computed list. In the UI, a small caption such as "computed from cited
figures" under the answer is enough; no redesign. Prompt 02 reuses the operands to highlight them.

**Example, before → after**

| Answer text | Today | After |
|---|---|---|
| "ROE for FY 25 was 18.44%" | fail: `25` | pass |
| "EBITDA increased by ₹482.06 million" (cited: 1,097.36 and 615.30) | fail | pass, computed `1097.36 − 615.30` |
| "a decline of 124.88 … about 2.2% of revenue" (cited: 200.28, 75.40, 3,415.82) | fail | pass, computed |
| "decreased by ₹482.60 million" (wrong) | fail | still fail |

## Tests (just for this change)

In `tests/test_numbers.py`: `FY 25` and `Q3 FY26` are not checked; a difference passes as computed; a rounded
percentage share passes as computed; a wrong number still fails; the pool-size cap skips the computed pass.
Also one check that `questions_traps.jsonl` validates (reuse whatever `tests/test_eval_tooling.py` already
does). Run `pytest tests/test_numbers.py tests/test_eval_tooling.py` and `ruff check` on changed files.

## Done when

- `eval/questions_traps.jsonl` (10 rows) validates; `eval/docs.yaml` has the two new keys; one line in `eval/README.md`.
- Number check ignores spaced period codes and labels computed figures; the tests above pass.
- `docs/progress.md` entry ("Improvements 01: trap set + number check"), one commit, no push.
