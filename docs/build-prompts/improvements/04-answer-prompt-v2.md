# 04 · Answer prompt v2: several values, honest labels, optional attribution check

**Model:** Sonnet 5.5, effort high · **Branch:** `improve/answer-quality-and-highlight` · **Needs:** prompt 01 (trap set)

## Read first

`docs/build-prompts/improvements/README.md`, `prompts/answer_doc_v1.yaml`, `app/llm/prompts.py` (how a version
is loaded), the `prompts:` and `answer:` blocks of `config.yaml`, `app/answering/document.py` (how the JSON reply
is parsed and checked), `app/llm/json_reply.py`, the last entries of `docs/progress.md`.

## The problem, measured

- **T004 (misleading):** "What was total equity … and what drove the change?" The answer said the IPO "raised
  ₹8,525.25 million of equity share capital and securities premium". The source says the IPO *aggregated*
  ₹8,525.25 Mn, of which ₹4,525.25 Mn was an offer for sale paid to selling shareholders. The company received
  ₹4,000 Mn gross, ₹3,731.36 Mn net. The number check passed, because the digits are in the source. The error is
  in the *label* the model put on the figure.
- **T006 (incomplete):** "What was ROE for FY2025?" The answer gave 18.44% and 16.88% without saying why there
  are two (different tables/definitions, and the FY26 report restates FY25 on the average-equity basis).

## Part A: `prompts/answer_doc_v2.yaml` (copy v1, add rules; keep the JSON contract identical)

Add, in the same plain style as the existing rules:

1. **Keep each figure's own label.** Describe a figure with what the source says it is. Don't turn a total into
   the entity's own amount, gross into net, an issue size into money received, a guidance figure into an actual,
   or another entity's figure into this one's. If the source splits a total, give the split.
2. **Several values for one metric.** If the sources give different values for the same metric and period, give
   each one with where it comes from (which report/table) and its basis or formula if the source states it.
   Never pick one silently and never average them.
3. **Which report.** When a prior-year figure appears in more than one report (an original and a later
   comparative), say which report the figure is from.

Keep it short; these rules must not make normal answers longer. Point `config.yaml` → `prompts.answer_doc` at
`answer_doc_v2`. The eval runner records prompt versions and refuses to mix them, so runs stay comparable.

**Before → after (expected)**

| | Answer |
|---|---|
| T004 today | "…driven primarily by the IPO, which raised ₹8,525.25 million of equity share capital and securities premium." |
| T004 v2 | "…driven by the IPO's fresh issue (₹4,000 Mn gross, ₹3,731.36 Mn net of expenses; the rest of the ₹8,525.25 Mn issue was an offer for sale by existing shareholders) and FY2026 profit of ₹1,044.00 Mn." |
| T006 today | "ROE for FY 25 was 18.44% (Schedule III ratio) and 16.88% (key financial ratio)." |
| T006 v2 | "The FY2025 report gives two ROE figures: 18.44% in the Schedule III ratios (PAT ÷ average shareholders' equity) and 16.88% in the Key Financial Ratios table." |

## Part B: optional attribution check, behind `answer.evidence_quotes: false` (off by default)

When the flag is on, the prompt variant asks for one extra JSON field:
`"evidence": [{"figure": "₹3,731.36 million", "quote": "<the exact sentence or table row from the source>"}]`.
Code then checks, for each item: the quote occurs in a cited chunk (compare with whitespace and ₹/backtick
normalised), and the figure's digits occur in the quote. A failed item adds a warning like the number check's
("⚠ figure not supported by a quoted source line"). Missing or malformed `evidence` → ignore it, never fail the
answer. Prompt 02 can later highlight the quote instead of single figures (leave a TODO, don't build it here).

Keep it **off** by default: it makes every answer longer (tokens and a little latency on Groq's 8K tokens/min
free tier). The simplest way to switch prompts is probably a second file (`answer_doc_v2_evidence.yaml`)
selected when the flag is on. Use your judgement.

## Measure (spends Groq quota; keep it small)

If quota allows: `python -m eval.run --questions eval/questions_traps.jsonl --run traps-v1` with v1 selected
first, then `--run traps-v2` with v2 (stop the API/Docker first; it shares `data/`). That is about 20 questions,
about 100K tokens, under the ~200K/day free limit. Record T004 and T006 before/after and any changes in other T
rows in `docs/progress.md`. If quota is short, do T004 and T006 only via `/query` and note it.

## Tests (just for this change)

`answer_doc_v2` loads and has the same placeholders as v1; the config points at it. Evidence check: a quote
present in the cited text with the figure passes; a quote not in the text warns; a figure missing from its quote
warns; a missing/malformed `evidence` field is ignored. The flag off means no evidence field is requested or
checked. Mock the LLM (see `tests/fakes.py`). Run touched test files and `ruff check` on changed files.

## Done when

- `answer_doc_v2.yaml` active; rules 1–3 in it; JSON contract unchanged.
- Evidence check implemented behind `answer.evidence_quotes` (default false).
- Before/after notes for T004/T006 in `docs/progress.md`; one commit, no push.
- Then the full check for all four prompts: `python -m pytest -q` and `ruff check .`.
