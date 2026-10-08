# 03 · Retrieval: line-item synonyms and a measured topic boost

**Model:** Sonnet 5.5, effort high · **Branch:** `improve/answer-quality-and-highlight` · **Needs:** prompt 01 (trap set)

## Read first

`docs/build-prompts/improvements/README.md`, `app/routing/enhancer.py` (`ABBREVIATIONS`, `SYNONYMS`, how
`search_query` is built), `app/retrieval/retriever.py`, `app/retrieval/bm25.py`, the `retrieval:` block of
`config.yaml`, `eval/README.md` (retrieval eval + CI gate), the last entries of `docs/progress.md`.

## The problem, measured

Two trap questions abstained because the answer never reached the model's 8 passages:

- **T009** "What dividend per share did the company pay for FY2026?" The report says "the Board has not
  recommended any dividend … for the financial year ended March 31 2026" (FY26 PDF p.36) and
  "Dividend Declared: None" (p.2). Retrieved instead: e-voting, IPO and share-capital pages.
- **T010** "What was finance cost in FY2026 and why did it decline…?" The Financial Highlights table row
  "Finance Cost 94.90 171.40" (FY26 PDF p.34) was not retrieved, **although "finance cost" is already in the
  query**. Retrieved instead: a ratio-formula page, a Board-report page and a note heading.

T004's "what drove the change" also got a notes page instead of the Board report's discussion.
`theta` (0.25) is not the cause: these went to the LLM, which correctly said INSUFFICIENT.

## Step 1: diagnose before changing (free, no LLM)

With the API running and `DOCQA_DEBUG=1`, call `POST /debug/retrieve` for T009, T010 and T004 (both the raw
question and the enhancer's `search_query`). Or stop the API and run
`python -m eval.retrieval_eval --questions eval/questions_traps.jsonl`, which lists misses and top pages
in `eval/results/retrieval_latest.json`. Find out *why* the right chunk loses: where does it rank in BM25 and in
dense? Is the table row split from its heading? Do the extra words ("why did it decline compared with") dominate?
Write the findings (3–6 lines) into `docs/progress.md`. Let them decide which of the changes below are worth
making. If the diagnosis points at a different, smaller fix, prefer that and say why.

## Step 2: candidate changes (keep what helps)

**A. Line-item synonyms (`app/routing/enhancer.py`, `SYNONYMS`).**
Today `SYNONYMS` has 7 entries (`net worth` → shareholders' funds, …), and nothing for finance cost or dividend.
Add a short list (about 8–12) of concept-to-concept aliases common in Indian annual reports, e.g.:

| phrase | appended to the search query |
|---|---|
| finance cost | finance costs; interest expense; borrowing costs |
| dividend | dividend declared; recommended dividend; dividend distribution |
| borrowings | debt; loans; term loan |
| other income | interest income; treasury income |
| net worth | total equity; shareholders' funds |
| capital expenditure | additions to property, plant and equipment |

(Idea from sifra-v2's `LINE_ITEM_SYNONYMS`; keep ours short and generic, no company-specific words.)
Watch the BM25 weight (`bm25_weight: 20`): appended words can also *dilute* a keyword query. Measure.

**B. Topic boost (`app/retrieval/retriever.py`), behind a config flag that is off by default.**
A small, fixed score bonus for a chunk whose own text contains a heading that matches the question's topic,
e.g. a dividend question favours chunks containing "DIVIDEND"; "what drove / why / reason" favours
"REVIEW OF OPERATIONS" / "PERFORMANCE REVIEW" / "FINANCIAL HIGHLIGHTS"; a finance-cost/EBITDA/PBT question
favours "FINANCIAL HIGHLIGHTS" and "Statement of Profit and Loss". Keep the topic → heading map in `config.yaml`
(or one small dict) so it can be tuned without code changes. (sifra-v2 does the same idea with a reranker
instruction per question shape; we do it with a cheap bonus.) Apply it after fusion, before the top-k cut.
It must stay O(candidates), no extra model or I/O.

## Step 3: measure (free) and decide

Stop the API (the eval opens the same `data/`), then run:

```
python -m eval.retrieval_eval --questions eval/questions_traps.jsonl   # trap set
python -m eval.retrieval_eval                                          # main 117-question set
python -m eval.retrieval_eval --fixture --check                        # the CI gate
```

Record before/after Recall@5 and MRR for the trap set and the main set in `docs/progress.md`.
Keep a change if it helps the trap set (T009/T010/T004 retrieved) **and** does not make the main set or the CI
gate worse beyond the gate's own tolerance. Otherwise leave it switched off (flag) or drop it, and say so.
If the CI gate's baseline changes on purpose, follow `eval/README.md` (`--write-baseline`, with a reason).

Optional (spends Groq quota, about 5K tokens a question): re-ask T009 and T010 through `/query` to see whether
they now get answered. Not required.

## Tests (just for this change)

Enhancer: a question with "finance cost" gets the aliases in `search_query`; one without doesn't. Retriever:
with the boost flag on, a chunk containing the matching heading moves up; with it off, order is unchanged. Use
the existing fakes in `tests/fakes.py`. Run the touched test files and `ruff check` on changed files.

## Done when

- Diagnosis written down; synonyms added; topic boost implemented behind a flag (on or off by measurement).
- Before/after retrieval numbers in `docs/progress.md`; one commit, no push.
