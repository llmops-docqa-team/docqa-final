# 06: Router, General Path, MIXED, `/query`

**Effort:** high · **Prereqs:** 05 · **Design refs:** §9, §11 (behaviour table, policy), §17 B, addendum §4 (router fallback metric)

````text
Read docs/design.md (§9, §11, §17) and docs/addendum.md, plus docs/progress.md.

Task: the real query endpoint. Route each question, run the right path(s) and compose one response.

Build:
- prompts/router_v1.yaml: includes the titles of uploaded documents (with their status) and ~8 few-shot
  examples, including the hard ones from design §9 ("What is EBITDA?" -> GENERAL; "What was X's EBITDA margin?"
  -> DOCUMENT; "What was the revenue?" with one report -> DOCUMENT; a MIXED example with a split).
- app/routing/router.py: gpt-oss-20b call in JSON mode -> Pydantic model {route, document_question, general_question}.
  Bad JSON -> retry once -> fall back to DOCUMENT with the whole question. Log the fallback as a warning and
  return router_ok=false so it shows up as a metric. If no documents exist at all, skip the router -> GENERAL.
- prompts/answer_general_v1.yaml + app/answering/general.py: answer from model knowledge. Tell the model to
  say when it's unsure of exact figures. Always attach the label
  "General knowledge — not from your documents; may be out of date."
- POST /query {question}: router -> DOCUMENT / GENERAL / MIXED. For MIXED, run both paths in parallel
  (threads or asyncio, your call) and return two labelled sections: document part and general part, each with
  its own status (the doc part may abstain while the general part answers).
  Handle all the cases in design §11: documents still processing (with n/N pages), FAILED documents,
  no documents, vague questions (proceed, abstain with a "be more specific" hint if weak).
  Question length limit ~500 chars (config).
- Response JSON: trace_id, route, sections[{kind, status, answer, citations[], label?, abstain_reason?,
  closest_pages?, warnings[]}], timings, tokens. Shape it so the UI is easy to build; adjust if needed.
- Remove or keep the step-05 debug endpoint as you see fit (keep it behind a debug flag if kept).

Tests (LLM mocked): router JSON parsing + fallback; no-docs shortcut; MIXED runs both parts and composes;
processing-doc message; general label always present; question too long -> 422.

Try 4 real questions by hand, one per route plus a near-miss, and note results.

When done: run tests, append a "Step 06" entry to docs/progress.md, and stop.
````
