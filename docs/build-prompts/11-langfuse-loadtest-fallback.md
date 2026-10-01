# 11: Langfuse, Load Test, Ollama Fallback

**Effort:** medium · **Prereqs:** 08 · **Design refs:** §12 (fallback), §13, §14 (throughput), §15 (Langfuse)

Three independent nice-to-haves. Do them in one session or split them; each is small.

````text
Read docs/design.md (§12, §13, §14, §15) and docs/progress.md.

Task: three small additions. Each one must be optional and off by default if its env vars are missing.

1. Langfuse tracing
   - Use the Langfuse SDK (@observe decorators or equivalent) to create a span tree per request:
     router -> retrieve -> generate (+ general path), with prompt versions, model IDs, token usage and the
     retrieved source IDs. Attach 👍/👎 feedback and judge scores as Langfuse scores.
   - Enabled only when LANGFUSE_* env vars exist. Must never block or fail a request.

2. Load test
   - scripts/load_test.py with asyncio + httpx: 1 / 3 / 5 concurrent users sending a mix of eval questions
     for a fixed duration. Report req/min, p50/p95 latency, error rate and 429 count, and where it breaks.
   - Write the results to eval/results/load_test.json and add them to progress.md.

3. LLM backend fallback
   - Config/env switch between Groq and a local Ollama (OpenAI-compatible base URL, model names in config),
     e.g. LLM_BACKEND=groq|ollama.
   - Optional automatic fallback: if Groq times out or returns 429 repeatedly, retry the call on Ollama when it's
     configured, and record which backend served it in the request log.
   - Document how to pull the fallback model (gpt-oss:20b, or llama3.2:3b for weaker laptops).

Tests: the backend selection/fallback logic (mocked); tracing disabled when env vars are missing.

When done: run tests, append a "Step 11" entry to docs/progress.md, and stop.
````
