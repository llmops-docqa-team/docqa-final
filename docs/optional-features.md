# Optional features: Langfuse tracing, Ollama fallback, load test

All three are **off unless you switch them on**, and none of them can block or fail a request. The request
log in SQLite stays the source of truth; these are for debugging, resilience and sizing.

## Langfuse tracing

Shows one span tree per request (design §15): why a particular answer went wrong, with the prompt versions,
models, token usage and retrieved chunk ids on it.

```
query
├─ router        generation: model, prompt version, tokens, route
├─ document      span: status, abstain reason, top score, cited chunks
│  ├─ retrieve   span: retrieved chunk ids and scores
│  └─ generate   generation: model, prompt version, tokens, backend
└─ general
   └─ generate
```

- **Switch on:** put `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env` (git-ignored).
  Docker Compose loads `.env` into the API container; the API reads real environment variables only, so when
  you run `uvicorn` yourself, export them first. Without both keys, or with `LANGFUSE_TRACING_ENABLED=false`,
  tracing is a no-op. The startup log says `tracing_on` or nothing.
- **What is sent:** ids, versions, models, token counts, timings, chunk ids, scores. **Not** the question, the
  answer or any document text, unless you set `tracing.capture_content: true` in `config.yaml` (then the
  question, the answer and the 300-character cited snippets are sent to Langfuse's servers; whole chunks never are).
- **Trace id = the request id.** The `trace_id` in the `/query` response and the `requests` table is the key:
  Langfuse's own id is derived from it, so a 👍 or a judge score that arrives later, even from another process, lands on the same trace.
- **Scores:** 👍/👎 become `user_feedback` (1 / -1) when `/feedback` is called. `scripts/judge_recent.py` attaches
  `judge_correct` and `judge_grounded` (boolean) to the traces it judges. The offline eval runner is not traced.
- Spans are exported in a background thread. A Langfuse outage costs nothing but the traces.

## Ollama: backend switch and automatic fallback

Groq and Ollama both speak the OpenAI API, so one client serves both.

1. Install Ollama (https://ollama.com) and pull a model:
   ```
   ollama pull gpt-oss:20b      # same family as the router; needs a 16 GB+ laptop
   ollama pull llama3.2:3b      # weaker machines (then set both models below to llama3.2:3b)
   ```
   Model names live in `config.yaml` under `llm.ollama` (`answer_model`, `router_model`).
2. **Automatic fallback** (Groq stays the primary): set `OLLAMA_BASE_URL` in `.env`. From the Docker container the
   host is `http://host.docker.internal:11434/v1`; for a local `uvicorn` use `http://localhost:11434/v1`.
   When a Groq call times out, is rate-limited (429) or fails with a 5xx after its normal retries, the same
   call is repeated on Ollama. Groq is then skipped for `llm.fallback_cooldown_seconds` (30) so the next
   requests do not each wait out the retries again. A rejected request (400, 401) is not retried elsewhere.
3. **Ollama as the primary:** `LLM_BACKEND=ollama` (or `llm.backend: ollama`). No fallback in that mode.
4. Every request row records `llm_backend` (`groq`, `ollama` or `groq+ollama`), `llm_fallbacks` (calls that
   fell back) and `llm_rate_limited` (429s seen). The `/query` response has the same in its `llm` block.
5. The dev LLM cache never stores a fallback answer, and the eval judge (Gemini) never falls back.

A local model is slower and less accurate than `gpt-oss-120b`: the abstention and number checks still run,
but expect different answers. `llm.ollama.timeout_seconds` (120) covers the first call, which loads the model.

## Load test

```
python scripts/load_test.py                       # against http://localhost:8000
python scripts/load_test.py --levels 1,3 --max-requests 4 --token-budget 30000 --cooldown 65
```

1 / 3 / 5 concurrent users send a seeded mix of `eval/questions.jsonl` questions (50% document, 30% general,
20% mixed). It prints requests/min, p50/p95 latency, error rate, 429 count and a verdict per level, says where
the system first breaks, and writes `eval/results/load_test.json` (git-ignored, like the other results).

It spends real quota, so it is small by default: at most 6 requests per level, 30 s per level, a 60,000-token
budget for the whole run, 65 s between levels. Raise the limits deliberately. Upload the EIG report first,
otherwise every question is answered as general knowledge (the script warns).

How to read it: a 429 is counted whenever the Groq backend answered 429 to one of the request's calls, even if a
retry then succeeded. A request that returns 200 but whose answer part failed because the LLM was unavailable
is **degraded**. Error rate = (HTTP failures + timeouts + degraded) / requests; **broken** means >= 5%,
**strained** means any 429 or p95 above 6 s.
