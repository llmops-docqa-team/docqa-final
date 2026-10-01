# Document-Aware Q&A System — Design Review & Architecture Plan

*LLMOps Portfolio Project · ML System Design & LLMOps · Scaler School of Technology*
*Prepared before implementation · 5 Oct 2026*

**Inputs reviewed:** current problem statement, Deliverables Checklist, Presentation Rubric, Week 1–7 course study guides. (The Sifra-v2 reference codebase was intentionally excluded; nothing here depends on it.)

---

## 1. Executive Summary

**What we should build:** a small, well-measured RAG system for **text-based English financial PDFs (company annual reports)**. Users upload a PDF, watch it process in the background, and ask questions. Each question is **routed** (DOCUMENT / GENERAL / MIXED). Document questions are answered **only from retrieved passages, with page citations**, or the system **abstains**. General questions are answered from the LLM's own knowledge and **clearly labelled** as such.

**The core idea is sound.** Routing + abstention + async ingestion is a good fit for this course: it is small enough to build, and every part of it can be measured. The current statement is too vague to grade well, though. It has no domain, no numbers, no evaluation plan and no ops story. Those cover 50% of the rubric (problem framing 20% + LLMOps depth 20% + part of trade-offs).

**Key decisions (in one line each):**

| Area | Decision |
|---|---|
| Scope | One domain (annual reports), text-layer PDFs only, single-turn questions, single user |
| Ingestion | Async with **one background worker thread** + SQLite job table. **No partial querying in MVP.** A document is queryable only when `READY` |
| Routing | **One structured LLM call** that classifies *and* splits mixed questions. It is benchmarked against a rules baseline and an embedding-score baseline on a labelled set |
| Retrieval | **Dense-only** (bge-small via fastembed + Chroma), page-aware chunks, top-5. BM25 and a reranker are **experiments**, adopted only if they move Recall@5 |
| Abstention | Two gates: retrieval-score threshold (calibrated) + LLM "insufficient evidence" signal. Then citation validation and a numeric grounding check |
| LLM | **gpt-oss-120b** (answers) and **gpt-oss-20b** (router), both open-weight (Apache 2.0), served free via Groq's API. Local Ollama fallback for the demo |
| LLMOps | Versioned prompts/config in git, per-request trace log (SQLite) + Langfuse, MLflow for eval runs, a CI retrieval-eval gate, thumbs feedback, sampled LLM-as-judge |
| Evaluation | ~85 hand-written, labelled questions over 2 annual reports. Router accuracy, Recall@5/MRR, answer correctness, citation accuracy, abstention precision/recall, latency p50/p95/p99 |
| Not building | Agents, LangChain/LlamaIndex, fine-tuning, OCR, Kafka/Redis/K8s, multi-tenant auth, chat memory, web search, model registry, feature store |

---

## 2. Current Problem Statement Review

> *Current:* "Build an AI-based document-aware QA system where users upload PDFs, the system processes them asynchronously, and users can ask natural-language questions. It distinguishes document / general / mixed questions, supports async ingestion (possibly querying partially indexed content), abstains when evidence is insufficient, and never blocks general questions on ingestion."

### What is good
- **Abstain rather than hallucinate** is the right core value, and it is *measurable* (abstention precision/recall).
- **Routing** gives the project a real ML component (classification + decomposition) beyond "plain RAG". It also has a clear failure mode to discuss.
- **Async ingestion** naturally gives us a sync-vs-async interaction to explain, which the rubric asks for explicitly.
- "General questions shouldn't be blocked by ingestion" is a good, concrete product rule.

### What is missing
| Missing | Why it hurts |
|---|---|
| A one-sentence **business objective** | Problem framing (20%) asks for it explicitly |
| **Domain / document type** | "Any PDF" makes chunking, evaluation and the demo impossible to reason about |
| **Numbers**: latency SLA, accuracy thresholds, doc size limits, scale | Rubric: "requirements with specific values". README: "no numbers, no credit" |
| Formal **ML problem** (input → output → target) | Rubric asks for it |
| Definition of **"sufficient evidence"** | Without it, abstention is a vibe, not a rule |
| **Evaluation set** and scoring method | A required deliverable (30–50+ Q&A pairs for RAG) |
| **Observability / cost / latency tracking** | A required deliverable |
| **Deployment, rollout, rollback** | LLMOps depth (20%) |
| **What's out of scope** | Rubric: "scope is bounded" |

### What is too broad
- **"Any PDF"** → scanned PDFs need OCR, slide decks and forms behave differently. Restrict to **text-layer annual reports**.
- **"Natural-language questions"** → includes whole-document summaries, multi-year calculations and cross-document comparisons. Restrict to **fact lookup and short explanations answerable from 1–3 passages**.
- **Multi-turn chat** is implied by "ask questions". Follow-ups like "and its profit?" need conversation memory. Make MVP **single-turn**.

### What is unnecessary (for MVP)
- **Partial querying during ingestion.** Our measured-to-be ingestion time is ~1–3 minutes per report. Partial querying adds tricky semantics ("not found *yet*" vs "not in the document") for little user value. See §8.

### What is ambiguous
- **When is a question "document-related"?** "What was Ellenbarrie's revenue in FY2025?" is DOCUMENT if an Ellenbarrie report is uploaded. If not, it is a general question that the LLM may answer *from memory*, which is risky for exact numbers. We need a stated policy (see §11).
- **"Mixed" answer format.** One blended paragraph or two labelled parts? (Recommendation: two labelled parts.)
- **General questions about current facts** ("Who is the PM of India?") are answered from the model's training data, which has a cutoff. Without web search, these can be out of date. The answer must say so.
- **Annual reports contain both standalone and consolidated figures.** "Revenue" is ambiguous inside the document itself.

### What could create implementation problems
| Risk | Why |
|---|---|
| **Tables** in annual reports | Text extraction scrambles table rows; numeric questions are exactly what users ask |
| **Near-miss questions** ("FY2030 revenue") | Retrieval finds the FY2025 revenue chunk with a *high* score, so a score threshold alone will not abstain |
| **Free-tier LLM rate limits** | Groq free plan: ~8K tokens/min, ~200K tokens/day per model. A full eval run can exceed a day's quota |
| **Embedding model limit** | bge-small truncates at 512 tokens, so chunks must stay under that |
| **Free hosting** | Hugging Face Spaces no longer offers free CPU Docker Spaces to new Spaces (reported mid-2026); storage on free hosts is usually ephemeral |

### What could improve grading / technical depth
1. Measured **experiments** (chunk size, BM25 on/off, threshold, router variants) logged in MLflow → real "we chose X over Y because Z" trade-offs.
2. A **router benchmark** that compares the LLM router against two cheap baselines.
3. **Judge calibration**: check LLM-as-judge against ~20 human labels and report the agreement.
4. A **CI eval gate**: a PR that drops Recall@5 fails the build.
5. A **failure-modes table** with detection and recovery for each mode.

---

## 3. Refined Problem Statement

> **Business objective (one sentence):** Help an analyst get correct, page-cited answers from long company annual reports in seconds instead of minutes of manual searching, without ever presenting an unsupported "fact" as coming from their document.

> **Problem statement:** Given a user's question and a set of uploaded, text-based English annual-report PDFs, the system (1) classifies the question as DOCUMENT, GENERAL or MIXED and splits mixed questions into parts; (2) for document parts, retrieves the top-5 passages from fully-indexed documents and generates an answer citing document and page, or **abstains** when the passages do not support an answer; (3) for general parts, answers from the LLM's own knowledge with an explicit "not from your documents" label. Documents are ingested asynchronously; general questions are never blocked by ingestion, and document questions about a still-processing document get an explicit "not ready yet" response.

**Formal ML/LLM problem:**

| | Definition |
|---|---|
| **Input** | Question `q` (text, ≤ 500 chars) + set of READY document chunks `C` + list of document titles |
| **Output** | `route ∈ {DOCUMENT, GENERAL, MIXED}`; for each document part either `{answer, citations[(doc, page)]}` or `ABSTAIN(reason)`; for each general part `{answer, label}` |
| **Prediction targets** | (a) route label (3-class classification); (b) relevance of chunks to `q` (ranking); (c) grounded answer text, or abstention (conditional generation + binary decision) |
| **Objective** | Maximise correct, cited answers on answerable questions **subject to** false-answer rate on unanswerable questions ≤ 10% |

---

## 4. What We Are Actually Building

**One sentence:** A single-user web app (Streamlit UI + FastAPI backend) for uploading annual-report PDFs and asking questions. It answers document questions with page citations or refuses, answers general questions with a label, and logs every request so latency, cost and quality can be measured.

**The four problems, kept separate:**

| Layer | The problem |
|---|---|
| **User problem** | Annual reports are 200–350 pages. Finding "consolidated revenue FY25" or "number of employees" means Ctrl-F through PDFs, and generic chatbots make numbers up |
| **Product problem** | Fast answers you can **verify** (citation → page), honest "I couldn't find it", no waiting on uploads for unrelated questions, and a clear status for processing documents |
| **ML problem** | (1) Route classification + decomposition; (2) dense retrieval/ranking of chunks; (3) an abstention decision (is the evidence sufficient?) |
| **LLM problem** | Language understanding in the router (intent + splitting); grounded generation with citations; general-knowledge answering; LLM-as-judge for offline/online evaluation |
| **LLMOps problem** | Version prompts/config/embedding model; evaluate retrieval, routing and answers offline; trace each request (latency per stage, tokens, cost, route, abstention); collect feedback; gate changes in CI; roll back by config |

---

## 5. Requirements

### Functional requirements
| ID | Requirement |
|---|---|
| F1 | Upload a PDF (≤ 25 MB, ≤ 400 pages, text layer required). Get a document ID immediately (HTTP 202) |
| F2 | See each document's status: `QUEUED → PROCESSING (n/N pages) → READY` or `FAILED (reason)` |
| F3 | Ask a question at any time. GENERAL questions are answered regardless of ingestion state |
| F4 | DOCUMENT answers cite `filename, page` for every claim and show the source snippet |
| F5 | Abstain with a clear reason when evidence is insufficient, a document is not ready, or there are no documents |
| F6 | MIXED questions return two labelled sections (document part, general part) |
| F7 | Thumbs up/down on each answer, stored with the request trace |
| F8 | Metrics page: p50/p95/p99 latency, route mix, abstention rate, errors, cost per request |

### Non-functional requirements (targets, to be verified by measurement)
| Category | Target |
|---|---|
| Query latency (Groq backend) | **p50 ≤ 3 s, p95 ≤ 6 s** end-to-end, measured on the eval set |
| Ingestion latency | **≤ 90 s for a 100-page report, ≤ 4 min for 350 pages** on a 4-core laptop CPU |
| Router accuracy | **≥ 90%** overall; **≥ 95%** recall on DOCUMENT (a missed document route is the costly error) |
| Retrieval | **Recall@5 ≥ 0.80** (gold page in top-5) |
| Answer correctness | **≥ 80%** on answerable document questions |
| Abstention | **≥ 90%** abstention on unanswerable questions; **≤ 10%** wrong abstention on answerable ones |
| Citation accuracy | **≥ 90%** of citations point to a gold page |
| Cost | $0 actual (free tiers); tracked as list-price-equivalent per request (≈ $0.001) |
| Scale | 1–5 concurrent users, ≤ 5 documents, ≤ 2,000 chunks per document |
| Reliability | No 5xx on valid input; LLM timeout → graceful message, not a crash |
| Reproducibility | `docker compose up` + seed script reproduces the demo; every eval run is logged with git hash and config |

### Back-of-envelope estimates (Week 5, Step 5)
| What | Estimate |
|---|---|
| Chunks per 300-page report | ~300 pages × ~3 chunks ≈ **900 chunks** |
| Vector storage | 900 × 384 dims × 4 B ≈ **1.4 MB** per report: trivial, no scaling concern |
| Embedding time | ~900 chunks on CPU with ONNX bge-small ≈ **30–90 s** (must measure) |
| Tokens per document query | router ~500 + answer prompt ~1,800 (5 chunks × ~350) + output ~250 ≈ **2.5K tokens** |
| Free-tier budget | 8K TPM per model → **~3 document queries/min** sustained; 200K tokens/day per model → a full LLM eval run must use the dev cache and/or be split across days |

---

## 6. MVP vs Nice-to-Have vs Future

### MVP (must have, no exceptions)
1. Upload → background ingestion → status polling → READY/FAILED
2. Page-aware chunking, bge-small embeddings, Chroma storage, embedding-version tag
3. LLM router (JSON output, validated) with a fail-safe default
4. Document path: retrieve top-5 → score gate → grounded answer → citation validation
5. General path with a "general knowledge" label; MIXED = both paths in parallel
6. Abstention messages for all cases in §11
7. Versioned prompts + config in repo; per-request trace log in SQLite
8. Eval set (~85 questions) + eval runner (router, retrieval, answers, abstention, latency) logged to MLflow
9. LLM-as-judge (different model family) + thumbs feedback
10. Streamlit metrics page; Dockerfile + docker-compose; README with numbers
11. pytest suite + GitHub Actions CI with a retrieval-eval gate

### Nice-to-have (only after MVP numbers are in the README)
- Langfuse tracing (recommended as the first nice-to-have; it's small)
- **Numeric grounding check** as a hard gate (MVP logs it as a warning; enforce only if eval shows it helps)
- BM25 + RRF hybrid retrieval (**only if** it lifts Recall@5 by ≥ 5 points)
- Cross-encoder reranker (same rule)
- Table-aware extraction (PyMuPDF `find_tables`) for financial statements
- Partial querying during ingestion (§8)
- Automatic LLM fallback to local Ollama on API timeout
- Follow-up question rewriting using the previous turn
- Public deployment

### Future work (explicitly out of scope)
OCR for scanned PDFs · web search for current-events questions · multi-user auth and per-user document isolation · conversation memory · cross-document comparisons and multi-year calculations · fine-tuning · agents · non-English documents · Excel/Word inputs · streaming token output.

---

## 7. End-to-End Architecture

```
                   ┌──────────────────── Streamlit UI ─────────────────────┐
                   │  Upload  │  Documents + status  │  Ask  │  Metrics   │
                   └─────┬────────────────┬───────────────┬────────────────┘
        REST multipart   │  REST JSON poll│ (every 2 s)   │ REST JSON (sync)
        → 202 Accepted   ▼                ▼               ▼
┌───────────────────────────────── FastAPI service ──────────────────────────────────┐
│                                                                                    │
│  INGESTION (async)                          QUERY (sync, per request)              │
│  POST /documents                            POST /query                            │
│     │ validate, save file, hash                │                                   │
│     ▼                                          ▼                                   │
│  SQLite: documents/jobs ◄──status──┐      Router (LLM, JSON) ──► route + sub-qs    │
│     │ enqueue                      │           │                                   │
│     ▼                              │     ┌─────┴───────────────┐                   │
│  Worker thread (1):                │     ▼                     ▼  (parallel)       │
│   parse (PyMuPDF) → chunk →        │  Document path         General path           │
│   embed (fastembed bge-small) →    │   embed q → Chroma       LLM answer           │
│   upsert batch → update progress ──┘   top-5 (READY docs)     + "general" label    │
│                    │                    → score gate                               │
│                    ▼                    → LLM grounded answer                      │
│              Chroma (vectors +          → citation + number checks                 │
│              chunk metadata)  ◄─────────┘                                          │
│                                          Compose answer ─► request log (SQLite)    │
│                                                           ─► Langfuse (async, opt) │
└────────────────────────────────────────────────────────────────────────────────────┘
        External: Groq API (OpenAI-compatible) — gpt-oss-20b (router), gpt-oss-120b (answers)
        Offline:  eval runner → MLflow (local) ; judge = different model family (Gemini Flash)
```

**Explainable in 2 minutes:** *"There are two paths. Uploads go into a background queue: we parse, chunk and embed them, then mark them ready. Questions go through a router that decides whether they need the documents. Document questions retrieve 5 passages and are answered only from them, with page citations, or we refuse. General questions go straight to the LLM and are labelled. Everything is logged, so we can measure it."*

### Stage-by-stage

| Stage | What / why | Input → Output | Tech | Failure modes | Needed? |
|---|---|---|---|---|---|
| Upload & validate | Reject bad files early | PDF → `doc_id`, 202 | FastAPI, PyMuPDF open | Not a PDF, too big, encrypted, no text layer | Yes |
| Job queue & status | Uploads can't block requests; user needs progress | doc_id → status rows | SQLite + `queue.Queue` + 1 thread | Crash mid-job → re-queue on restart | Yes |
| Parse | Get text **with page numbers** (citations) | PDF → `[(page, text)]` | PyMuPDF | Tables scrambled, headers/footers noise | Yes |
| Chunk | Units small enough to retrieve precisely | pages → chunks (~400 tok) | Own ~40-line function | Chunk > 512 tokens truncated by embedder | Yes |
| Embed & index | Make chunks searchable | chunks → vectors in Chroma | fastembed (ONNX) bge-small-en-v1.5; Chroma | Model mismatch between index and query | Yes |
| Route | Decide if docs are needed; split mixed | q + doc titles → JSON | gpt-oss-20b, JSON mode, Pydantic | Invalid JSON, wrong class | Yes (see §9) |
| Retrieve | Find evidence | sub-q → top-5 chunks + scores | Chroma cosine, `status=READY` filter | Right page not in top-5 | Yes |
| Evidence gate | Don't call LLM on clearly irrelevant evidence | top score → pass/abstain | Threshold θ (calibrated) | Near-miss questions pass the gate | Yes (cheap) |
| Generate (doc) | Grounded answer with citations | sub-q + chunks → answer JSON | gpt-oss-120b | Ignores context, invents numbers | Yes |
| Validate output | Catch fake citations / unsupported numbers | answer → ok / abstain / warning | Python checks | Units/rounding false alarms | Yes |
| General answer | Answer non-document parts | sub-q → labelled answer | gpt-oss-120b | Outdated facts | Yes |
| Compose & log | One response; measurable system | parts → response; trace row | FastAPI, SQLite | Logging must never fail the request | Yes |

---

## 8. Document Ingestion

### Design (deliberately simple)
- `POST /documents` validates the file and computes a SHA-256 hash. A duplicate hash reuses the existing document. The endpoint saves the file to `data/uploads/`, inserts a `documents` row (`QUEUED`) and returns **202 + doc_id**.
- **One worker thread**, started with the app, takes jobs from an in-process queue:
  1. `PROCESSING`: parse pages with PyMuPDF.
  2. **Text-layer check**: if fewer than ~50 characters per page on average → `FAILED: "scanned PDF not supported"`.
  3. Chunk → embed in batches of 32 → **upsert to Chroma every ~20 pages** → update `pages_done` in SQLite (the progress bar).
  4. `READY` with stats (`pages, chunks, seconds, embedding_model`); or `FAILED` with the error message.
- **Deterministic chunk IDs** (`{doc_id}:{page}:{i}`), so re-running a job overwrites rather than duplicates. On app start, any `PROCESSING` document is re-queued, which handles crashes.
- UI polls `GET /documents` every 2 s.

### Why not Celery / Redis / RQ?
One user, a handful of documents, CPU-bound work. A thread + SQLite gives the same visible behaviour (upload → processing → ready/failed) with zero extra services. **Trade-off accepted:** jobs don't survive a hard crash mid-job without the re-queue step, and ingestion is serial (one document at a time). At 10× usage this is the first thing to replace (with RQ + Redis or a separate worker process).

### Partial querying during ingestion — verdict: **not in MVP**
| | Full readiness (MVP) | Partial querying |
|---|---|---|
| User value | Wait ~1–3 min once per document | Answers a minute earlier, sometimes |
| Semantics | "Not ready" is clear | "Not found **in pages 1–120 of 300**". The user can't tell whether it will appear later |
| Abstention | One rule | Must distinguish "absent" vs "not indexed yet" and report coverage |
| Evaluation | Simple | Answers depend on timing, so evaluation is non-deterministic |
| Effort | — | Small code change, larger testing/explanation cost |

Because we already upsert in page batches, enabling partial querying later is cheap: drop the `status=READY` filter and add a coverage note to abstentions. **Do it only if measured ingestion for a real report exceeds ~3 minutes.** That's the condition under which it actually helps.

### Behaviour while a document is processing
- GENERAL questions: answered normally.
- DOCUMENT questions, no READY docs: *"EIG_AR_FY25.pdf is still processing (140/312 pages). I'll be able to answer document questions once it's Ready."*
- DOCUMENT questions, some docs READY, others processing: retrieve from READY docs. If the system abstains, it adds *"Note: X.pdf is still processing and may contain the answer."*

---

## 9. Query Routing

### Options compared
| Option | Pros | Cons | Verdict |
|---|---|---|---|
| Keyword rules | Free, instant, transparent | Breaks on paraphrase; can't split mixed questions | **Baseline** |
| Retrieval-score threshold ("is top chunk similarity > τ?") | Free byproduct of retrieval; knows what's in the doc | Can't split mixed; general questions sharing words with the doc score high ("What is EBITDA?") | **Baseline** |
| Trained small classifier (DistilBERT/LogReg) | Fast, cheap at serve time | Needs hundreds of labelled examples we don't have; still can't split | Reject |
| Local LLM router | Free, offline | Slow on CPU (5–20 s) | Reject (fallback only) |
| **Structured LLM call** | Understands intent; **splits MIXED into sub-questions** in the same call; doc titles give context | ~0.5 s and ~500 tokens per query; can fail to produce valid JSON | **Chosen** |

**Why an LLM, despite "keep it simple":** MIXED questions need **decomposition** ("Ellenbarrie's FY25 revenue" + "Who is the PM of India?"), and only an LLM does that without a pile of brittle rules. One call does both jobs.

### Router contract
Prompt (versioned, `prompts/router_v1.yaml`) includes the **titles of uploaded documents** (READY and processing) and ~8 few-shot examples, including hard ones:
- "What is EBITDA?" → GENERAL (a definition, even though the report mentions EBITDA)
- "What was Ellenbarrie's EBITDA margin?" → DOCUMENT
- "What was the revenue?" (with one report uploaded) → DOCUMENT

Output (validated with Pydantic):
```json
{"route": "MIXED",
 "document_question": "What was Ellenbarrie's revenue in FY2025?",
 "general_question": "Who is the Prime Minister of India?"}
```

**Fail-safe:** invalid JSON → retry once → otherwise treat the whole question as **DOCUMENT**. Worst case the user gets an abstention, never an invented document fact.

**No documents uploaded at all:** skip the router; everything is GENERAL (with the label).

### How we justify it with data
On the labelled eval set (~85 questions), report accuracy and a 3×3 confusion matrix for: (a) rules, (b) retrieval-score threshold, (c) LLM router. Expected outcome is that the LLM router wins mainly on MIXED. **If a baseline reaches ≥ 90% on DOCUMENT/GENERAL, say so honestly**: a hybrid (rules first, LLM only when uncertain) then becomes a future cost optimisation.

---

## 10. Retrieval / RAG

| Decision | Choice | Notes |
|---|---|---|
| Parser | PyMuPDF, page by page | Keeps page numbers for citations. Strip repeated headers/footers (lines that appear on > 50% of pages) |
| Chunking | Recursive split **within a page**, ~400 tokens, ~60 overlap | Never crosses pages, so each citation is exactly one page. 400 keeps us under bge-small's 512-token limit with room for the header |
| Contextual header | Prepend `"{doc title} — page {n}"` to each chunk before embedding | Cheap; helps "Ellenbarrie's revenue" match chunks that never say "Ellenbarrie" |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d) via **fastembed** (ONNX) | Fast on CPU, MIT licence, no PyTorch needed (smaller image). Use the bge **query instruction prefix** for queries |
| Store | **Chroma** (persistent, local), cosine | Course rule: prototype → Chroma. Metadata filter on `doc_id`/`status` |
| Metadata per chunk | `doc_id, filename, page, chunk_idx, text, char_len, embedding_model, ingest_version` | |
| Retrieval | Dense top-**8**, pass top-**5** to the LLM | ~1,800 tokens of context; fits free-tier TPM |
| Hybrid / BM25 | **Experiment only** (`rank_bm25` + RRF, ~40 lines) | Adopt if Recall@5 improves ≥ 5 points. Financial reports have exact terms (FY codes, "consolidated") where BM25 *may* help |
| Reranker | **Experiment only** (`bge-reranker-base` or ms-marco-MiniLM on CPU) | Adopt if it improves answer correctness enough to justify +~1 s on CPU |
| Query rewriting / HyDE / multi-query | **Rejected** | Extra LLM calls and quota for unproven gains |
| Context construction | Each chunk wrapped as `<source id="S3" doc="..." page="47">…</source>`; system prompt says text inside is **data, not instructions** | Basic defence against indirect prompt injection (Week 7, Incident 1) |
| Citations | LLM returns `{"answer": "...", "citations": ["S1","S3"], "status": "ANSWERED"\|"INSUFFICIENT"}`; we map IDs → (filename, page, snippet) | The LLM never types page numbers itself, so it can't invent them |
| Embedding version guard | Store model name in the Chroma collection metadata. On startup, refuse to serve if config ≠ index | The "silent failure" from Weeks 4/6, prevented in 5 lines |

---

## 11. Answering / Abstention

### Evidence policy (what "sufficient" means, operationally)
A document part is **answered** only if **all** hold:
1. Top retrieval score ≥ **θ** (calibrated on the eval set: pick θ where the false-answer rate on unanswerable questions is ≤ 10%. The course demo used 0.5 vs a rule-of-thumb 0.7, which shows it must be calibrated per model).
2. The LLM returns `status = ANSWERED` (it was told to return `INSUFFICIENT` if the sources don't contain the answer, and to **never** use outside knowledge for document questions).
3. At least one citation ID is valid (exists in the provided sources).

Otherwise → **abstain**. Gate 1 catches off-topic questions cheaply (no LLM call). Gate 2 catches **near-miss** questions ("FY2030 revenue" retrieves FY2025 chunks with high scores), which a threshold alone cannot catch. That's why both exist.

**Numeric grounding check (MVP: warn + log; enforce later if it helps):** extract every number in the answer, normalise it (commas, ₹, %, crore), and check it appears in the cited chunks. If it fails → show a "⚠ number not found verbatim in source" badge and log `number_check=fail`. Financial users care most about numbers, and this ~30-line check measures hallucinated figures directly.

**Domain rule in the answer prompt:** if both standalone and consolidated figures appear, state which one is quoted (or give both).

### Behaviour table
| Situation | Response |
|---|---|
| Sufficient evidence | Answer + citations `[EIG_AR_FY25.pdf, p.47]` with expandable snippets |
| Insufficient evidence | *"I couldn't find this in your documents (searched: EIG_AR_FY25.pdf). I won't guess."* + "closest sections: p.112, p.47" |
| Document still processing | *"X.pdf is still processing (140/312 pages). Ask again when it's Ready."* |
| Document failed | *"X.pdf couldn't be processed: no text layer (scanned PDF)."* |
| No documents uploaded | Router skipped; answered as GENERAL with label |
| General question | Answer + label *"General knowledge — not from your documents; may be out of date."* |
| Mixed | Two sections, **📄 From your documents** and **🌐 General knowledge**, each with its own status. The document part may abstain while the general part answers |
| Ambiguous / vague ("tell me about it") | No clarification dialogue in MVP. Router + retrieval proceed; weak evidence → abstain with a hint to be more specific |
| LLM timeout / API error | *"The answer service is temporarily unavailable."* For document questions, also show the top-3 retrieved passages (the course's circuit-breaker fallback) |

**Policy for document-style questions with no matching document** ("Ellenbarrie revenue FY25" with no Ellenbarrie report uploaded): it routes GENERAL and is answered from model knowledge **with the label**. The general prompt tells the model to say when it is unsure of exact figures. We accept this; the label makes the source clear.

---

## 12. Models

| Role | Model | Where it runs | Why | Licence |
|---|---|---|---|---|
| Answer generation (doc + general) | **gpt-oss-120b** | Groq API, free plan (`reasoning_effort=low`) | Open-weight, strong at reading tables/numbers, very fast on Groq; on the free plan | Apache 2.0 |
| Router | **gpt-oss-20b** | Groq API, free plan | Classification + splitting needs speed, not depth; **separate free-tier quota** from the answer model | Apache 2.0 |
| Embeddings | **bge-small-en-v1.5** | In-process, CPU (fastembed/ONNX) | 33M params, fast on laptops, good English retrieval quality, 512-token limit | MIT |
| Judge (offline/online eval only) | **Gemini Flash** (Google AI Studio free tier) | API | **Different model family** from the generator, to avoid self-preference bias (Week 7); course-recommended | Proprietary (eval only) |
| Demo fallback | gpt-oss-20b via **Ollama** (on a 16 GB+ laptop), or `llama3.2:3b` on weaker machines | Local | Works without internet; same API shape (OpenAI-compatible) | Apache 2.0 / Llama licence |

**Why a hosted API for an "open-source first" project:** student laptops cannot run a 120B model, and a local 8–20B model on CPU takes 10–40 s per answer, which breaks the latency target and the demo. We use **open-weight models** through a free hosted endpoint, so there's no lock-in: the same weights can be self-hosted later (Ollama/vLLM) by changing `base_url` in config. **Trade-off accepted:** internet dependency and third-party rate limits.

**Why only one client:** Groq and Ollama both expose OpenAI-compatible endpoints, so one `openai` SDK client covers both. Switching backends = changing `LLM_BASE_URL` and `LLM_MODEL`.

> ⚠ Verify before building: Groq's free-plan model list and limits change often. As of Oct 2026 the free plan lists gpt-oss-120b/20b at ~30 RPM, 1K requests/day, 8K tokens/min, 200K tokens/day *per model*; the Llama models are no longer listed on the free plan. Also confirm the Gemini free tier.

---

## 13. MLOps / LLMOps

Only components that solve a real problem in **this** system:

| Component | Real problem it solves | Implementation | Keep? |
|---|---|---|---|
| **Prompt versioning** | Prompt changes silently change behaviour ("every prompt change is a deployment", Week 7) | `prompts/*.yaml` with `version`; active versions in `config.yaml`; version logged per request | ✅ |
| **Config versioning** | Reproduce any result; deliverable requires config in repo | `config.yaml`: models, θ, chunk size, top-k; git hash logged | ✅ |
| **Embedding-version guard** | Index/query model mismatch = silent garbage | Model name in collection metadata; startup check | ✅ |
| **Experiment tracking** | We'll run ~15 eval configurations and must compare them | **MLflow** (local file store), only for eval runs: params, metrics, git hash, eval-set hash | ✅ |
| **Offline evaluation** | "We measured whether it works" | Eval runner (§14) | ✅ |
| **CI regression gate** | Stop a PR from silently breaking retrieval | GitHub Actions: pytest + **retrieval-only eval** (no LLM calls, so free and deterministic); fail if Recall@5 drops > 3 points vs the committed baseline | ✅ |
| **Request tracing** | p50/p99, cost, route mix, abstention rate need raw data | SQLite `requests` table (one row per query, per-stage timings) | ✅ |
| **Langfuse** | Inspect *why* a specific answer went wrong (span tree, prompt, sources); attach judge scores; course's recommended tool | Cloud free tier; `@observe` decorators; async, never blocks requests | ✅ nice-to-have #1 |
| **User feedback** | Online quality signal (rubric: online eval) | 👍/👎 → SQLite (+ Langfuse score) | ✅ |
| **Online LLM-as-judge** | Thumbs are sparse and biased (~2% of users click) | Script that scores the last N logged answers for groundedness/correctness | ✅ (manual/daily script) |
| **Dev LLM cache** | Free-tier daily token caps make repeated eval runs impossible | Disk cache keyed by hash(model, prompt, params), **eval/dev only** | ✅ |
| **Deployment & rollback** | Rubric asks for strategy | Docker image tagged by git SHA; rollback = redeploy previous tag or flip prompt version/model in config | ✅ |
| Model registry | We train no models | — | ❌ |
| DVC | Eval set is small text, versioned in git | — | ❌ |
| Feature store, Kafka, Airflow, K8s | No streaming features, no scheduled pipelines, single node | — | ❌ |
| Evidently / drift platform | Drift signals are a few SQL queries on our own log | — | ❌ |
| Production answer cache | Low traffic; cache invalidation on re-upload adds bugs | — | ❌ |

### Deployment strategy (for the LLMOps-depth slide)
- **Model serving:** LLMs via hosted API (OpenAI-compatible, free tier); embeddings in-process on CPU; one container (FastAPI + worker + Streamlit) via docker-compose.
- **Rollout:** a new prompt/model/threshold must (1) pass CI, (2) beat or match the current version on the offline eval (logged in MLflow), then (3) be switched on via config. For a demo-scale system, a **shadow comparison** is enough: re-run last week's logged questions through the new version and compare judge scores. We don't need live A/B traffic.
- **Rollback:** revert `config.yaml` (prompt version, model) or redeploy the previous image tag. An embedding-model change requires a **re-index**, done as a new Chroma collection built alongside the old one and switched by config (blue-green, as in Week 6).

---

## 14. Evaluation

### Eval set (hand-written, domain-specific, committed as `eval/questions.jsonl`)
Corpus: **2 annual reports of Ellenbarrie Industrial Gases (EIG): FY25 and FY26**, plus a small third PDF for the live upload demo. Because the FY26 report also contains FY25 comparatives, gold answers must record which report *and* which year's figure is meant, and near-miss questions should use years or metrics neither report covers (e.g. FY2030).

| Slice | # | Purpose |
|---|---|---|
| Answerable document questions | 40 | Mix: ~15 numbers from tables, ~15 narrative facts, ~10 "explain" questions. Gold answer + gold page(s) |
| Unanswerable document questions | 12 | **Near-misses**: wrong year, a metric the report doesn't disclose, a different company. Gold = ABSTAIN |
| General questions | 18 | Including traps: definitions of terms that appear in the report ("What is working capital?") |
| Mixed questions | 15 | Gold route + both gold sub-answers |
| **Total** | **~85** | Every row carries `route` as a label → also the router test set |

Row format:
```json
{"id":"D07","question":"What was Ellenbarrie's consolidated revenue from operations in FY2025?",
 "route":"DOCUMENT","answerable":true,"gold_answer":"₹[x] crore","gold_pages":[{"doc":"eig_fy25","page":47}],
 "type":"numeric"}
```
*(Example values are illustrative; take the real ones from the report.)*

Split the work: each team member writes ~20 questions; a second member verifies each gold page.

### Metrics and scoring method
| Metric | Method | Type |
|---|---|---|
| Router accuracy + confusion matrix | Exact label match | Programmatic |
| Recall@5, MRR | Any retrieved chunk on a gold page | Programmatic, free; used in CI |
| Answer correctness | Numeric questions: normalised number match. Others: LLM-judge (0/1 vs gold answer) | Programmatic + judge |
| Groundedness / faithfulness | LLM-judge: "is every claim supported by the cited sources?" | Judge |
| Citation accuracy | Cited page ∈ gold pages | Programmatic |
| Abstention | Precision/recall of ABSTAIN on answerable vs unanswerable; false-answer rate | Programmatic |
| Number-check failure rate | From the validator | Programmatic |
| Latency p50/p95/p99 (per stage) | Request log | Programmatic |
| Ingestion time / page | 3 PDFs × 3 runs | Programmatic |
| Throughput | 40-line asyncio load script, 1/3/5 concurrent users; report req/min and where it breaks (expected: Groq 429s) | Programmatic |
| Failure rate | 5xx + timeouts / total | Programmatic |

**Judge calibration (cheap, high credibility):** two team members hand-label 20 answers; report agreement with the judge (% agree, or Cohen's κ). If agreement < 80%, fix the judge prompt before trusting its numbers.

**Why a custom judge prompt instead of RAGAS:** fewer dependencies, far fewer LLM calls per sample (which matters under free-tier caps), and every criterion can be explained in the viva. Mention RAGAS as the standard alternative.

### Planned experiments (each = one MLflow run)
1. Chunk size 250 / 400 / 600 tokens → Recall@5
2. Dense vs dense+BM25 (RRF) → Recall@5
3. Top-k 3 / 5 / 8 → correctness vs tokens
4. θ sweep → abstention trade-off curve (choose θ)
5. Router: rules vs score-threshold vs LLM → accuracy
6. (Optional) reranker on/off → correctness, latency

These produce the trade-off slide with real numbers.

---

## 15. Observability

**Source of truth:** SQLite `requests` table, one row per query:
`trace_id, ts, question_len, route, router_ok, prompt_versions, model_ids, t_router_ms, t_embed_ms, t_retrieve_ms, t_llm_ms, t_total_ms, top_score, n_sources, status (answered/abstained/not_ready/error), abstain_reason, citations_valid, number_check, tokens_in, tokens_out, cost_usd_equiv, error, feedback`.
Plus a `documents` table: `pages, chunks, ingest_seconds, status, error`.

**Streamlit Metrics page** (it reads the SQLite file directly) covers **all five monitoring categories** from Week 7. The rubric needs 3.

| Category | What we show |
|---|---|
| Operational | p50/p95/p99 total and per stage; error/timeout rate; tokens and cost-equivalent per request; ingestion seconds/page |
| Input | Question length distribution; route mix (DOC/GEN/MIX); upload failures by reason |
| Output | Abstention rate; citation-invalid rate; number-check failure rate; answer length |
| Quality | 👍/👎 rate; sampled judge scores (from the judge script) |
| Drift | Daily median **top retrieval score** (falling = questions moving away from the documents); route-mix shift week over week |

**Langfuse** (nice-to-have #1): span tree per request (router → retrieve → generate), prompts and sources viewable, judge scores attached. **Why both:** the SQLite log is ours, always on, works offline, and feeds the README numbers. Langfuse is for debugging individual bad answers. If time is short, SQLite alone satisfies the deliverable ("Streamlit dashboard … showing this data").

**Alerts:** not needed for a demo system. Show thresholds as coloured indicators on the dashboard instead (e.g. p95 > 6 s red; abstention rate > 40% amber).

**Logging hygiene:** never log uploaded document text in traces beyond chunk IDs and snippets; logging failures are caught and never fail a request.

---

## 16. Recommended Technology Stack

| Layer | Choice | Alternative (brief) |
|---|---|---|
| Frontend | **Streamlit** (chat, upload, status, metrics page) | Gradio; plain HTML |
| Backend | **FastAPI** (REST JSON; multipart upload; 202 + polling) | Flask |
| Async processing | **1 background thread + SQLite job table** | RQ + Redis (if scaling) |
| PDF parsing | **PyMuPDF** (AGPL: fine for a public open-source repo) | pypdf (BSD, weaker); Docling (best tables, heavy on CPU) |
| Chunking | **Own function** (page-aware recursive) | LangChain splitter (unneeded dependency) |
| Embeddings | **bge-small-en-v1.5 via fastembed** | bge-base (better, ~3× slower); e5-small |
| Vector store | **Chroma** (persistent, local) | Qdrant local mode; pgvector |
| Metadata/jobs/logs | **SQLite** | Postgres (unneeded) |
| LLM | **gpt-oss-120b (answers) + gpt-oss-20b (router) on Groq free plan**, via `openai` SDK | Gemini Flash free tier; Ollama local |
| Reranker | None (experiment: bge-reranker-base) | ms-marco-MiniLM |
| Orchestration framework | **None (plain Python)**, per Week 4 advice | LangChain/LlamaIndex (rejected) |
| Validation | **Pydantic** (router + answer JSON) | — |
| Evaluation | **Own eval runner** + Gemini Flash judge | RAGAS, DeepEval |
| Experiment tracking | **MLflow** (local) | W&B |
| Observability | **SQLite request log + Streamlit metrics**; **Langfuse Cloud** (free tier) | Arize Phoenix |
| Testing / CI | **pytest + GitHub Actions** (unit tests + retrieval-eval gate) | — |
| Containers | **Dockerfile + docker-compose** | — |
| Deployment | **Primary: local docker-compose for the demo.** Optional public URL on any free container host (live deployment is "good to have, not mandatory") | HF Space only if a teammate has PRO or an older free Space |

---

## 17. End-to-End Workflow

**A. Upload**
1. User drops `EIG_AR_FY25.pdf` → `POST /documents` → validate → hash → save → `documents(status=QUEUED)` → **202 {doc_id}**.
2. Worker: `PROCESSING` → parse 312 pages → chunk (~900) → embed in batches → upsert every 20 pages → `pages_done` updates → `READY (312 pages, 903 chunks, 74 s)`.
3. UI shows a progress bar, then a green "Ready".

**B. Question: "What was Ellenbarrie's revenue in FY2025 and who is the PM of India?"**
1. `POST /query` → trace_id created.
2. Router (gpt-oss-20b) with doc titles → `MIXED` + two sub-questions (~0.5 s).
3. In parallel:
   - **Doc path:** embed sub-question → Chroma top-8 over READY docs → top score 0.81 ≥ θ → top-5 wrapped as `<source id=S1 … page=47>` → gpt-oss-120b returns `{status: ANSWERED, answer, citations:[S1]}` → citation IDs valid → number check passes.
   - **General path:** gpt-oss-120b → answer + "general knowledge" label.
4. Compose two sections → response JSON → UI renders the answer, citation chips (click → snippet) and 👍/👎.
5. Row written to `requests` (timings per stage, tokens, route, status). Langfuse trace sent asynchronously.

**C. Offline**
`python -m eval.run --config config.yaml` → runs ~85 questions (cache on) → computes metrics → logs to MLflow → writes `eval/results/latest.json` → README table updated.

---

## 18. 15-Minute Demo

**Principle:** the risky parts (long ingestion, the network) are never on the critical path of the demo.

| Time | Segment | Content |
|---|---|---|
| 0:00–2:00 | Problem framing | Business objective sentence; the user pain; the ML problem (input/output/targets); scope in/out |
| 2:00–4:30 | Architecture | The one diagram (§7); sync vs async paths; 3 key decisions |
| 4:30–9:30 | **Live demo** | ① Upload a **small (~20-page) PDF** → show progress → Ready (~15 s). ② While it processes, ask a GENERAL question → answered instantly. ③ On the **pre-indexed annual report**: document question → answer + page citation → click to show the snippet. ④ MIXED question → two labelled sections. ⑤ **Abstention moment**: "What was Ellenbarrie's revenue in FY2030?" → refuses and shows the closest pages. ⑥ Quick look at the metrics page |
| 9:30–12:30 | LLMOps & evaluation | Eval set design; results table; router benchmark; θ trade-off curve; CI gate; prompt versioning; Langfuse trace of one request |
| 12:30–14:30 | Trade-offs & failure modes | 3–4 "we chose X over Y because Z" (with numbers); what breaks at 10× |
| 14:30–15:00 | Limitations / next steps | — |

Rotate speakers per segment (presentation quality = equitable sharing).

**Fallbacks (prepared, tested the day before):**
1. Annual reports are **pre-indexed** (a seed script builds the Chroma index at container start).
2. LLM backend switch via env var: Groq → **local Ollama** if the network or API fails (answers slower, same flow).
3. A **screen recording** of the full demo flow as a last resort.
4. Demo questions are already in the eval set, so we know their expected outputs.

**Prepare for Q&A** (peer-review questions are predictable): "What breaks first at 10×?" (Groq rate limits → 429s, then the single ingestion thread). "Most consequential decision?" (abstain-first evidence policy / routing before retrieval). "Why not hybrid search?" (show the measured Recall@5 delta). "How do you know the judge is right?" (calibration agreement).

---

## 19. Trade-offs

The rubric format, "We chose X over Y because Z", with the cost we accept:

1. **We chose dense-only retrieval over hybrid + reranking** because our corpus is a few reports and our measured Recall@5 is [X]. *Cost:* may miss exact-term matches; BM25 was measured as experiment 2 and adopted/rejected on its numbers.
2. **We chose an LLM router over rules or a trained classifier** because mixed questions must be *split*, and only an LLM does that reliably without labelled training data. *Cost:* ~0.5 s and ~500 tokens per query, plus a new failure mode (bad JSON), handled by the DOCUMENT fail-safe.
3. **We chose to abstain over answering at low evidence** because a wrong figure with a citation destroys trust faster than "I couldn't find it". *Cost:* some answerable questions get refused (wrong-abstention rate [X]%); θ was set from the measured trade-off curve.
4. **We chose full-document readiness over partial querying** because ingestion takes ~[X] s and partial results make abstentions ambiguous. *Cost:* users wait ~1–3 min before asking about a new document.
5. **We chose open-weight models on a free hosted API over local inference** because laptop CPUs give 10–40 s answers. *Cost:* internet dependency and rate limits. We mitigate with a local Ollama fallback and keep no lock-in (OpenAI-compatible API).
6. **We chose a thread + SQLite over Celery/Redis** because we have one user and a handful of documents. *Cost:* serial ingestion; this is the first component to replace at scale.
7. **We chose plain Python over LangChain/LlamaIndex** because every step must be debuggable and explainable in a viva. *Cost:* we write ~200 more lines ourselves.

### Limitations we accept consciously
No scanned PDFs · tables may extract imperfectly · single-turn only · general answers may be outdated (no web search) · single user, no auth · serial ingestion · free-tier rate limits cap throughput at a few queries/minute · English only.

---

## 20. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Table numbers extracted in the wrong order → wrong answers | High | High | Eval slice for table questions; number check; try PyMuPDF `find_tables` as nice-to-have; report the table-question accuracy separately and honestly |
| Groq free-tier limits hit during eval/demo | High | Medium | Dev cache; router and answers on separate model quotas; split eval runs; Ollama fallback |
| Free model list / limits change | Medium | Medium | Model IDs in config only; one OpenAI-compatible client; Gemini as alternative backend |
| θ badly calibrated → over- or under-abstaining | Medium | High | θ chosen from the eval sweep; logged in MLflow; LLM gate as second layer |
| Router misclassifies DOCUMENT as GENERAL → model answers a document question from memory | Medium | High | Few-shot hard examples; measure DOCUMENT recall; general prompt warns about exact figures; the label shows the source |
| Prompt injection inside a PDF | Low | Medium | `<source>` wrapping + "data not instructions" system rule; no tools that act |
| Ingestion slower than expected on the deployment CPU | Medium | Low | Pre-indexed seed; small demo PDF for live upload |
| Eval set takes longer than expected | Medium | High | Start it in week 1 in parallel with coding; ~20 questions per member |
| No free public hosting available | Medium | Low | Deployment is optional; local docker-compose demo + recording |
| Team scope creep (agents, hybrid, chat memory…) | High | High | §6 MVP list is a contract; anything else needs a measured reason |

---

## 21. Architecture Decisions

| # | Decision | Why | Alternative | Why not | Trade-off accepted |
|---|---|---|---|---|---|
| AD1 | Restrict to text-layer English annual reports | Makes chunking, evaluation and the demo concrete | Any PDF | Unbounded scope; OCR needed | Smaller audience |
| AD2 | Async ingestion via 1 thread + SQLite | Same UX as a queue, zero extra services | Celery/RQ + Redis | Extra infrastructure for 1 user | Serial processing; restart re-queue needed |
| AD3 | No partial querying in MVP | Ingestion is short; avoids ambiguous abstentions | Query completed batches | Complexity > value at this ingestion time | ~1–3 min wait per new doc |
| AD4 | LLM router with JSON + DOCUMENT fail-safe | Needs decomposition of MIXED | Rules / embeddings / classifier | Can't split; no training data | Latency + tokens per query |
| AD5 | Page-bounded ~400-token chunks with contextual header | Exact page citations; under the 512-token embed limit | Fixed 512 across pages; semantic chunking | Citation ambiguity; CPU cost | Some answers span two pages |
| AD6 | bge-small via fastembed | Fast CPU embeddings, small image | OpenAI embeddings; bge-base | Paid / slower | Slightly lower retrieval quality |
| AD7 | Chroma, dense-only | Simplest working store; course rule | Qdrant; hybrid BM25 | Not needed until measured | May miss exact-term matches |
| AD8 | Two-gate abstention + citation-ID validation | Score gate is cheap; LLM gate catches near-misses | Score threshold only | Fails on near-miss questions | Some false abstentions |
| AD9 | gpt-oss-120b/20b on Groq free plan | Open-weight, fast, free | Local Ollama; paid GPT-4o-mini | Too slow on CPU / not free | Network + rate-limit dependency |
| AD10 | Different-family judge (Gemini Flash) | Avoids self-preference bias | Same model as judge | Biased scores | Second API key |
| AD11 | SQLite request log as observability source of truth; Langfuse optional | Numbers we own; works offline | Langfuse only; Prometheus + Grafana | External dependency; overkill | Hand-built dashboard |
| AD12 | MLflow only for eval runs | ~15 configs to compare | Spreadsheet; W&B | Not reproducible / hosted | One more tool |
| AD13 | Retrieval-only eval as CI gate | Free, deterministic, catches silent retrieval regressions | Full LLM eval in CI | Costs quota; non-deterministic | Generation regressions caught only by manual eval |
| AD14 | Plain Python, no framework | Debuggable, explainable | LangChain / LlamaIndex | Hidden abstractions | More code |

---

## 22. Open Questions (team to answer before coding)

1. **Team size and deadline?** These set how much of the nice-to-have list is realistic.
2. **Which 2 annual reports?** *Decided:* Ellenbarrie Industrial Gases FY25 and FY26 (`EIG AR FY25.pdf`, `EIG AR FY26.pdf`). Still to check: do they have clean text layers (select text in a PDF viewer)?
3. **Is the instructor fine with a free hosted API (Groq) for open-weight models?** Or must inference be fully local? Fully local changes the latency targets and the demo plan.
4. **Hardware:** does anyone have a GPU, Apple Silicon or a 16 GB+ RAM laptop? This decides the Ollama fallback model.
5. **Public deployment:** do we want it (good-to-have)? Does anyone have an existing HF account with PRO or a free Space, or another free host?
6. **Policy confirmation:** is "general answers are labelled and may be outdated" acceptable, or does the team want web search? (Recommendation: no, it's future work.)
7. **Mixed answer format:** confirm two labelled sections rather than one blended paragraph.
8. Does anyone need conversation memory for their demo story? (Recommendation: no.)

---

## 23. Recommended Implementation Order

Each step ends with something runnable. The eval set starts **on day 1** in parallel.

| Step | Build | Done when |
|---|---|---|
| 0 | Repo skeleton, `config.yaml`, `prompts/`, Dockerfile, CI running pytest | `docker compose up` shows an empty Streamlit page talking to FastAPI |
| 1 | **Eval set v1** (parallel track): pick reports, write ~85 labelled questions | `eval/questions.jsonl` committed, gold pages verified by a second person |
| 2 | Ingestion: validate → parse → chunk → embed → Chroma; SQLite status; worker thread; re-queue on start; embedding-version guard | Upload a report, watch it reach READY, chunk counts sensible |
| 3 | Retrieval + **retrieval-only eval** (Recall@5, MRR) + MLflow logging; **CI gate** | Baseline Recall@5 number exists and is in the README |
| 4 | Document answer path: prompt v1, JSON output, citation mapping, two-gate abstention, number check | Doc questions answered with citations; near-miss questions abstain |
| 5 | Router (+ fail-safe), general path, MIXED composition, all §11 messages | All 4 routes work in the UI |
| 6 | Request log + Streamlit metrics page + feedback buttons | p50/p95/p99 visible after running the eval |
| 7 | Full eval runner: router benchmark (3 variants), correctness, judge, abstention, latency; judge calibration on 20 labels | Results table + confusion matrix generated by one command |
| 8 | Experiments (chunk size, BM25, top-k, θ sweep) → choose final config | Trade-off numbers for slides |
| 9 | Langfuse tracing (nice-to-have #1); load test script; Ollama fallback switch | Trace screenshot; throughput number |
| 10 | README (problem, architecture, numbers, setup), resume line, seed script, demo recording, slides, rehearsal ×2 | Full 15-min run under time, fallback tested |

**Only after step 10:** further nice-to-haves (tables, partial querying, reranker), each with a measured reason.

---

## Final Gate Before Implementation

**Product: what exactly are we building?**
A single-user web app for asking questions about uploaded annual-report PDFs. It answers with page citations or refuses, and it answers general questions with a clear label.

**User: what can the user do?**
Upload text-based PDFs, watch their processing status, ask document, general or mixed questions, click citations to see source snippets, and give thumbs feedback.

**Data: where does data enter and how does it move?**
PDF → FastAPI (validate, save) → SQLite job → worker (PyMuPDF → chunks → bge-small vectors) → Chroma. Question → router → Chroma (READY docs) → LLM → validator → response → SQLite log (+ Langfuse). The eval set lives in git; eval results go to MLflow.

**ML: what are the actual ML/LLM components?**
(1) LLM router: 3-class classification + decomposition. (2) Embedding model: dense retrieval/ranking. (3) Abstention decision: calibrated score threshold + LLM sufficiency judgement. (4) LLM grounded generation and general answering. (5) LLM-as-judge for evaluation.

**Retrieval: how does it find relevant information?**
Page-bounded ~400-token chunks with a document/page header, embedded with bge-small. Cosine top-8 from Chroma over READY documents; top-5 go to the LLM as tagged sources. BM25/reranker only if experiments justify them.

**Routing: how does it tell document, general and mixed apart?**
One gpt-oss-20b call with the uploaded document titles and few-shot examples returns validated JSON `{route, document_question, general_question}`. Invalid output → DOCUMENT (fail-safe). It is benchmarked against rules and a retrieval-score baseline.

**Ingestion: what happens while a document is processing?**
The status and page progress are visible. General questions work normally. Document questions about it get a "still processing (n/N pages)" response. Other READY documents stay queryable.

**Reliability: when does it answer, and when does it abstain?**
It answers only if the top score ≥ θ **and** the LLM reports sufficient evidence **and** the citations are valid. Otherwise it abstains with a reason and the closest pages. LLM failure → show the top passages without a summary.

**LLMOps: which concepts are actually demonstrated?**
Prompt/config versioning, embedding-version guard, offline eval with a hand-written set, experiment tracking (MLflow), CI regression gate, per-request tracing of latency/tokens/cost, monitoring across all 5 categories, online feedback + sampled LLM-as-judge with calibration, rollout via eval gate and rollback via config or image tag.

**Evaluation: how do we know it works?**
~85 labelled questions. Router accuracy, Recall@5/MRR, answer correctness, citation accuracy, abstention precision/recall, number-check rate, judge agreement with humans, latency p50/p95/p99, ingestion s/page, load-test throughput. All reproducible by one command and reported in the README.

**Technology: what exact tools/models?**
Streamlit, FastAPI, PyMuPDF, fastembed + bge-small-en-v1.5, Chroma, SQLite, gpt-oss-120b + gpt-oss-20b on the Groq free plan (OpenAI SDK), Ollama fallback, Pydantic, Gemini Flash judge, MLflow, Langfuse (optional), pytest + GitHub Actions, Docker Compose.

**Scope: what are we deliberately NOT building?**
Agents, frameworks (LangChain/LlamaIndex), fine-tuning, OCR, hybrid/rerank by default, query rewriting, chat memory, web search, multi-user auth, Kafka/Redis/K8s, model registry, feature store, production answer cache, partial querying (MVP).

**Trade-offs: what limitations are we consciously accepting?**
Imperfect table extraction, possibly outdated general answers, single-turn only, serial ingestion, internet/rate-limit dependency, some false abstentions in exchange for very few false answers.

**Implementation: what should the implementation agent eventually build?**
Exactly the §6 MVP list, in the §23 order, against the §5 targets. Steps 1 (eval set) and 2 (ingestion) start in parallel, and nothing from the nice-to-have list is built until the MVP numbers are in the README.
