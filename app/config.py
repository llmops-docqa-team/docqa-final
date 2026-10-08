"""Typed settings: config.yaml for tunables, env vars (by name) for secrets and overrides."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent


class OllamaConfig(BaseModel):
    """A local Ollama server (OpenAI-compatible). Model names are Ollama tags: `ollama pull gpt-oss:20b`
    (16 GB+ laptop) or `ollama pull llama3.2:3b` (weaker machine; then set both models below to it)."""

    base_url: str = "http://localhost:11434/v1"   # from a container: http://host.docker.internal:11434/v1
    answer_model: str = "gpt-oss:20b"
    router_model: str = "gpt-oss:20b"
    timeout_seconds: float = 120                  # a local model on CPU is slow, and the first call loads it


class LLMConfig(BaseModel):
    # Primary backend: groq (base_url/models below) or ollama (the `ollama` block). Env LLM_BACKEND overrides.
    backend: Literal["groq", "ollama"] = "groq"
    ollama: OllamaConfig = OllamaConfig()
    # Automatic fallback: when the groq backend times out or stays rate-limited (after its retries), the call
    # is repeated on Ollama and the primary is skipped for `fallback_cooldown_seconds`. Off unless this flag
    # or the OLLAMA_BASE_URL env var turns it on.
    fallback_enabled: bool = False
    fallback_cooldown_seconds: float = 30
    base_url: str
    answer_model: str
    router_model: str
    rewrite_model: str = ""  # query enhancer's rewrite call; "" = router_model
    reasoning_effort: str = "low"
    timeout_seconds: float = 20
    judge_model: str = ""
    # The judge is a different model family (Gemini) behind its OpenAI-compatible endpoint.
    # The key is GEMINI_API_KEY.
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # Gemini's hidden "thinking" tokens count against max_tokens and can cut the JSON verdict off mid-way
    # (finish_reason "length"), so the judge runs with thinking off.
    judge_reasoning_effort: str = "none"
    max_retries: int = 2                  # extra attempts on 429 / 5xx / timeout / connection errors
    retry_backoff_seconds: float = 0.5    # doubles each retry
    retry_max_wait_seconds: float = 10.0  # a Retry-After longer than this fails fast instead of waiting
    # Dev/eval only: disk cache of responses keyed by hash(model, messages, params). Off unless the
    # FINCHAT_LLM_CACHE env var (or this flag) turns it on; FINCHAT_ENV=prod forces it off.
    dev_cache: bool = False
    cache_dir: str = "data/llm_cache"


class PromptsConfig(BaseModel):
    router: str
    answer_doc: str
    answer_general: str
    judge: str
    rewrite: str = "rewrite_v1"


class EmbeddingConfig(BaseModel):
    # "model2vec": static embeddings, ~2,000x faster on CPU. "fastembed": ONNX transformer (bge-small).
    # Changing backend or model needs a re-index (the API refuses to start on a mismatched index).
    backend: Literal["model2vec", "fastembed"] = "fastembed"
    model: str
    batch_size: int = 32
    # Put in front of every query (not passages). bge needs it; set "" for models that do not.
    query_prefix: str = "Represent this sentence for searching relevant passages: "


class ChunkingConfig(BaseModel):
    size_tokens: int = 400
    overlap_tokens: int = 60


class TopicRule(BaseModel):
    when: list[str]       # words in the question (case-insensitive, matched from the start of a word)
    headings: list[str]   # a chunk containing one of these (case-insensitive) is favoured


class TopicBoostConfig(BaseModel):
    """Hybrid only. After fusion, a chunk whose text holds a heading that fits the question's topic gets
    `bonus` x the best fused score added, so the Financial Highlights table can overtake note prose for a
    "why did finance cost fall" question. Only the best `depth` fused candidates are looked at. The class
    default is off; config.yaml turns it on."""

    enabled: bool = False
    bonus: float = 0.15
    depth: int = 30
    topics: list[TopicRule] = []


class RetrievalConfig(BaseModel):
    fetch_k: int = 8
    top_k: int = 5
    theta: float = 0.5
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"  # hybrid = dense + BM25 fused with weighted RRF
    bm25_weight: float = 3.0  # weight of the BM25 list in the fusion (dense = 1.0)
    pool: int = 50  # how deep each ranking is read before fusing
    rrf_k: int = 60
    # Hybrid only: keyword search's best N chunks always land inside the `top_k` the model sees. Fusion
    # rewards chunks both lists find, so an exact-figure table chunk that keyword search ranks #1 can lose
    # to prose that both lists find mildly when the dense search ranks the table outside its pool. 0 = off.
    bm25_keep_top: int = 1
    # A table split into pieces: when one piece reaches the model, the rest of that table comes with it
    # (small-to-big), within answer.context_max_tokens. False = off.
    table_siblings: bool = True
    topic_boost: TopicBoostConfig = TopicBoostConfig()


class AnswerConfig(BaseModel):
    max_tokens: int = 1500            # completion budget; gpt-oss reasoning tokens count against it
    snippet_chars: int = 300          # citation snippet length
    closest_pages: int = 3            # "closest sections" shown on an abstention
    fallback_passages: int = 3        # passages shown when the LLM is unavailable
    context_max_tokens: int = 3500    # cap on the passages sent to the model (Groq free tier: 8,000 TPM)


class QueryConfig(BaseModel):
    max_question_chars: int = 500     # longer questions get a 422
    router_max_tokens: int = 600      # completion budget; gpt-oss reasoning tokens count too
    general_max_tokens: int = 1000
    router_max_docs: int = 30         # document titles listed to the router (the rest are summarised)
    doc_title_chars: int = 80         # a longer file name is cut when shown to the router
    enhance: bool = True              # query enhancer: company/period scoping + abbreviation expansion
    rewrite: bool = True              # ... plus one small LLM call that fixes spelling and names the company
    rewrite_max_tokens: int = 400     # gpt-oss reasoning tokens count too
    require_company: bool = True      # several companies loaded: a document question must name or pick one


class TracingConfig(BaseModel):
    """Langfuse tracing. It is on only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set."""

    # Off: a trace holds ids, versions, models, token counts, scores and timings only. On: it also holds
    # the question, the answer and the cited snippets (never whole chunks); that text goes to Langfuse.
    capture_content: bool = False


class ApiConfig(BaseModel):
    # /debug/* endpoints (no auth, and /debug/answer_doc spends LLM quota). Off unless this flag or the
    # FINCHAT_DEBUG env var turns them on.
    debug_endpoints: bool = False


class Rate(BaseModel):
    """List-price-equivalent USD per million tokens (we run on a free tier; this is what it would cost)."""

    input: float
    output: float


class Threshold(BaseModel):
    """Colour rule for a dashboard indicator. `lower_is_worse` flips it (e.g. retrieval score)."""

    amber: float | None = None
    red: float | None = None
    lower_is_worse: bool = False


def _default_thresholds() -> dict[str, Threshold]:
    return {
        "p95_total_ms": Threshold(amber=4000, red=6000),
        "error_rate": Threshold(amber=0.02, red=0.05),
        "abstention_rate": Threshold(amber=0.40, red=0.70),
        "citation_invalid_rate": Threshold(amber=0.05, red=0.15),
        "number_check_fail_rate": Threshold(amber=0.10, red=0.25),
        "router_fallback_rate": Threshold(amber=0.02, red=0.10),
        "thumbs_down_share": Threshold(amber=0.20, red=0.40),
        "median_top_score": Threshold(amber=0.55, red=0.45, lower_is_worse=True),
        "route_mix_shift": Threshold(amber=0.20, red=0.40),
    }


def _default_pricing() -> dict[str, Rate]:
    return {
        "openai/gpt-oss-120b": Rate(input=0.15, output=0.60),
        "openai/gpt-oss-20b": Rate(input=0.075, output=0.30),
        "default": Rate(input=0.15, output=0.60),
    }


class ObservabilityConfig(BaseModel):
    pricing_per_mtok: dict[str, Rate] = _default_pricing()   # model id -> rates; "default" for unknown ids
    thresholds: dict[str, Threshold] = _default_thresholds()
    min_requests_for_indicator: int = 5   # fewer requests than this: show the number, no colour
    # Opt-in: also store each question, the answers and the cited source text in `request_content`
    # (a separate table), so scripts/judge_recent.py has something to judge. Off by default.
    log_content: bool = False


class EvalConfig(BaseModel):
    """Pacing and limits for `python -m eval.run` (free-tier quotas are the constraint, not CPU)."""

    tpm_budget: int = 6500                 # tokens/min per model the runner allows (Groq free cap: 8,000)
    judge_min_interval_seconds: float = 6.5   # between live judge calls (Gemini free tier: ~10 requests/min)
    judge_max_tokens: int = 1500
    # Gemini's free tier answers 429 without a Retry-After once the per-minute limit is hit; waiting this long
    # (then trying again, up to the retries below) is what gets through, not the client's 0.5 s backoff.
    judge_rate_limit_wait_seconds: float = 30.0
    judge_rate_limit_retries: int = 3
    judge_source_chars: int = 3000         # each cited source is cut to this many characters for the judge
    # Live judge calls allowed per `eval.run` invocation (Gemini's free tier gives ~20 requests a day, so
    # the testing phase judges only a few answers). Cached replays are free; 0 = no limit.
    # `--judge-limit` overrides.
    judge_max_calls: int = 10


class UploadConfig(BaseModel):
    max_mb: int = 25
    max_pages: int = 400


class IngestionConfig(BaseModel):
    upsert_every_pages: int = 20
    max_failed_page_ratio: float = 0.5   # FAILED if more than this fraction of pages could not be read


class OCRConfig(BaseModel):
    min_chars: int = 120
    min_image_area_ratio: float = 0.6
    dpi: int = 200
    language: str = "eng"


class ParsingConfig(BaseModel):
    repeat_line_ratio: float = 0.5   # edge line on > this fraction of text pages = header/footer
    repeat_min_pages: int = 3        # skip header/footer stripping on docs shorter than this
    repeat_edge_lines: int = 3       # only the first/last N lines of a page are candidates
    table_min_rows: int = 2          # header + at least one body row
    table_min_cols: int = 2
    # Text pages are parsed in parallel worker processes (PyMuPDF is not thread-safe; processes are its
    # recommended route). 0 = one per CPU core minus one (at most 8); 1 = no worker processes.
    workers: int = 0
    parallel_min_pages: int = 24     # smaller documents are parsed in-process (start-up not worth it)
    parallel_batch_pages: int = 8    # pages per task; results still come back in page order


class PathsConfig(BaseModel):
    data_dir: str = "data"
    sqlite_path: str = "data/finchat.sqlite"
    upload_dir: str = "data/uploads"
    chroma_dir: str = "data/chroma"
    model_cache_dir: str = "data/models"


class Settings(BaseModel):
    llm: LLMConfig
    prompts: PromptsConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    answer: AnswerConfig = AnswerConfig()
    query: QueryConfig = QueryConfig()
    api: ApiConfig = ApiConfig()
    tracing: TracingConfig = TracingConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    eval: EvalConfig = EvalConfig()
    upload: UploadConfig
    ingestion: IngestionConfig
    ocr: OCRConfig
    parsing: ParsingConfig = ParsingConfig()
    paths: PathsConfig

    # Secrets: read from env by name only, never stored in config.yaml.
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    @property
    def sqlite_path(self) -> Path:
        return self.resolve(self.paths.sqlite_path)

    @property
    def upload_dir(self) -> Path:
        return self.resolve(self.paths.upload_dir)

    @property
    def chroma_dir(self) -> Path:
        return self.resolve(self.paths.chroma_dir)

    @property
    def model_cache_dir(self) -> Path:
        return self.resolve(self.paths.model_cache_dir)

    @property
    def llm_cache_dir(self) -> Path:
        return self.resolve(self.llm.cache_dir)


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    cfg_path = Path(path or os.environ.get("FINCHAT_CONFIG") or ROOT / "config.yaml")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    settings = Settings(**data)
    llm = settings.llm
    if os.environ.get("OLLAMA_BASE_URL"):
        llm.ollama.base_url = os.environ["OLLAMA_BASE_URL"]
        llm.fallback_enabled = True  # setting the variable is how the fallback is switched on
    backend = (os.environ.get("LLM_BACKEND") or llm.backend).strip().lower()
    if backend not in ("groq", "ollama"):
        raise ValueError(f"LLM_BACKEND must be 'groq' or 'ollama', not {backend!r}")
    llm.backend = backend
    if backend == "ollama":  # the local server becomes the primary; there is nothing left to fall back to
        llm.base_url = llm.ollama.base_url
        llm.answer_model = llm.ollama.answer_model
        llm.router_model = llm.ollama.router_model
        llm.rewrite_model = llm.ollama.router_model
        llm.timeout_seconds = llm.ollama.timeout_seconds
        llm.fallback_enabled = False
    if os.environ.get("LLM_BASE_URL"):
        settings.llm.base_url = os.environ["LLM_BASE_URL"]
    if os.environ.get("LLM_MODEL"):
        settings.llm.answer_model = os.environ["LLM_MODEL"]
    if os.environ.get("FINCHAT_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        settings.api.debug_endpoints = True
    settings.groq_api_key = os.environ.get("GROQ_API_KEY") or None
    settings.gemini_api_key = os.environ.get("GEMINI_API_KEY") or None
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()