"""SQLite init (create-if-not-exists) and connection helper."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    sha256         TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'QUEUED',
    pages_total    INTEGER,
    pages_done     INTEGER NOT NULL DEFAULT 0,
    chunks         INTEGER NOT NULL DEFAULT 0,
    ingest_seconds REAL,
    error          TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);

CREATE TABLE IF NOT EXISTS requests (
    trace_id         TEXT PRIMARY KEY,
    ts               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    question_len     INTEGER,
    route            TEXT,
    router_ok        INTEGER,
    prompt_versions  TEXT,
    model_ids        TEXT,
    t_router_ms      REAL,
    t_embed_ms       REAL,
    t_retrieve_ms    REAL,
    t_llm_ms         REAL,
    t_total_ms       REAL,
    top_score        REAL,
    n_sources        INTEGER,
    status           TEXT,
    abstain_reason   TEXT,
    citations_valid  INTEGER,
    number_check     TEXT,
    tokens_in        INTEGER,
    tokens_out       INTEGER,
    cost_usd_equiv   REAL,
    error            TEXT,
    feedback         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);

-- Uploads refused before a document row exists (not a PDF, too big, unreadable, ...). Reasons only: no names.
CREATE TABLE IF NOT EXISTS upload_failures (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    status_code  INTEGER,
    reason       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_upload_failures_ts ON upload_failures(ts);

-- Opt-in (observability.log_content): the text of a request, kept apart from `requests` so that table stays
-- content-free and this one can be purged on its own. Read only by scripts/judge_recent.py.
CREATE TABLE IF NOT EXISTS request_content (
    trace_id  TEXT PRIMARY KEY,
    ts        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    question  TEXT NOT NULL,
    sections  TEXT NOT NULL          -- JSON list: {kind, status, question, answer, sources[]}
);
"""


# Columns added after step 00. Applied to existing databases by init_db (create-if-not-exists only
# covers new tables), so a data/docqa.sqlite from an earlier step keeps working.
ADDED_DOCUMENT_COLUMNS = {
    "embedding_model": "TEXT",
    "ingest_version": "INTEGER",
    "n_text_pages": "INTEGER",
    "n_table_pages": "INTEGER",   # pages with >= 1 table (a subset of the text pages)
    "n_ocr_pages": "INTEGER",
    "n_failed_pages": "INTEGER",
    "embed_seconds": "REAL",
    "page_timings": "TEXT",       # JSON list: parse+chunk seconds per page
    "company": "TEXT",            # catalog: guessed from the file name on upload, editable (PATCH)
    "report_type": "TEXT",        # Annual report | Quarterly results | Investor presentation | ...
    "period": "TEXT",             # canonical period: FY26 | Q1 FY26
}

# Step 08 additions to `requests` (same migration trick as above).
ADDED_REQUEST_COLUMNS = {
    "app_version": "TEXT",         # git SHA (or DOCQA_GIT_SHA) of the running code
    "router_fallback": "TEXT",     # why the router fell back to DOCUMENT (bad_json | llm_unavailable)
    "answer_chars": "INTEGER",     # length of the answer text(s) shown; 0 when nothing was answered
    "cited_chunks": "TEXT",        # JSON list of chunk ids cited (ids only, never document text)
    "judge_correct": "INTEGER",    # 0/1, written by the step 09 judge script (NULL = not judged)
    "judge_grounded": "INTEGER",   # 0/1, same
    "llm_backend": "TEXT",         # step 11: backend(s) that served the calls: groq | ollama | groq+ollama
    "llm_fallbacks": "INTEGER",    # LLM calls that fell back from Groq to Ollama
    "llm_rate_limited": "INTEGER", # 429 responses seen from the LLM backend while serving the request
}


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | Path) -> None:
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        have = {r["name"] for r in conn.execute("PRAGMA table_info(documents)")}
        for name, ddl in ADDED_DOCUMENT_COLUMNS.items():
            if name not in have:
                conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {ddl}")
        have = {r["name"] for r in conn.execute("PRAGMA table_info(requests)")}
        for name, ddl in ADDED_REQUEST_COLUMNS.items():
            if name not in have:
                conn.execute(f"ALTER TABLE requests ADD COLUMN {name} {ddl}")
        conn.commit()
    finally:
        conn.close()
