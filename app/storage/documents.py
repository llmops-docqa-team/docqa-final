"""`documents` table access. One short-lived connection per call; API threads and worker never share one."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.storage.db import connect

QUEUED, PROCESSING, PARTIAL, READY, FAILED = "QUEUED", "PROCESSING", "PARTIAL", "READY", "FAILED"
UNFINISHED = (QUEUED, PROCESSING, PARTIAL)
QUERYABLE = (PARTIAL, READY)

_UPDATABLE = {
    "status", "pages_total", "pages_done", "chunks", "ingest_seconds", "error", "embedding_model",
    "ingest_version", "n_text_pages", "n_table_pages", "n_ocr_pages", "n_failed_pages", "embed_seconds",
    "page_timings", "company", "report_type", "period",
}


class DocumentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _exec(self, sql: str, args: tuple = ()) -> tuple[list[dict[str, Any]], int]:
        conn = connect(self.path)
        try:
            cur = conn.execute(sql, args)
            rows = [dict(r) for r in cur.fetchall()]
            conn.commit()
            return rows, cur.rowcount
        finally:
            conn.close()

    # ---- reads ----------------------------------------------------------------------------------
    def _query(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        return self._exec(sql, args)[0]

    def get(self, doc_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return _decode(rows[0]) if rows else None

    def get_by_sha(self, sha256: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM documents WHERE sha256 = ? ORDER BY created_at LIMIT 1", (sha256,))
        return _decode(rows[0]) if rows else None

    def list(self) -> list[dict[str, Any]]:
        return [_decode(r) for r in self._query("SELECT * FROM documents ORDER BY created_at, id")]

    def unfinished_ids(self) -> list[str]:
        marks = ",".join("?" * len(UNFINISHED))
        rows = self._query(
            f"SELECT id FROM documents WHERE status IN ({marks}) ORDER BY created_at, id", UNFINISHED
        )
        return [r["id"] for r in rows]

    # ---- writes ---------------------------------------------------------------------------------
    def insert(self, doc_id: str, filename: str, sha256: str, pages_total: int) -> bool:
        """False if the id already exists (two uploads of the same file racing each other)."""
        from app.catalog import parse_filename  # app.catalog imports this module for the status names

        meta = parse_filename(filename)
        try:
            self._exec(
                "INSERT INTO documents"
                " (id, filename, sha256, status, pages_total, company, report_type, period)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, filename, sha256, QUEUED, pages_total, meta.company, meta.report_type, meta.period),
            )
        except sqlite3.IntegrityError:
            return False
        return True

    def update(self, doc_id: str, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"unknown document fields: {sorted(unknown)}")
        if isinstance(fields.get("page_timings"), list):
            fields["page_timings"] = json.dumps(fields["page_timings"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(
            f"UPDATE documents SET {cols}, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
            (*fields.values(), doc_id),
        )

    def reset_for_processing(self, doc_id: str, status: str) -> None:
        """Back to a clean slate (new run, crash recovery or retry of a FAILED upload)."""
        self.update(
            doc_id, status=status, pages_done=0, chunks=0, error=None, ingest_seconds=None,
            embed_seconds=None, n_text_pages=None, n_table_pages=None, n_ocr_pages=None,
            n_failed_pages=None, page_timings=None, embedding_model=None, ingest_version=None,
        )

    def delete(self, doc_id: str) -> bool:
        return self._exec("DELETE FROM documents WHERE id = ?", (doc_id,))[1] > 0


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("page_timings"):
        row["page_timings"] = json.loads(row["page_timings"])
    return row
