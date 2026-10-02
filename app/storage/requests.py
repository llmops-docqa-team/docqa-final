"""`requests` table access (one row per /query, keyed by trace_id): the request log and 👍/👎 feedback.

Two writers share a row and must not overwrite each other: `log` (at /query time) never touches `feedback`,
and `set_feedback` only touches `feedback`. Either may come first."""

from __future__ import annotations

import json
from pathlib import Path

from app.observability.request_log import LOG_COLUMNS
from app.storage.db import connect


def _flag(v: bool | None) -> int | None:
    return None if v is None else int(v)


class RequestStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def log(self, record: dict) -> None:
        """Upsert the request's log row, leaving `feedback` alone. Raises on a bad record or a database
        error: the caller decides that logging must never fail a request."""
        cols = [c for c in LOG_COLUMNS if c in record]
        unknown = set(record) - set(LOG_COLUMNS)
        if unknown or "trace_id" not in cols:
            raise ValueError(f"bad request record (unknown columns: {sorted(unknown)})")
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "trace_id")
        sql = (
            f"INSERT INTO requests ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "
            f"ON CONFLICT(trace_id) DO UPDATE SET {updates}"
        )
        conn = connect(self.path)
        try:
            conn.execute(sql, [record[c] for c in cols])
            conn.commit()
        finally:
            conn.close()

    def log_upload_failure(self, status_code: int | None, reason: str) -> None:
        conn = connect(self.path)
        try:
            conn.execute(
                "INSERT INTO upload_failures (status_code, reason) VALUES (?, ?)", (status_code, reason)
            )
            conn.commit()
        finally:
            conn.close()

    def set_feedback(self, trace_id: str, value: int) -> None:
        """Store 👍 (1) / 👎 (-1) on the request's row. Upsert: a trace id with no logged row (say, a request
        from before the log existed) gets a row holding only the feedback; the metrics ignore such rows."""
        conn = connect(self.path)
        try:
            conn.execute(
                "INSERT INTO requests (trace_id, feedback) VALUES (?, ?) "
                "ON CONFLICT(trace_id) DO UPDATE SET feedback = excluded.feedback",
                (trace_id, value),
            )
            conn.commit()
        finally:
            conn.close()

    def log_content(self, trace_id: str, question: str, sections: list[dict]) -> None:
        """Keep the text of a request (opt-in, `observability.log_content`). Raises on a database error."""
        conn = connect(self.path)
        try:
            conn.execute(
                "INSERT INTO request_content (trace_id, question, sections) VALUES (?, ?, ?) "
                "ON CONFLICT(trace_id) DO UPDATE SET "
                "question = excluded.question, sections = excluded.sections",
                (trace_id, question, json.dumps(sections, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    def recent_answered(self, limit: int, *, only_unjudged: bool = True) -> list[dict]:
        """The last `limit` requests whose status is `answered`, newest first, each with `question` and
        `sections` (parsed) or None for both when no content was stored for it."""
        where = "r.status = 'answered'" + (" AND r.judge_correct IS NULL" if only_unjudged else "")
        conn = connect(self.path)
        try:
            rows = conn.execute(
                "SELECT r.trace_id, r.ts, c.question, c.sections FROM requests r "
                "LEFT JOIN request_content c ON c.trace_id = r.trace_id "
                f"WHERE {where} ORDER BY r.ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = dict(r)
            d["sections"] = json.loads(d["sections"]) if d["sections"] else None
            out.append(d)
        return out

    def set_judge(self, trace_id: str, correct: bool | None, grounded: bool | None) -> None:
        """Store the judge's 0/1 scores. Only touches the two judge columns, and only on a row that exists."""
        conn = connect(self.path)
        try:
            conn.execute(
                "UPDATE requests SET judge_correct = ?, judge_grounded = ? WHERE trace_id = ?",
                (_flag(correct), _flag(grounded), trace_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, trace_id: str) -> dict | None:
        conn = connect(self.path)
        try:
            row = conn.execute("SELECT * FROM requests WHERE trace_id = ?", (trace_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
