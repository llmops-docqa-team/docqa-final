"""Thin client for the FinChat API. Every failure becomes an `ApiError` carrying a message safe to show
to a user (no stack traces, no URLs of internals beyond the base URL)."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_URL = "http://localhost:8000"
QUERY_TIMEOUT = 90  # router + answer, plus up to two rate-limit waits
UPLOAD_TIMEOUT = 120
QUICK_TIMEOUT = 8


class ApiError(Exception):
    """`str(error)` is the user-facing message. `status` is the HTTP status (None: API unreachable)."""

    def __init__(self, message: str, status: int | None = None, *, unreachable: bool = False):
        super().__init__(message)
        self.status = status
        self.unreachable = unreachable


def error_message(status: int, body: Any) -> str:
    """The message for an HTTP error. 4xx carry a readable `detail` from the API; 5xx never show internals."""
    if status >= 500:
        return "The FinChat service ran into a problem. Please try again in a moment."
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if status == 422:
        return "The request was not accepted. Please check the input and try again."
    if status == 404:
        return "That item was not found."
    return f"The request was rejected (HTTP {status})."


class ApiClient:
    def __init__(self, base_url: str | None = None, session: requests.Session | None = None):
        self.base_url = (base_url or os.environ.get("FINCHAT_API_URL") or DEFAULT_URL).rstrip("/")
        self.session = session or requests.Session()

    def _call(self, method: str, path: str, *, timeout: float, raw: bool = False, **kwargs: Any) -> Any:
        """The decoded JSON body; with `raw=True` the response itself (for an image)."""
        try:
            resp = self.session.request(method, self.base_url + path, timeout=timeout, **kwargs)
        except requests.Timeout as exc:
            raise ApiError("The service is taking too long to respond. Please try again.") from exc
        except requests.ConnectionError as exc:
            raise ApiError(
                f"Can't reach the FinChat service at {self.base_url}. Is it running?", unreachable=True
            ) from exc
        except requests.RequestException as exc:
            raise ApiError("Something went wrong talking to the FinChat service.") from exc
        if raw and resp.status_code < 400:
            return resp
        try:
            body = resp.json()
        except ValueError:
            body = None
        if resp.status_code >= 400:
            raise ApiError(error_message(resp.status_code, body), resp.status_code)
        return body

    def health(self) -> bool:
        try:
            self._call("GET", "/health", timeout=3)
        except ApiError:
            return False
        return True

    def list_documents(self) -> list[dict]:
        return self._call("GET", "/documents", timeout=QUICK_TIMEOUT) or []

    def upload(self, filename: str, data: bytes) -> dict:
        """Returns the API's UploadResponse (`duplicate: true` when the same file was already uploaded)."""
        return self._call(
            "POST",
            "/documents",
            timeout=UPLOAD_TIMEOUT,
            files={"file": (filename, data, "application/pdf")},
        )

    def delete_document(self, doc_id: str) -> None:
        self._call("DELETE", f"/documents/{doc_id}", timeout=QUICK_TIMEOUT)

    def ask(self, question: str, company: str | None = None, enhance: bool = True) -> dict:
        """`company` limits the document search to one catalog company (None: every document);
        `enhance=False` searches the question exactly as typed."""
        body: dict[str, Any] = {"question": question}
        if company:
            body["company"] = company
        if not enhance:
            body["enhance"] = False
        return self._call("POST", "/query", timeout=QUERY_TIMEOUT, json=body)

    def page_highlight(
        self, doc_id: str, pdf_page: int, terms: list[str], context: str = ""
    ) -> tuple[bytes, int]:
        """The cited PDF page as a PNG with the rows holding `terms` highlighted, and how many figures were
        found on it (0: the plain page)."""
        params: dict[str, Any] = {"term": terms}
        if context:
            params["ctx"] = context
        resp = self._call(
            "GET",
            f"/documents/{doc_id}/pages/{int(pdf_page)}/highlight",
            timeout=QUICK_TIMEOUT,
            raw=True,
            params=params,
        )
        try:
            matches = int(resp.headers.get("X-Highlight-Matches", 0))
        except ValueError:
            matches = 0
        return resp.content, matches

    def catalog(self) -> dict:
        """Companies, report types and periods of the uploaded documents (GET /catalog)."""
        return self._call("GET", "/catalog", timeout=QUICK_TIMEOUT) or {"companies": []}

    def update_document(self, doc_id: str, **fields: str) -> dict:
        """Correct a document's catalog entry: company, report_type, period."""
        return self._call("PATCH", f"/documents/{doc_id}", timeout=QUICK_TIMEOUT, json=fields)

    def send_feedback(self, trace_id: str, value: int) -> None:
        self._call("POST", "/feedback", timeout=QUICK_TIMEOUT, json={"trace_id": trace_id, "value": value})
