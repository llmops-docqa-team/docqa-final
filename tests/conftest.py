"""Shared test helpers: a fake embedder, small PDF builders, and an app wired to temp dirs."""
from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Sequence

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, load_settings
from app.main import create_app

PARA = (
    "Revenue from operations grew steadily during the year, driven by strong demand across all "
    "segments and geographies. Management expects the momentum to continue into the next fiscal "
    "year, subject to input cost trends and the broader macroeconomic environment."
)


class FakeEmbedder:
    """Deterministic 16-d vectors from a hash. No model download, no network."""

    def __init__(self, model_name: str = "fake-embed-v1", *, fail_on: str | None = None):
        self.model_name = model_name
        self.fail_on = fail_on          # raise if any text in a batch contains this marker
        self.batch_sizes: list[int] = []
        self.gate: threading.Event | None = None   # if set, block on the Nth call until released
        self.gate_on_call = 2
        self.entered_gate = threading.Event()

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(b - 127.5) / 127.5 for b in digest[:16]]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        if self.gate is not None and len(self.batch_sizes) == self.gate_on_call:
            self.entered_gate.set()
            assert self.gate.wait(10), "test never released the gate"
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise RuntimeError("embedder exploded")
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def page_body(label: str, i: int) -> str:
    """A page whose lines differ from every other page's (digits are normalised by the header/footer
    stripper, so a per-page *word* is needed or the whole page would look like a repeated header)."""
    tag = "".join(chr(97 + (i * 7 + k * 3) % 26) for k in range(9))
    return f"{label} {tag} overview. " + " ".join(f"{PARA} Note on {tag}{chr(97 + j)}." for j in range(2))


def make_pdf(path, pages: list[str] | int = 3, *, label: str = "doc") -> bytes:
    """Text PDF with one body per page. `pages` is a list of bodies or a page count."""
    bodies = pages if isinstance(pages, list) else [page_body(label, i) for i in range(pages)]
    doc = fitz.open()
    for body in bodies:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(72, 80, 523, 760), body, fontsize=10)
    doc.save(str(path))
    doc.close()
    return path.read_bytes()


def make_scanned_pdf(path, n: int = 1) -> None:
    """Image-only PDF (a rasterised text page), so every page takes the OCR route."""
    src = fitz.open()
    page = src.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(72, 80, 523, 760), "SCANNED PAGE QUARTERLY REVENUE " * 6, fontsize=10)
    pix = page.get_pixmap(dpi=100)
    src.close()
    doc = fitz.open()
    for _ in range(n):
        p = doc.new_page(width=595, height=842)
        p.insert_image(p.rect, pixmap=pix)
    doc.save(str(path))
    doc.close()


def make_table_pdf(path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(72, 80, 523, 200), "Segment results are summarised below.", fontsize=10)
    rows = [["Segment", "FY24", "FY25"], ["Cables", "1,200", "1,450"], ["Panels", "800", "950"]]
    x0, y0, cw, rh = 72, 300, 120, 24
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            page.draw_rect(rect, width=0.8)
            page.insert_text((rect.x0 + 6, rect.y0 + 16), cell, fontsize=10)
    doc.save(str(path))
    doc.close()


def fake_ocr(page, cfg) -> str:
    return "Scanned report text recovered by the fake OCR engine.\n\nSecond paragraph of scanned text."


@pytest.fixture(autouse=True)
def _no_tracing_or_fallback(monkeypatch):
    """Tests never talk to Langfuse or Ollama, whatever the developer's shell has set."""
    from app.observability.tracing import Tracer, set_tracer

    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name in ("OLLAMA_BASE_URL", "LLM_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    set_tracer(Tracer())
    yield
    set_tracer(Tracer())


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = load_settings()
    s.paths.sqlite_path = str(tmp_path / "db.sqlite")
    s.paths.upload_dir = str(tmp_path / "uploads")
    s.paths.chroma_dir = str(tmp_path / "chroma")
    s.paths.model_cache_dir = str(tmp_path / "models")
    s.ingestion.upsert_every_pages = 2
    s.api.debug_endpoints = True     # /debug/* are off by default; step 04/05 tests use them
    s.query.rewrite = False          # the scripted fake LLMs answer router/doc/general calls only
    s.query.require_company = False  # tests upload several unrelated files
    return s


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make_client(settings, monkeypatch):
    """Factory: `with make_client(embedder, ocr_fn=...) as client:` runs the real lifespan."""
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    def factory(embedder, ocr_fn=fake_ocr) -> TestClient:
        return TestClient(create_app(embedder=embedder, ocr_fn=ocr_fn))

    return factory


def wait_for(client: TestClient, doc_id: str, statuses: tuple[str, ...], timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    doc: dict = {}
    while time.monotonic() < deadline:
        doc = client.get(f"/documents/{doc_id}").json()
        if doc.get("status") in statuses:
            return doc
        time.sleep(0.02)
    raise AssertionError(f"document never reached {statuses}; last seen: {doc}")


def upload(client: TestClient, path_or_bytes, name: str = "report.pdf", content_type="application/pdf"):
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else path_or_bytes.read_bytes()
    return client.post("/documents", files={"file": (name, data, content_type)})
