"""Eyeball how a real PDF parses and chunks.

    python scripts/parse_preview.py report.pdf [--samples 3] [--pages 1-20] [--no-ocr]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.ingestion.chunking import chunk_page, title_from_filename  # noqa: E402
from app.ingestion.pdf_parse import OCRUnavailableError, parse_pdf  # noqa: E402


def _page_range(spec: str | None) -> tuple[int, int]:
    if not spec:
        return 1, 10**9
    lo, _, hi = spec.partition("-")
    return int(lo), int(hi or lo)


def _no_ocr(page, cfg):
    raise OCRUnavailableError("OCR disabled (--no-ocr)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--samples", type=int, default=3, help="sample chunks to print (spread over the doc)")
    ap.add_argument("--pages", help="pdf page range to process, e.g. 5-20")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR (scanned pages are reported as errors)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

    settings = load_settings()
    path = Path(args.pdf)
    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    title = title_from_filename(path.name)
    lo, hi = _page_range(args.pages)

    print(f"{path.name}  doc_id={doc_id}  title={title!r}")
    print(f"{'pdf':>4} {'label':>6} {'kind':<5} {'chars':>6} {'img%':>5} {'tables':>6} {'chunks':>6}  note")
    all_chunks, kinds, t0 = [], Counter(), time.perf_counter()
    for page in parse_pdf(path, settings.ocr, settings.parsing, ocr_fn=_no_ocr if args.no_ocr else None):
        if page.pdf_page < lo:
            continue
        if page.pdf_page > hi:
            break
        chunks = chunk_page(page, doc_id=doc_id, filename=path.name, doc_title=title, cfg=settings.chunking)
        all_chunks += chunks
        kinds.update(c.source_kind for c in chunks)
        print(
            f"{page.pdf_page:>4} {page.page_label:>6} {page.source_kind:<5} {len(page.text):>6} "
            f"{page.image_area_ratio * 100:>4.0f}% {len(page.tables):>6} {len(chunks):>6}  {page.error or ''}"
        )
    secs = time.perf_counter() - t0
    print(f"\n{len(all_chunks)} chunks {dict(kinds)} in {secs:.1f}s")

    step = max(1, len(all_chunks) // max(args.samples, 1))
    for c in all_chunks[::step][: args.samples]:
        print(f"\n--- {c.id}  [{c.source_kind}]  page {c.page} (label {c.page_label})  {c.char_len} chars")
        print(c.embed_text[:700] + (" …" if len(c.embed_text) > 700 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
