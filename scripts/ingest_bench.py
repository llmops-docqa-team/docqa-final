"""Ingest one PDF through the real pipeline (real fastembed + Chroma + tesseract) and print seconds/page.

    python scripts/ingest_bench.py report.pdf [--pages 1-60] [--rasterise] [--dpi 150]

--pages      only ingest this pdf page range (a trimmed copy is made)
--rasterise  turn the pages into an image-only PDF first, to mimic a clean scan (forces the OCR route)
Uses a throwaway SQLite/Chroma/uploads dir, but the real model cache (data/models), so the model
downloads once.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf as fitz  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.ingestion.embedder import embedder_from_settings  # noqa: E402
from app.ingestion.index import VectorIndex  # noqa: E402
from app.ingestion.worker import IngestionWorker, upload_path  # noqa: E402
from app.storage.db import init_db  # noqa: E402
from app.storage.documents import DocumentStore  # noqa: E402


def _prepare(src: Path, dst: Path, pages: str | None, rasterise: bool, dpi: int) -> None:
    lo, hi = 1, 10**9
    if pages:
        a, _, b = pages.partition("-")
        lo, hi = int(a), int(b or a)
    with fitz.open(str(src)) as doc:
        hi = min(hi, doc.page_count)
        out = fitz.open()
        if not rasterise:
            out.insert_pdf(doc, from_page=lo - 1, to_page=hi - 1)
        else:
            for i in range(lo - 1, hi):
                pix = doc[i].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                page = out.new_page(width=doc[i].rect.width, height=doc[i].rect.height)
                page.insert_image(page.rect, pixmap=pix)
        out.save(str(dst), garbage=3, deflate=True)
        out.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--pages")
    ap.add_argument("--rasterise", action="store_true")
    ap.add_argument("--dpi", type=int, default=150, help="dpi of the rasterised 'scan' (not the OCR dpi)")
    args = ap.parse_args()

    settings = load_settings()
    work = Path(tempfile.mkdtemp(prefix="finchat-bench-"))
    settings.paths.sqlite_path = str(work / "db.sqlite")
    settings.paths.upload_dir = str(work / "uploads")
    settings.paths.chroma_dir = str(work / "chroma")

    init_db(settings.sqlite_path)
    settings.upload_dir.mkdir(parents=True)
    doc_id = "bench"
    path = upload_path(settings, doc_id)
    _prepare(Path(args.pdf), path, args.pages, args.rasterise, args.dpi)
    with fitz.open(str(path)) as d:
        n_pages = d.page_count

    embedder = embedder_from_settings(settings)
    t0 = time.perf_counter()
    embedder.embed_documents(["warm up"])  # model load/download is not ingestion time
    print(f"model ready in {time.perf_counter() - t0:.1f}s (excluded below)")

    store = DocumentStore(settings.sqlite_path)
    index = VectorIndex(settings.chroma_dir, embedder.model_name)
    store.insert(doc_id, Path(args.pdf).name, "bench", n_pages)
    IngestionWorker(settings, store, index, embedder).process(doc_id)

    d = store.get(doc_id)
    if d["status"] != "READY":
        print(f"FAILED: {d['status']} {d['error']}")
        return 1
    timings = d["page_timings"]
    n = d["pages_total"]
    print(f"file:           {Path(args.pdf).name} ({n} pages{', rasterised' if args.rasterise else ''})")
    print(f"pages:          text={d['n_text_pages']} (with tables={d['n_table_pages']}) "
          f"ocr={d['n_ocr_pages']} failed={d['n_failed_pages']}")
    print(f"chunks:         {d['chunks']}  (vectors in Chroma: {index.count(doc_id)})")
    print(f"total:          {d['ingest_seconds']:.1f}s -> {d['ingest_seconds'] / n:.3f} s/page")
    print(f"  parse+chunk:  {sum(timings):.1f}s  mean {statistics.mean(timings):.3f} s/page "
          f"(median {statistics.median(timings):.3f}, max {max(timings):.2f})")
    rest = d["ingest_seconds"] - sum(timings) - d["embed_seconds"]
    ms_per_chunk = d["embed_seconds"] / max(d["chunks"], 1) * 1000
    print(f"  embed:        {d['embed_seconds']:.1f}s ({ms_per_chunk:.0f} ms/chunk)")
    print(f"  other (Chroma upsert, SQLite): {rest:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
