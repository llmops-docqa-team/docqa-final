"""Re-run ingestion for documents already uploaded (after a parsing/chunking change: an INGEST_VERSION bump).

    python scripts/reingest.py            # every document stored with an older ingest_version
    python scripts/reingest.py --all      # every document
    python scripts/reingest.py DOC_ID ... # these documents

Uses the real config, SQLite, uploads and Chroma index. Stop the API first (one writer at a time).
Needs no LLM key: only parsing, the local embedding model and Chroma.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_settings  # noqa: E402
from app.ingestion.embedder import embedder_from_settings  # noqa: E402
from app.ingestion.index import INGEST_VERSION, VectorIndex  # noqa: E402
from app.ingestion.worker import IngestionWorker  # noqa: E402
from app.storage.db import init_db  # noqa: E402
from app.storage.documents import DocumentStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc_ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="re-ingest every document, not just outdated ones")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    settings = load_settings(args.config)
    init_db(settings.sqlite_path)
    store = DocumentStore(settings.sqlite_path)
    embedder = embedder_from_settings(settings)
    index = VectorIndex(settings.chroma_dir, embedder.model_name)
    worker = IngestionWorker(settings, store, index, embedder)

    docs = store.list()
    if args.doc_ids:
        docs = [d for d in docs if d["id"] in set(args.doc_ids)]
    elif not args.all:
        docs = [d for d in docs if (d.get("ingest_version") or 0) < INGEST_VERSION]
    if not docs:
        print("nothing to re-ingest")
        return 0
    for d in docs:
        t0 = time.perf_counter()
        worker.process(d["id"])  # same job the API's worker runs; marks the document FAILED on error
        after = store.get(d["id"])
        note = f", error: {after['error']}" if after.get("error") else ""
        print(f"{d['id']} {d['filename']}: {after['status']}, {after.get('chunks')} chunks, "
              f"{time.perf_counter() - t0:.0f}s{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
