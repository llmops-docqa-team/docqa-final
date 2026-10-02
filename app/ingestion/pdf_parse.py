"""PDF -> one PageRecord per page (PyMuPDF text, tables, per-page OCR fallback).

Pure with respect to the app: no DB, no embeddings. OCR is injectable (`ocr_fn`) so tests and the
preview script don't need tesseract.

Two-pass strategy: text-native pages are yielded first (milliseconds each), then pages requiring OCR
are yielded in a second pass. This means the bulk of the document is searchable almost immediately,
and slow OCR pages trickle in afterwards — critical for async queries during ingestion.
"""
from __future__ import annotations

import os

import pytesseract

# Windows: point pytesseract at the default Tesseract install, so OCR works without Tesseract on PATH.
if os.name == "nt" and os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import multiprocessing
import os
import re
import shutil
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import pymupdf as fitz

from app.config import OCRConfig, ParsingConfig
from app.ingestion.tables import BBox, Table, find_page_tables, find_text_tables, fix_rupee, has_row_labels
from app.observability.logging import get_logger

log = get_logger("ingestion.parse")

# ---- Tesseract on Windows without PATH ---------------------------------------------------------
# The Windows installer (UB-Mannheim) puts tesseract.exe here but does not always add it to PATH. When
# `tesseract` is not on PATH, OCR uses the first of these that exists. TESSERACT_CMD (an env var holding the
# full path to tesseract.exe) wins over both. On Linux/macOS (apt, brew) nothing changes.
WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def tesseract_cmd() -> str | None:
    """The tesseract executable to use instead of the PATH lookup, or None to keep pytesseract's default."""
    explicit = os.environ.get("TESSERACT_CMD", "").strip()
    if explicit:
        return explicit
    if os.name != "nt" or shutil.which("tesseract"):
        return None
    return next((p for p in WINDOWS_TESSERACT_PATHS if os.path.isfile(p)), None)

Paragraphs = list[list[str]]  # paragraph -> lines


class OCRUnavailableError(RuntimeError):
    """Tesseract (or pytesseract/Pillow) is missing."""


@dataclass
class PageRecord:
    pdf_page: int            # 1-based index in the file
    page_label: str          # printed label from the PDF, else str(pdf_page)
    text: str                # paragraphs separated by blank lines; table regions removed
    image_area_ratio: float
    source_kind: str         # "text" | "ocr"
    tables: list[Table] = field(default_factory=list)
    error: str | None = None  # set when the page could not be processed (e.g. OCR unavailable)
    heading: str = ""         # the page's title ("Standalone Balance Sheet as at March 31, 2025"), or ""


OcrFn = Callable[["fitz.Page", OCRConfig], str]


# ---- scanned-page detection -------------------------------------------------------------------

def image_area_ratio(page: fitz.Page) -> float:
    """Fraction of the page covered by images (union of image rectangles, clipped to the page)."""
    page_rect = page.rect
    area = page_rect.width * page_rect.height
    if area <= 0:
        return 0.0
    rects = []
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page_rect
        if not r.is_empty:
            rects.append(r)
    return min(1.0, _union_area(rects) / area)


def _union_area(rects: list[fitz.Rect]) -> float:
    if not rects:
        return 0.0
    xs = sorted({x for r in rects for x in (r.x0, r.x1)})
    total = 0.0
    for x0, x1 in zip(xs, xs[1:], strict=False):
        spans = sorted((r.y0, r.y1) for r in rects if r.x0 <= x0 and r.x1 >= x1)
        covered, end = 0.0, float("-inf")
        for y0, y1 in spans:
            y0 = max(y0, end)
            if y1 > y0:
                covered += y1 - y0
                end = y1
        total += covered * (x1 - x0)
    return total


def is_scanned_page(text_chars: int, image_ratio: float, ocr: OCRConfig) -> bool:
    return text_chars < ocr.min_chars and image_ratio >= ocr.min_image_area_ratio


# ---- text extraction --------------------------------------------------------------------------

def _clean_line(s: str) -> str:
    """Collapse whitespace, fold ligatures (NFKC: 'ﬁ' -> 'fi', so 'Oﬀerings' is searchable) and restore
    a rupee sign extracted as a backtick."""
    return fix_rupee(re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip())


def native_paragraphs(page: fitz.Page, exclude: list[BBox] | None = None) -> Paragraphs:
    """Text blocks in reading order as paragraphs of lines. Blocks centred in `exclude` are skipped."""
    out: Paragraphs = []
    for x0, y0, x1, y1, text, _no, kind in page.get_text("blocks", sort=True):
        if kind != 0:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if exclude and any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in exclude):
            continue
        lines = [ln for ln in (_clean_line(x) for x in text.split("\n")) if ln]
        if lines:
            out.append(lines)
    return out


# A table's title is the text printed just above it, in its own column (financial statements put
# "Standalone Balance Sheet / As at ..." and the unit line there, and the table itself has neither).
TITLE_MAX_BLOCKS = 3
TITLE_MAX_CHARS = 300
TITLE_MAX_GAP = 120.0  # points between the title block's bottom and the table's top


def _overlaps(a: BBox, b: BBox) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _in_box(cx: float, cy: float, b: BBox) -> bool:
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def table_titles(page: fitz.Page, tables: list[Table], repeated: set[str]) -> list[Table]:
    """The same tables, each with `title` set from the text blocks right above it (nearest last).

    Blocks inside any table, repeated header/footer lines and blocks above the previous table in the same
    column are ignored. Never raises: no title is better than a failed page."""
    if not tables:
        return tables
    try:
        blocks = []
        for x0, y0, x1, y1, text, _no, kind in page.get_text("blocks", sort=True):
            if kind != 0:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(_in_box(cx, cy, t.bbox) for t in tables):
                continue
            lines = [ln for ln in (_clean_line(x) for x in text.split("\n")) if ln]
            lines = [ln for ln in lines if line_key(ln) not in repeated]
            if lines:
                blocks.append((x0, y0, x1, y1, " ".join(lines)))
        out = []
        for t in tables:
            tx0, ty0, tx1, _ = t.bbox
            floor = max(
                (o.bbox[3] for o in tables if o is not t and o.bbox[3] <= ty0 + 2
                 and min(o.bbox[2], tx1) - max(o.bbox[0], tx0) > 0),
                default=0.0,
            )
            above = [
                b for b in blocks
                if tx0 - 5 <= (b[0] + b[2]) / 2 <= tx1 + 5
                and b[3] <= ty0 + 2 and ty0 - b[3] <= TITLE_MAX_GAP and b[1] >= floor - 2
            ]
            above.sort(key=lambda b: b[3], reverse=True)  # nearest first
            picked, used = [], 0
            for b in above[:TITLE_MAX_BLOCKS]:
                if picked and used + len(b[4]) > TITLE_MAX_CHARS:
                    break
                picked.append(b)
                used += len(b[4])
            title = " ".join(b[4] for b in sorted(picked, key=lambda b: b[3]))
            out.append(replace(t, title=title[:TITLE_MAX_CHARS]))
        return out
    except Exception as exc:
        log.warning("table_title_failed", page=page.number + 1, error=str(exc))
        return tables


def ocr_paragraphs(text: str) -> Paragraphs:
    out: Paragraphs = []
    for chunk in re.split(r"\n\s*\n", text):
        lines = [ln for ln in (_clean_line(x) for x in chunk.split("\n")) if ln]
        if lines:
            out.append(lines)
    return out


def paragraphs_to_text(paras: Paragraphs) -> str:
    return "\n\n".join(" ".join(p) for p in paras)


# ---- page heading ----------------------------------------------------------------------------
# A page's heading is its biggest text near the top ("Standalone Balance Sheet" + "as on 31st March, 2025").
# Every chunk of the page gets it in front (see chunking.chunk_page), so a figure deep in a long statement
# is still found by the statement's name: without it, the chunk holding "Total equity 9,07,400.25" has none
# of the words "standalone", "balance sheet" or "March 31, 2025" and loses to notes pages that do.
HEADING_MIN_RATIO = 1.2     # vs the page's median font size
HEADING_TOP_SHARE = 0.3     # only the top 30% of the page
HEADING_MIN_CHARS = 8       # skips page numbers, section numbers and one-word decorations
HEADING_MAX_CHARS = 200
_SUBTITLE = re.compile(r"^(as at|as on|for the (year|period|quarter))\b", re.IGNORECASE)


def page_heading(page: fitz.Page) -> str:
    """The page's title line(s), or "" when no line stands out. Two-column spreads give "A | B".
    Never raises: a page without a heading is better than a failed page."""
    try:
        lines = []  # (size, x0, y0, text)
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                text = _clean_line("".join(s["text"] for s in line["spans"]))
                if spans and text:
                    lines.append((max(s["size"] for s in spans), line["bbox"][0], line["bbox"][1], text))
        if not lines:
            return ""
        median = statistics.median(size for size, *_ in lines)
        top = [ln for ln in lines if ln[2] < HEADING_TOP_SHARE * page.rect.height]
        big = [
            ln for ln in top
            if ln[0] >= HEADING_MIN_RATIO * median and len(ln[3]) >= HEADING_MIN_CHARS
            and not re.fullmatch(r"[\d\W]+", ln[3])
        ]
        if not big:
            return ""
        biggest = max(size for size, *_ in big)
        parts: list[str] = []
        for size, x0, y0, text in sorted(big, key=lambda ln: (ln[1], ln[2])):
            if size < biggest - 0.5:
                continue
            sub = next(
                (t for _s, x, y, t in top if _SUBTITLE.match(t) and abs(x - x0) < 40 and 0 < y - y0 < 40),
                "",
            )
            part = f"{text} {sub}" if sub and sub.lower() not in text.lower() else text
            if part not in parts:
                parts.append(part)
        return " | ".join(parts)[:HEADING_MAX_CHARS]
    except Exception as exc:
        log.warning("page_heading_failed", page=page.number + 1, error=str(exc))
        return ""


# ---- header/footer stripping ------------------------------------------------------------------

def line_key(line: str) -> str:
    """Normalised form so 'Page 3 of 300' and 'Page 4 of 300' count as the same line."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.lower())).strip()


def _edge_positions(n_lines: int, k: int) -> set[int]:
    return set(range(min(k, n_lines))) | set(range(max(0, n_lines - k), n_lines))


def edge_keys(paras: Paragraphs, k: int) -> set[str]:
    lines = [ln for p in paras for ln in p]
    return {line_key(lines[i]) for i in _edge_positions(len(lines), k)}


def find_repeated_keys(per_page_edge_keys: list[set[str]], cfg: ParsingConfig) -> set[str]:
    """Edge lines present on > cfg.repeat_line_ratio of the pages that have text."""
    pages = [keys for keys in per_page_edge_keys if keys]
    if len(pages) < cfg.repeat_min_pages:
        return set()
    counts = Counter(key for keys in pages for key in keys)
    return {key for key, n in counts.items() if key and n > cfg.repeat_line_ratio * len(pages)}


def strip_repeated(paras: Paragraphs, repeated: set[str], k: int) -> Paragraphs:
    if not repeated:
        return paras
    flat = [(pi, ln) for pi, p in enumerate(paras) for ln in p]
    drop = {i for i in _edge_positions(len(flat), k) if line_key(flat[i][1]) in repeated}
    kept: Paragraphs = [[] for _ in paras]
    for i, (pi, ln) in enumerate(flat):
        if i not in drop:
            kept[pi].append(ln)
    return [p for p in kept if p]


# ---- OCR --------------------------------------------------------------------------------------

def ocr_page_image(page: fitz.Page, ocr: OCRConfig) -> str:
    """Render the page and run tesseract on it."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OCRUnavailableError(f"pytesseract/Pillow not installed: {exc}") from exc
    cmd = tesseract_cmd()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    pix = page.get_pixmap(dpi=ocr.dpi, colorspace=fitz.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    try:
        return pytesseract.image_to_string(img, lang=ocr.language)
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailableError(
            "Tesseract is not installed or not on PATH (apt install tesseract-ocr / brew install "
            "tesseract / install from https://github.com/UB-Mannheim/tesseract/wiki, or set TESSERACT_CMD "
            "to the full path of tesseract.exe)"
        ) from exc


# ---- text pages (shared by the in-process and the parallel path) -------------------------------

def _text_page_record(page: fitz.Page, repeated: set[str], parsing: ParsingConfig) -> PageRecord:
    pdf_page = page.number + 1
    tables = find_page_tables(page, min_rows=parsing.table_min_rows, min_cols=parsing.table_min_cols)
    rebuilt = find_text_tables(page, exclude=[t.bbox for t in tables if has_row_labels(t)])
    if rebuilt:  # they replace any label-less scrap that find_tables made of the same figures
        tables = [
            t for t in tables if has_row_labels(t) or not any(_overlaps(t.bbox, r.bbox) for r in rebuilt)
        ]
        tables = sorted([*tables, *rebuilt], key=lambda t: (t.bbox[1], t.bbox[0]))
    tables = table_titles(page, tables, repeated)
    paras = native_paragraphs(page, exclude=[t.bbox for t in tables])
    paras = strip_repeated(paras, repeated, parsing.repeat_edge_lines)
    return PageRecord(
        pdf_page, page.get_label() or str(pdf_page), paragraphs_to_text(paras), image_area_ratio(page),
        "text", tables, heading=page_heading(page),
    )


def _parse_text_pages(path: str, page_indexes: list[int], repeated: set[str], parsing: ParsingConfig):
    """Worker-process task: open the PDF and parse these text pages (PyMuPDF objects can't cross processes,
    so each worker opens its own copy; only the PageRecords come back)."""
    with fitz.open(path) as doc:
        return [_text_page_record(doc[i], repeated, parsing) for i in page_indexes]


def parse_workers(parsing: ParsingConfig, n_pages: int) -> int:
    """How many worker processes to use for `n_pages` text pages (1 = parse in-process)."""
    if n_pages < parsing.parallel_min_pages:
        return 1
    if parsing.workers > 0:
        return parsing.workers
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _iter_text_pages(
    path: str, doc: fitz.Document, indexes: list[int], repeated: set[str], parsing: ParsingConfig
) -> Iterator[PageRecord]:
    """Text pages in page order: in-process for small documents, else batches spread over worker processes.
    Results are yielded in order as each batch finishes, so indexing (and the PARTIAL status) can start
    before the whole document is parsed. Closing the generator (a cancelled job) stops the pool."""
    workers = parse_workers(parsing, len(indexes))
    if workers <= 1:
        for i in indexes:
            yield _text_page_record(doc[i], repeated, parsing)
        return
    size = max(1, parsing.parallel_batch_pages)
    batches = [indexes[k : k + size] for k in range(0, len(indexes), size)]
    # "spawn" on every OS: fork from a process that runs the API's threads can deadlock, and Windows has
    # only spawn anyway. Workers import this module afresh; the parent passes plain data.
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    log.info("parse_parallel", workers=workers, pages=len(indexes), batches=len(batches))
    try:
        futures = [pool.submit(_parse_text_pages, path, b, repeated, parsing) for b in batches]
        for fut in futures:
            yield from fut.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ---- main entry -------------------------------------------------------------------------------

def parse_pdf(
    path: str | Path,
    ocr: OCRConfig,
    parsing: ParsingConfig,
    *,
    ocr_fn: OcrFn | None = None,
) -> Iterator[PageRecord]:
    """Yield one PageRecord per page in two passes: text-native pages first, then OCR pages.

    **Pass 1 (fast):** all pages whose native text meets the threshold are extracted, chunked and
    yielded immediately — typically milliseconds per page. After this pass the vast majority of the
    document is searchable.

    **Pass 2 (slow):** pages identified as scanned/image-heavy are rendered and OCR'd via Tesseract,
    then yielded. Only pages that actually need it pay the OCR cost.

    This ordering means the full document is queryable much sooner during async ingestion, because
    the balance sheet on page 78 (for example) no longer waits behind 77 pages of sequential
    processing — it is indexed in the first fast sweep.

    Raises on unreadable / password-protected files. A page whose OCR fails is yielded with
    `error` set and empty text; the rest of the document still parses.
    """
    ocr_fn = ocr_fn or ocr_page_image
    with fitz.open(str(path)) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected")

        # Pre-pass (cheap, no OCR/tables): find lines repeated on the edges of most pages.
        edges = [edge_keys(native_paragraphs(p), parsing.repeat_edge_lines) for p in doc]
        repeated = find_repeated_keys(edges, parsing)

        # Classify every page up front (cheap) so text pages and OCR pages go in two passes.
        text_pages: list[int] = []
        ocr_deferred: list[tuple[int, int, str, float]] = []  # (page_index, pdf_page, label, ratio)
        for page in doc:
            ratio = image_area_ratio(page)
            if is_scanned_page(len(page.get_text("text").strip()), ratio, ocr):
                pdf_page = page.number + 1
                ocr_deferred.append((page.number, pdf_page, page.get_label() or str(pdf_page), ratio))
            else:
                text_pages.append(page.number)

        # --- Pass 1: text-native pages (fast; in parallel worker processes for long documents) ---
        yield from _iter_text_pages(str(path), doc, text_pages, repeated, parsing)

        if ocr_deferred:
            log.info(
                "ocr_pass_starting",
                n_pages=len(ocr_deferred),
                page_numbers=[d[1] for d in ocr_deferred],
            )

        # --- Pass 2: OCR pages (slow) ---
        for page_index, pdf_page, label, ratio in ocr_deferred:
            page = doc[page_index]
            error, paras = None, []
            try:
                paras = ocr_paragraphs(ocr_fn(page, ocr))
            except Exception as exc:
                error = f"OCR failed on page {pdf_page}: {exc}"
                log.warning("ocr_failed", page=pdf_page, error=str(exc))
            paras = strip_repeated(paras, repeated, parsing.repeat_edge_lines)
            yield PageRecord(pdf_page, label, paragraphs_to_text(paras), ratio, "ocr", [], error)