"""Page records -> chunks. Splits recursively (paragraph -> sentence -> word) and never crosses a page."""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from app.config import ChunkingConfig
from app.ingestion.pdf_parse import PageRecord
from app.ingestion.tables import table_to_markdown_chunks
from app.ingestion.tokens import estimate_tokens

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PARA_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    filename: str
    page: int            # pdf page index (1-based)
    page_label: str
    chunk_idx: int       # index within its kind on the page (text/ocr, or table pieces)
    source_kind: str     # "text" | "table" | "ocr"
    text: str            # stored and shown as-is
    embed_text: str      # "{doc title} — page {label}" header + text; this is what gets embedded
    char_len: int


def title_from_filename(filename: str) -> str:
    return re.sub(r"[_\s]+", " ", Path(filename).stem).strip()


def _hard_split(s: str, size: int) -> list[str]:
    """Last resort for a single 'word' bigger than the budget (base64 blobs, long URLs)."""
    if estimate_tokens(s) <= size or len(s) < 2:
        return [s]
    mid = len(s) // 2
    return _hard_split(s[:mid], size) + _hard_split(s[mid:], size)


def _split_paragraph(para: str, size: int) -> list[str]:
    if estimate_tokens(para) <= size:
        return [para]
    out: list[str] = []
    for sentence in _SENTENCE_RE.split(para):
        if estimate_tokens(sentence) <= size:
            out.append(sentence)
        else:
            for word in sentence.split():
                out.extend(_hard_split(word, size))
    return out


def _pieces(text: str, size: int) -> list[tuple[str, str]]:
    """(piece, separator-before-it). Paragraphs first; oversized ones fall to sentences, then words."""
    out: list[tuple[str, str]] = []
    paras = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    for pi, para in enumerate(paras):
        for j, piece in enumerate(_split_paragraph(para, size)):
            out.append((piece, ("\n\n" if pi else "") if j == 0 else " "))
    return out


def _overlap_tail(text: str, budget: int) -> str:
    """Trailing words of `text` worth at most `budget` tokens."""
    tail: list[str] = []
    used = 0
    for word in reversed(text.split()):
        t = estimate_tokens(word)
        if used + t > budget:
            break
        tail.append(word)
        used += t
    return " ".join(reversed(tail))


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Chunks of <= `size` estimated tokens; consecutive chunks share ~`overlap` trailing tokens."""
    if overlap >= size:
        raise ValueError("overlap_tokens must be smaller than size_tokens")
    chunks: list[str] = []
    cur = ""
    cur_tokens = 0
    for piece, sep in _pieces(text, size):
        t = estimate_tokens(piece)
        if cur and cur_tokens + t > size:
            chunks.append(cur)
            tail = _overlap_tail(cur, min(overlap, size - t))
            cur, cur_tokens, sep = tail, estimate_tokens(tail), " "
        cur = cur + sep + piece if cur else piece
        cur_tokens += t
    if cur:
        chunks.append(cur)
    return chunks


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _with_heading(heading: str, text: str) -> str:
    """`text` with the page heading in front, unless the text already starts with it (the page's first
    chunk usually does)."""
    if not heading or _norm(text).startswith(_norm(heading)):
        return text
    return f"{heading}\n\n{text}"


def _table_title(heading: str, title: str) -> str:
    """A table's title, with the page heading in front when the title does not already name it."""
    if not heading:
        return title
    if title and any(_norm(part) in _norm(title) for part in heading.split(" | ")):
        return title
    return f"{heading} — {title}" if title else heading


def chunk_page(
    page: PageRecord, *, doc_id: str, filename: str, doc_title: str, cfg: ChunkingConfig
) -> list[Chunk]:
    header = f"{doc_title} — page {page.page_label}"

    def make(chunk_id: str, idx: int, kind: str, text: str) -> Chunk:
        return Chunk(
            id=chunk_id, doc_id=doc_id, filename=filename, page=page.pdf_page,
            page_label=page.page_label, chunk_idx=idx, source_kind=kind, text=text,
            embed_text=f"{header}\n{text}", char_len=len(text),
        )

    heading = page.heading.strip()
    # Room for the heading inside the size budget, so a chunk with it in front stays within size_tokens.
    budget = cfg.size_tokens
    if heading:
        budget = max(cfg.overlap_tokens + 1, cfg.size_tokens - estimate_tokens(heading) - 1)
    chunks = [
        make(f"{doc_id}:{page.pdf_page}:{i}", i, page.source_kind, _with_heading(heading, text))
        for i, text in enumerate(split_text(page.text, budget, cfg.overlap_tokens))
    ]
    i = 0
    for table in page.tables:
        for md in table_to_markdown_chunks(table.rows, cfg.size_tokens, _table_title(heading, table.title)):
            chunks.append(make(f"{doc_id}:{page.pdf_page}:t{i}", i, "table", md))
            i += 1
    return chunks


def chunk_document(
    pages: Iterable[PageRecord], *, doc_id: str, filename: str, doc_title: str | None, cfg: ChunkingConfig
) -> Iterator[Chunk]:
    title = doc_title or title_from_filename(filename)
    for page in pages:
        yield from chunk_page(page, doc_id=doc_id, filename=filename, doc_title=title, cfg=cfg)
