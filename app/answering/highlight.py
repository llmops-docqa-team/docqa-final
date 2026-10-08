"""'Show in PDF': which figures to highlight on a cited page, and the cropped picture of that page.

`highlight_terms` runs when an answer is built (string work on text already in memory). `render_highlight`
runs only when the user clicks the button: it finds the figures on the PDF page with PyMuPDF, shades the
whole printed row (so the row label is covered), boxes the figure, and renders a cropped PNG. No LLM, and
the PDF is never saved: the marks are drawn on the in-memory page only.
"""

from __future__ import annotations

import re

import pymupdf

from app.answering.numbers import _candidates, extract_numbers, parse_number

MAX_TERMS = 12
MAX_TERM_LEN = 32
TERM_PATTERN = re.compile(rf"[0-9A-Za-z,.%()\-]{{1,{MAX_TERM_LEN}}}")

_WORD = re.compile(r"[a-z]{3,}")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class PageOutOfRange(ValueError):
    """The requested PDF page does not exist."""


# ---------------------------------------------------------------- which figures (answer time)


def _printed_forms(chunk_text: str) -> dict:
    """Decimal -> the first way that number is printed in the chunk ('94.90', '1,097.36')."""
    forms: dict = {}
    for token, _start, _end in _candidates(chunk_text):
        value = parse_number(token)
        if value is not None:
            forms.setdefault(value, token)
    return forms


def _operands(computed: str) -> list[str]:
    """'482.06 = 1097.36 − 615.30' -> ['1097.36', '615.30'] (the right-hand side, without a '× 100')."""
    rhs = computed.partition("=")[2].replace("× 100", "")
    return _NUMBER.findall(rhs)


def highlight_terms(answer: str, computed: list[str], chunk_text: str) -> list[str]:
    """The answer's figures that literally occur in `chunk_text`, as printed there. A figure that is
    computed (not printed anywhere) contributes its operands instead. At most MAX_TERMS, in answer order."""
    forms = _printed_forms(chunk_text)
    if not forms:
        return []
    values = [q.mantissa for q in extract_numbers(answer, for_answer=True)]
    for how in computed:
        values += [v for v in map(parse_number, _operands(how)) if v is not None]
    terms: list[str] = []
    for v in values:
        term = forms.get(v)
        if term is None or term in terms or not TERM_PATTERN.fullmatch(term):
            continue
        if term.isdigit() and len(term) < 3:  # "12" would match half the page
            continue
        terms.append(term)
    return terms[:MAX_TERMS]


# ---------------------------------------------------------------- the picture (click time)


def _words_of(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _row_words(words: list, rect: pymupdf.Rect) -> list:
    """Words on the same printed line(s) as `rect`: those whose vertical centre falls inside its height."""
    return [w for w in words if rect.y0 <= (w[1] + w[3]) / 2 <= rect.y1]


def _row_rect(words: list, hit: pymupdf.Rect) -> pymupdf.Rect:
    row = pymupdf.Rect(hit)
    for w in _row_words(words, hit):
        row |= pymupdf.Rect(w[:4])
    return row


def _exact_hits(page: pymupdf.Page, term: str, words: list) -> list[pymupdf.Rect]:
    """`search_for` matches substrings ('94.90' in '1,094.90'); keep hits that are a whole printed word."""
    out = []
    for hit in page.search_for(term):
        inside = [w[4].strip("()%*,;:") for w in words if pymupdf.Rect(w[:4]).intersects(hit)]
        if term in inside or term.strip("()%") in inside:
            out.append(hit)
    return out


def _prefer_matching_rows(hits: list, words: list, context: str) -> list:
    """Several places hold the figure: keep those whose row shares the most words with the cited chunk."""
    if len(hits) < 2 or not context:
        return hits
    ctx = _words_of(context)
    scores = [len(_words_of(" ".join(w[4] for w in _row_words(words, h))) & ctx) for h in hits]
    best = max(scores)
    if best == 0:
        return hits
    return [h for h, s in zip(hits, scores, strict=True) if s == best]


def render_highlight(
    pdf_path, pdf_page: int, terms: list[str], *, context: str = "", zoom: float = 1.6, margin: float = 150
) -> tuple[bytes, int]:
    """PNG of `pdf_page` (1-based) cropped around the rows holding `terms`, and the number of figures found.
    With no match the whole page is rendered and the count is 0."""
    with pymupdf.open(pdf_path) as doc:
        if not 1 <= pdf_page <= doc.page_count:
            raise PageOutOfRange(f"The PDF has {doc.page_count} pages.")
        page = doc[pdf_page - 1]
        words = page.get_text("words")
        hits: list[pymupdf.Rect] = []
        for term in terms:
            hits += _prefer_matching_rows(_exact_hits(page, term, words), words, context)

        if not hits:
            return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png"), 0

        rows = [_row_rect(words, h) for h in hits]
        for row in {tuple(round(v) for v in r): r for r in rows}.values():  # one shade per printed row
            page.add_highlight_annot(row)
        for hit in hits:
            box = page.add_rect_annot(hit)
            box.set_colors(stroke=(0.85, 0.1, 0.1))
            box.set_border(width=1.2)
            box.update()

        area = rows[0]
        for row in rows[1:]:
            area |= row
        clip = pymupdf.Rect(page.rect.x0, area.y0 - margin, page.rect.x1, area.y1 + margin) & page.rect
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip)
        return pix.tobytes("png"), len(hits)
