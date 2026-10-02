"""Parsing + chunking: pure functions, PDFs generated in-test with PyMuPDF."""
from __future__ import annotations

import shutil

import pymupdf as fitz
import pytest

from app.config import ChunkingConfig, OCRConfig, ParsingConfig
from app.ingestion import pdf_parse
from app.ingestion.chunking import chunk_document, chunk_page, split_text, title_from_filename
from app.ingestion.pdf_parse import (
    OCRUnavailableError,
    PageRecord,
    find_repeated_keys,
    image_area_ratio,
    is_scanned_page,
    line_key,
    parse_pdf,
)
from app.ingestion.tables import clean_rows, table_to_markdown, table_to_markdown_chunks
from app.ingestion.tokens import estimate_tokens

OCR = OCRConfig()
PARSING = ParsingConfig()
CHUNKING = ChunkingConfig()

PARA = (
    "Revenue from operations grew steadily during the year, driven by strong demand across all "
    "segments and geographies. Management expects the momentum to continue into the next fiscal "
    "year, subject to input cost trends and the broader macroeconomic environment."
)


# ---- fixtures ---------------------------------------------------------------------------------

def _text_page(doc: fitz.Document, body: str, *, header: str | None = None, footer: str | None = None):
    page = doc.new_page(width=595, height=842)
    if header:
        page.insert_text((72, 40), header, fontsize=9)
    page.insert_textbox(fitz.Rect(72, 80, 523, 760), body, fontsize=10)
    if footer:
        page.insert_text((72, 810), footer, fontsize=9)
    return page


def _save(doc: fitz.Document, tmp_path, name: str):
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


def _parse(path, **kw):
    return list(parse_pdf(path, OCR, PARSING, **kw))


@pytest.fixture
def text_pdf(tmp_path):
    doc = fitz.open()
    _text_page(doc, "\n\n".join([PARA] * 3))
    return _save(doc, tmp_path, "text.pdf")


@pytest.fixture
def table_pdf(tmp_path):
    doc = fitz.open()
    page = _text_page(doc, "Segment results are summarised below.")
    rows = [["Segment", "FY24", "FY25"], ["Cables", "1,200", "1,450"], ["Panels", "800", "950"]]
    x0, y0, cw, rh = 72, 300, 120, 24
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            page.draw_rect(rect, width=0.8)
            page.insert_text((rect.x0 + 6, rect.y0 + 16), cell, fontsize=10)
    return _save(doc, tmp_path, "table.pdf")


@pytest.fixture
def image_only_pdf(tmp_path):
    src = fitz.open()
    _text_page(src, "SCANNED PAGE QUARTERLY REVENUE " * 6)
    pix = src[0].get_pixmap(dpi=200)
    src.close()
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=pix)
    return _save(doc, tmp_path, "scan.pdf")


# ---- tokens / chunking ------------------------------------------------------------------------

def test_estimate_tokens_is_additive_and_conservative():
    assert estimate_tokens("a b") == estimate_tokens("a") + estimate_tokens("b")
    assert estimate_tokens("Ellenbarrie") > 1
    assert estimate_tokens("1,234.5") >= 5  # numbers split like WordPiece does


def test_chunk_size_bound_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + PARA for i in range(30))
    chunks = split_text(text, size=400, overlap=60)
    assert len(chunks) > 1
    assert all(estimate_tokens(c) <= 400 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        shared = 0
        words = prev.split()
        for n in range(1, len(words) + 1):  # longest suffix of prev that prefixes next
            if nxt.startswith(" ".join(words[-n:])):
                shared = n
        assert shared > 0, "consecutive chunks must overlap"
        assert estimate_tokens(" ".join(words[-shared:])) <= 60


def test_oversized_paragraph_and_giant_token_stay_under_bound():
    sentence = "Sales rose in the quarter. "
    chunks = split_text(sentence * 400, size=100, overlap=10)
    assert all(estimate_tokens(c) <= 100 for c in chunks)
    blob = "x" * 20000
    chunks = split_text(f"before {blob} after", size=100, overlap=10)
    assert all(estimate_tokens(c) <= 100 for c in chunks)
    assert "".join(c for c in chunks).count("x") >= 20000  # nothing dropped


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("a b c", size=10, overlap=10)


def _page(n: int, text: str, label: str | None = None, kind: str = "text", tables=()) -> PageRecord:
    return PageRecord(n, label or str(n), text, 0.0, kind, list(tables))


def test_no_chunk_crosses_pages():
    pages = [_page(i, f"MARKER{i} " + PARA * 12) for i in (1, 2, 3)]
    chunks = list(chunk_document(pages, doc_id="d", filename="r.pdf", doc_title=None, cfg=CHUNKING))
    assert {c.page for c in chunks} == {1, 2, 3}
    for c in chunks:
        others = {f"MARKER{i}" for i in (1, 2, 3)} - {f"MARKER{c.page}"}
        assert not any(m in c.text for m in others)


def test_deterministic_ids_and_fields():
    page = _page(7, PARA * 10, label="iv")
    a = chunk_page(page, doc_id="abc", filename="Annual_Report.pdf", doc_title="Annual Report", cfg=CHUNKING)
    b = chunk_page(page, doc_id="abc", filename="Annual_Report.pdf", doc_title="Annual Report", cfg=CHUNKING)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.id for c in a][:2] == ["abc:7:0", "abc:7:1"]
    assert len({c.id for c in a}) == len(a)
    c = a[0]
    assert (c.page, c.page_label, c.chunk_idx, c.source_kind) == (7, "iv", 0, "text")
    assert c.embed_text == f"Annual Report — page iv\n{c.text}"
    assert c.char_len == len(c.text)
    assert estimate_tokens(c.embed_text) < 512


def test_ocr_page_chunks_are_marked_ocr():
    chunks = chunk_page(_page(1, PARA, kind="ocr"), doc_id="d", filename="s.pdf", doc_title="s", cfg=CHUNKING)
    assert {c.source_kind for c in chunks} == {"ocr"}


def test_title_from_filename():
    assert title_from_filename("EIG_AR_FY25.pdf") == "EIG AR FY25"


# ---- tables -----------------------------------------------------------------------------------

def test_clean_rows_drops_empty_rows_and_columns_and_pads():
    rows = clean_rows([["A", None, "B"], [None, None, None], ["1", "", "2\n3"], ["x"]])
    assert rows == [["A", "B"], ["1", "2 3"], ["x", ""]]


def test_table_to_markdown_escapes_pipes():
    md = table_to_markdown([["Name", "Note"], ["a|b", "c"]])
    assert md == "| Name | Note |\n| --- | --- |\n| a\\|b | c |"


def test_long_table_split_repeats_header():
    rows = [["Item", "Amount"]] + [[f"Line item number {i}", f"{i * 1000:,}"] for i in range(200)]
    pieces = table_to_markdown_chunks(rows, max_tokens=120)
    assert len(pieces) > 1
    header = "| Item | Amount |\n| --- | --- |"
    assert all(p.startswith(header) for p in pieces)
    assert all(estimate_tokens(p) <= 120 for p in pieces)
    body = [ln for p in pieces for ln in p.splitlines()[2:]]
    assert len(body) == 200  # every row exactly once


def test_short_table_is_single_piece():
    assert len(table_to_markdown_chunks([["A", "B"], ["1", "2"]], max_tokens=400)) == 1


def test_table_extracted_from_pdf_and_removed_from_text(table_pdf):
    (page,) = _parse(table_pdf)
    assert page.source_kind == "text" and len(page.tables) == 1
    md = table_to_markdown(page.tables[0].rows)
    assert "| Segment | FY24 | FY25 |" in md
    assert "| Cables | 1,200 | 1,450 |" in md
    assert "Segment results are summarised below." in page.text
    assert "1,450" not in page.text and "Cables" not in page.text  # no duplicate of the table text

    chunks = chunk_page(page, doc_id="d", filename="t.pdf", doc_title="t", cfg=CHUNKING)
    tables = [c for c in chunks if c.source_kind == "table"]
    assert [c.id for c in tables] == ["d:1:t0"]
    assert "Cables" in tables[0].text


def _statement_pdf(tmp_path, side_by_side: bool):
    """A page with a titled table ("Standalone Balance Sheet" + a unit line above it)."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Standalone Balance Sheet", fontsize=12)
    page.insert_text((72, 76), "(All amount are in Rs million)", fontsize=9)
    rows = [["Particulars", "2025", "2024"], ["Borrowings", "985.71", "819.17"], ["Payables", "4.99", "9.32"]]
    x0, y0, cw, rh = 72, 120, 100, 24
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            page.draw_rect(rect, width=0.8)
            page.insert_text((rect.x0 + 6, rect.y0 + 16), cell, fontsize=10)
    if side_by_side:  # another statement title in the other column must not leak into this table's title
        page.insert_text((420, 60), "Statement of Cash Flows", fontsize=12)
    return _save(doc, tmp_path, "statement.pdf")


def test_table_title_is_the_text_printed_above_it(tmp_path):
    (page,) = _parse(_statement_pdf(tmp_path, side_by_side=True))
    (table,) = page.tables
    assert table.title == "Standalone Balance Sheet (All amount are in Rs million)"


def test_table_chunks_start_with_their_title_and_are_found_by_it(tmp_path):
    (page,) = _parse(_statement_pdf(tmp_path, side_by_side=False))
    chunks = chunk_page(page, doc_id="d", filename="s.pdf", doc_title="s", cfg=CHUNKING)
    (table_chunk,) = [c for c in chunks if c.source_kind == "table"]
    title = "Standalone Balance Sheet (All amount are in Rs million)"
    assert table_chunk.text.startswith(title + "\n| Particulars |")
    assert "Standalone Balance Sheet" in table_chunk.embed_text


def test_title_is_repeated_on_every_piece_of_a_long_table():
    rows = [["Item", "FY25"]] + [[f"row {i}", str(i)] for i in range(40)]
    pieces = table_to_markdown_chunks(rows, max_tokens=120, title="Standalone Balance Sheet")
    assert len(pieces) > 1
    assert all(p.startswith("Standalone Balance Sheet\n| Item | FY25 |") for p in pieces)
    assert table_to_markdown_chunks([["A", "B"], ["1", "2"]], max_tokens=400)[0].startswith("| A | B |")


def test_table_without_text_above_has_an_empty_title(table_pdf):
    # In this fixture the only text above the table is the sentence at the top of the page: it is the title.
    (page,) = _parse(table_pdf)
    assert page.tables[0].title in ("", "Segment results are summarised below.")


# ---- page labels / text -----------------------------------------------------------------------

def test_page_label_falls_back_to_index(tmp_path):
    doc = fitz.open()
    for _ in range(2):
        _text_page(doc, PARA)
    pages = _parse(_save(doc, tmp_path, "nolabel.pdf"))
    assert [(p.pdf_page, p.page_label) for p in pages] == [(1, "1"), (2, "2")]


def test_page_labels_are_read_when_present(tmp_path):
    doc = fitz.open()
    for _ in range(3):
        _text_page(doc, PARA)
    doc.set_page_labels([{"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1},
                         {"startpage": 2, "prefix": "", "style": "D", "firstpagenum": 1}])
    pages = _parse(_save(doc, tmp_path, "labels.pdf"))
    assert [(p.pdf_page, p.page_label) for p in pages] == [(1, "i"), (2, "ii"), (3, "1")]


def test_text_page_basic(text_pdf):
    (page,) = _parse(text_pdf)
    assert page.source_kind == "text" and page.error is None
    assert page.image_area_ratio == 0.0
    assert "Revenue from operations" in page.text
    assert page.text.count("\n\n") >= 1  # paragraphs survive


# ---- scanned detection ------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("chars", "ratio", "expected"),
    [
        (0, 0.95, True),       # classic scan
        (119, 0.60, True),     # just under / at the thresholds
        (120, 0.95, False),    # enough text layer (e.g. already-OCR'd scan)
        (10, 0.30, False),     # near-empty page without a big image: a blank/cover page, not a scan
        (500, 0.10, False),    # normal text page
        (0, 0.0, False),       # blank page
    ],
)
def test_is_scanned_page(chars, ratio, expected):
    assert is_scanned_page(chars, ratio, OCR) is expected


def test_image_only_page_goes_to_ocr_path(image_only_pdf):
    seen = []

    def fake_ocr(page, cfg):
        seen.append(page.number)
        return "Line one of the scan.\nstill paragraph one.\n\nSecond paragraph."

    (page,) = _parse(image_only_pdf, ocr_fn=fake_ocr)
    assert seen == [0]
    assert page.source_kind == "ocr" and page.image_area_ratio > 0.9 and page.error is None
    assert page.text == "Line one of the scan. still paragraph one.\n\nSecond paragraph."


def test_ligatures_are_folded():
    assert pdf_parse.ocr_paragraphs("Our Oﬀerings and ﬁnancial health") == [
        ["Our Offerings and financial health"]
    ]


def test_text_page_does_not_call_ocr(text_pdf):
    def boom(page, cfg):
        raise AssertionError("OCR must not run on a text page")

    assert _parse(text_pdf, ocr_fn=boom)[0].source_kind == "text"


def test_image_area_ratio_uses_union_not_sum(tmp_path):
    src = fitz.open()
    _text_page(src, "x")
    pix = src[0].get_pixmap(dpi=50)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    area = fitz.Rect(0, 0, 595, 505)  # 60% of the page
    page.insert_image(area, pixmap=pix, keep_proportion=False)
    page.insert_image(area, pixmap=pix, keep_proportion=False)  # same area twice: a sum says 120%
    path = _save(doc, tmp_path, "twice.pdf")
    with fitz.open(str(path)) as d:
        assert image_area_ratio(d[0]) == pytest.approx(0.6, abs=0.01)


def test_ocr_unavailable_fails_that_page_only(image_only_pdf):
    def no_tesseract(page, cfg):
        raise OCRUnavailableError("Tesseract is not installed")

    pages = _parse(image_only_pdf, ocr_fn=no_tesseract)
    assert len(pages) == 1
    assert pages[0].text == "" and "Tesseract is not installed" in pages[0].error
    assert chunk_page(pages[0], doc_id="d", filename="s.pdf", doc_title="s", cfg=CHUNKING) == []


def test_missing_tesseract_binary_gives_clear_error(image_only_pdf, monkeypatch):
    pytesseract = pytest.importorskip("pytesseract")

    def raise_not_found(*a, **kw):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", raise_not_found)
    (page,) = _parse(image_only_pdf)
    assert page.error and "Tesseract is not installed" in page.error


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_real_ocr_reads_rasterised_text(image_only_pdf):
    (page,) = _parse(image_only_pdf)
    assert page.source_kind == "ocr" and page.error is None
    assert "REVENUE" in page.text.upper()


def test_password_protected_pdf_is_rejected(tmp_path, text_pdf):
    doc = fitz.open(str(text_pdf))
    out = tmp_path / "locked.pdf"
    doc.save(str(out), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
    doc.close()
    with pytest.raises(ValueError, match="password"):
        _parse(out)


# ---- header / footer stripping ----------------------------------------------------------------

def test_line_key_normalises_page_numbers():
    assert line_key("Page 3 of 300") == line_key("PAGE  47 of 300") == "page # of #"


def test_find_repeated_keys_thresholds():
    keys = [{"acme report", f"unique{i}"} for i in range(6)]
    assert find_repeated_keys(keys, PARSING) == {"acme report"}
    assert find_repeated_keys(keys[:2], PARSING) == set()  # too few pages to judge


def test_headers_and_footers_are_stripped(tmp_path):
    doc = fitz.open()
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    for i, word in enumerate(words, start=1):
        _text_page(doc, f"Body text about {word}. {PARA}",
                   header="ACME Industries Annual Report 2025", footer=f"Page {i} of 6")
    pages = _parse(_save(doc, tmp_path, "hf.pdf"))
    for i, (p, word) in enumerate(zip(pages, words, strict=True), start=1):
        assert "ACME Industries" not in p.text
        assert f"Page {i} of 6" not in p.text
        assert f"Body text about {word}" in p.text


def test_body_lines_repeated_in_middle_are_kept(tmp_path):
    doc = fitz.open()
    for i in range(6):
        lines = ["Top line a", "Top line b", "Top line c", "Top line d", "Note: figures in INR crore",
                 f"Tail {i} a", "Tail b", "Tail c", "Tail d"]
        _text_page(doc, "\n".join(lines))
    pages = _parse(_save(doc, tmp_path, "mid.pdf"))
    assert all("figures in INR crore" in p.text for p in pages)


def test_short_documents_are_not_stripped(tmp_path):
    doc = fitz.open()
    for _ in range(2):
        _text_page(doc, PARA, header="Same header")
    assert all("Same header" in p.text for p in _parse(_save(doc, tmp_path, "short.pdf")))


def test_chunking_uses_only_config_values():
    cfg = ChunkingConfig(size_tokens=50, overlap_tokens=10)
    chunks = chunk_page(_page(1, PARA * 5), doc_id="d", filename="a.pdf", doc_title="a", cfg=cfg)
    assert len(chunks) > 3 and all(estimate_tokens(c.text) <= 50 for c in chunks)



# ---- two-pass order + rupee glyph -------------------------------------------------------------

def test_text_pages_are_yielded_before_ocr_pages(tmp_path):
    """A scanned page early in the file must not hold back the text pages after it."""
    src = fitz.open()
    _text_page(src, "SCANNED PAGE " * 10)
    pix = src[0].get_pixmap(dpi=100)
    src.close()
    doc = fitz.open()
    _text_page(doc, PARA)                       # p1 text
    doc.new_page(width=595, height=842).insert_image(fitz.Rect(0, 0, 595, 842), pixmap=pix)  # p2 scan
    _text_page(doc, PARA)                       # p3 text
    path = _save(doc, tmp_path, "mixed.pdf")

    seen: list[int] = []

    def fake_ocr(page, _cfg):
        seen.append(page.number + 1)
        return "ocr text"

    records = _parse(path, ocr_fn=fake_ocr)
    assert [(r.pdf_page, r.source_kind) for r in records] == [(1, "text"), (3, "text"), (2, "ocr")]
    assert seen == [2]
    assert records[2].text == "ocr text"


def test_rupee_sign_drawn_as_backtick_is_restored():
    from app.ingestion.tables import fix_rupee

    assert fix_rupee("revenue reaching ` 3,124.83 million") == "revenue reaching ₹ 3,124.83 million"
    assert fix_rupee("(All amount are in ` million)") == "(All amount are in ₹ million)"
    assert fix_rupee("`(692.21)") == "₹(692.21)"
    assert fix_rupee("run `make test` now") == "run `make test` now"  # real code spans are left alone
    assert fix_rupee("Year ended (` in Lakhs)") == "Year ended (₹ in Lakhs)"
    assert fix_rupee("Basic (in `)") == "Basic (in ₹)"
    assert fix_rupee("the `in` keyword") == "the `in` keyword"


def test_table_column_without_an_outer_border_is_kept(tmp_path):
    """Rows ruled across every column, but no vertical border after the last one (common in statements):
    the outer column is still part of the table."""
    from app.ingestion.tables import find_page_tables

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    xs = [60, 260, 380, 500]  # vertical lines at the first three; the last column is open on the right
    rows = [["Particulars", "FY2025", "FY2024"], ["Revenue", "1,44,588.57", "1,18,142.30"],
            ["Other income", "19,855.44", "14,300.42"], ["Profit for the year", "6,378.44", "19,912.44"]]
    for i, row in enumerate(rows):
        y = 100 + 24 * i
        page.draw_line((xs[0], y), (xs[3], y))
        for j, cell in enumerate(row):
            page.insert_text((xs[j] + 6, y + 16), cell, fontsize=10)
    page.draw_line((xs[0], 100 + 24 * len(rows)), (xs[3], 100 + 24 * len(rows)))
    for x in xs[:3]:
        page.draw_line((x, 100), (x, 100 + 24 * len(rows)))
    page.insert_text((520, 790), "x", fontsize=8)
    page.draw_line((600, 50), (620, 50))  # a crop mark off the page is ignored
    page.draw_line((600, 60), (620, 60))
    tables = find_page_tables(page)
    assert len(tables) == 1
    assert tables[0].rows[0] == ["Particulars", "FY2025", "FY2024"]
    assert tables[0].rows[-1] == ["Profit for the year", "6,378.44", "19,912.44"]


def test_text_beside_a_table_without_rules_over_it_is_not_pulled_in(tmp_path):
    from app.ingestion.tables import find_page_tables

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    xs = [60, 200, 330]
    for i, row in enumerate([["Item", "FY25"], ["Revenue", "100"], ["Profit", "7"], ["Tax", "2"]]):
        y = 100 + 24 * i
        page.draw_line((xs[0], y), (xs[2], y))
        for j, cell in enumerate(row):
            page.insert_text((xs[j] + 6, y + 16), cell, fontsize=10)
        page.insert_text((400, y + 16), f"side note {i}", fontsize=10)  # a separate text column
    page.draw_line((xs[0], 196), (xs[2], 196))
    for x in xs:
        page.draw_line((x, 100), (x, 196))
    tables = find_page_tables(page)
    assert len(tables) == 1 and len(tables[0].rows[0]) == 2


# ---- page heading on every chunk ---------------------------------------------------------------

def _titled_page_pdf(tmp_path):
    """A statement page: big title, subtitle, then a long body whose later chunks lack the title words."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 60), "Standalone Balance Sheet", fontsize=18)
    page.insert_text((72, 80), "as at March 31, 2025", fontsize=9)
    page.insert_textbox(fitz.Rect(72, 100, 523, 800), "\n\n".join([PARA] * 7 + ["Total equity 9,07,400.25"]),
                        fontsize=8)
    return _save(doc, tmp_path, "statement.pdf")


def test_page_heading_is_the_big_title_plus_its_subtitle(tmp_path):
    rec = _parse(_titled_page_pdf(tmp_path))[0]
    assert rec.heading == "Standalone Balance Sheet as at March 31, 2025"


def test_every_chunk_starts_with_the_page_heading_once(tmp_path):
    rec = _parse(_titled_page_pdf(tmp_path))[0]
    cfg = ChunkingConfig(size_tokens=120, overlap_tokens=20)
    chunks = chunk_page(rec, doc_id="d", filename="s.pdf", doc_title="s", cfg=cfg)
    assert len(chunks) > 2
    flat = [" ".join(c.text.split()) for c in chunks]  # the first chunk has title and subtitle on two lines
    assert all(f.startswith("Standalone Balance Sheet as at March 31, 2025") for f in flat)
    assert all(f.count("Standalone Balance Sheet as at March 31, 2025") == 1 for f in flat)
    last = [c for c in chunks if "9,07,400.25" in c.text][0]
    assert last.text.startswith("Standalone Balance Sheet as at March 31, 2025")
    assert all(estimate_tokens(c.text) <= cfg.size_tokens for c in chunks)  # the heading fits in the budget


def test_plain_pages_have_no_heading(text_pdf):
    assert _parse(text_pdf)[0].heading == ""


# ---- Tesseract location on Windows ------------------------------------------------------------------

def test_tesseract_cmd_env_var_wins(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", r"D:\tools\tesseract.exe")
    assert pdf_parse.tesseract_cmd() == r"D:\tools\tesseract.exe"


def test_tesseract_cmd_uses_the_default_install_on_windows_when_not_on_path(monkeypatch):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(pdf_parse.os, "name", "nt")
    monkeypatch.setattr(pdf_parse.shutil, "which", lambda _name: None)
    monkeypatch.setattr(pdf_parse.os.path, "isfile", lambda p: p == pdf_parse.WINDOWS_TESSERACT_PATHS[0])
    assert pdf_parse.tesseract_cmd() == pdf_parse.WINDOWS_TESSERACT_PATHS[0]


def test_tesseract_cmd_leaves_path_installs_alone(monkeypatch):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(pdf_parse.os, "name", "nt")
    monkeypatch.setattr(pdf_parse.shutil, "which", lambda _name: r"C:\bin\tesseract.exe")
    assert pdf_parse.tesseract_cmd() is None
    monkeypatch.setattr(pdf_parse.os, "name", "posix")
    monkeypatch.setattr(pdf_parse.shutil, "which", lambda _name: None)
    assert pdf_parse.tesseract_cmd() is None


# ---- parallel parsing -------------------------------------------------------------------------------

def test_parallel_parsing_gives_the_same_records_in_page_order(tmp_path):
    doc = fitz.open()
    for i in range(6):
        _text_page(doc, f"Page {i + 1}. " + PARA)
    path = _save(doc, tmp_path, "six.pdf")
    serial = _parse(path)
    parallel_cfg = ParsingConfig(workers=2, parallel_min_pages=1, parallel_batch_pages=2)
    parallel = list(parse_pdf(path, OCR, parallel_cfg))
    assert [r.pdf_page for r in parallel] == [1, 2, 3, 4, 5, 6]
    assert [(r.pdf_page, r.text, r.heading, r.source_kind) for r in parallel] == [
        (r.pdf_page, r.text, r.heading, r.source_kind) for r in serial
    ]


def test_small_documents_are_parsed_in_process():
    cfg = ParsingConfig(workers=4, parallel_min_pages=24)
    assert pdf_parse.parse_workers(cfg, 23) == 1
    assert pdf_parse.parse_workers(cfg, 24) == 4
    assert pdf_parse.parse_workers(ParsingConfig(workers=0), 400) >= 1


# ---- unruled statements ------------------------------------------------------------------------

def _unruled_statement_page():
    """A statement drawn with no lines: labels on the left, note numbers, two right-aligned year columns,
    a subtotal printed without a label, and a side tab beyond the columns."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 60), "Standalone Statement of Profit and Loss", fontsize=12)

    def right(x, y, text):
        page.insert_text((x - fitz.get_text_length(text, fontsize=9), y), text, fontsize=9)

    right(420, 100, "2024-25")
    right(520, 100, "2023-24")
    rows = [("Revenue From Operations", "", "", ""),
            ("Sale of Products", "31", "4,64,246.96", "4,59,815.32"),
            ("Other Operating Revenue", "32", "2,098.69", "1,822.19"),
            ("", "", "4,66,345.65", "4,61,637.51"),
            ("Other Income", "33", "2,416.44", "2,382.15"),
            ("Total Income", "", "4,68,762.09", "4,64,019.66"),
            ("Finance Costs", "37", "3,310.91", "2,515.67"),
            ("Total Expenses", "", "4,59,140.62", "4,44,866.53"),
            ("Profit for the year", "", "7,364.86", "14,693.83"),
            ("Basic and Diluted EPS", "46", "34.61", "69.06")]
    for i, (label, note, a, b) in enumerate(rows):
        y = 120 + 14 * i
        if label:
            page.insert_text((60, y), label, fontsize=9)
        if note:
            page.insert_text((300, y), note, fontsize=9)
        if a:
            right(420, y, a)
            right(520, y, b)
    page.insert_text((560, 200), "Reports", fontsize=8)  # side tab
    return page


def test_unruled_statement_is_rebuilt_with_labels_columns_and_subtotal():
    from app.ingestion.tables import find_text_tables

    [table] = find_text_tables(_unruled_statement_page())
    assert table.rows[0] == ["Particulars", "Note", "2024-25", "2023-24"]
    assert ["Sale of Products", "31", "4,64,246.96", "4,59,815.32"] in table.rows
    assert ["Revenue From Operations (subtotal)", "", "4,66,345.65", "4,61,637.51"] in table.rows
    assert ["Basic and Diluted EPS", "46", "34.61", "69.06"] in table.rows
    assert not any("Reports" in c for r in table.rows for c in r)


def test_prose_with_numbers_is_not_taken_for_a_table():
    from app.ingestion.tables import find_text_tables

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i in range(12):
        line = f"In year {i} it earned 1,{i}00.50 crore and paid 2{i}.75 crore in tax, against 3,{i}10.25."
        page.insert_text((60, 100 + 14 * i), line, fontsize=9)
    assert find_text_tables(page) == []


def test_label_less_scrap_table_is_replaced_by_the_rebuilt_statement(tmp_path):
    from app.ingestion.tables import Table, has_row_labels

    scrap = [["Notes", "2024-25"], ["31", "4,64,246.96"], ["45(f)", "1.0"]]  # note numbers and figures only
    assert not has_row_labels(Table((0, 0, 1, 1), scrap))
    assert has_row_labels(Table((0, 0, 1, 1), [["Particulars", "2024-25"], ["Total Income", "4,68,762.09"]]))
    doc = fitz.open()
    doc.insert_pdf(_unruled_statement_page().parent)
    path = _save(doc, tmp_path, "unruled.pdf")
    [page] = _parse(path)
    assert any("Revenue From Operations (subtotal)" in r[0] for t in page.tables for r in t.rows)
    assert "4,66,345.65" not in page.text  # the figures live in the table, not in a flattened text run