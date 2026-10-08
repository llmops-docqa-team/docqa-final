"""'Show in PDF': figure terms for a citation, the cropped highlighted page, and the endpoint."""

from __future__ import annotations

import pymupdf as fitz
import pytest

from app.answering.document import DocumentAnswerer
from app.answering.highlight import PageOutOfRange, highlight_terms, render_highlight
from app.config import load_settings
from tests.conftest import upload, wait_for
from tests.test_answer_document import FakeLLM, FakeRetriever, chunk, ok, result

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def rows_pdf(tmp_path):
    path = tmp_path / "rows.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 200), "Finance Cost 94.90 171.40", fontsize=11)
    page.insert_text((72, 230), "Other Income 500.49 359.49", fontsize=11)
    page.insert_text((72, 260), "Total Expenses 1,094.90 800.00", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------- render_highlight


def test_finds_the_figure_and_returns_a_cropped_png(rows_pdf):
    png, n = render_highlight(rows_pdf, 1, ["94.90"])
    assert n == 1 and png.startswith(PNG)
    assert fitz.Pixmap(png).height < 842 * 1.6 * 0.7  # cropped, not the whole page


def test_does_not_match_inside_a_longer_number(rows_pdf):
    # "1,094.90" contains "094.90"-like digits; only the whole word 94.90 counts
    assert render_highlight(rows_pdf, 1, ["94.90"])[1] == 1


def test_unknown_term_renders_the_plain_page(rows_pdf):
    png, n = render_highlight(rows_pdf, 1, ["777.77"])
    assert n == 0 and png.startswith(PNG)
    assert fitz.Pixmap(png).height == pytest.approx(842 * 1.6, abs=2)


def test_page_out_of_range(rows_pdf):
    with pytest.raises(PageOutOfRange):
        render_highlight(rows_pdf, 5, ["94.90"])


def test_several_places_prefer_the_row_that_shares_words_with_the_chunk(tmp_path):
    path = tmp_path / "two.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Finance Cost 94.90", fontsize=11)
    page.insert_text((72, 300), "Depreciation 94.90", fontsize=11)
    doc.save(str(path))
    doc.close()
    assert render_highlight(path, 1, ["94.90"])[1] == 2
    assert render_highlight(path, 1, ["94.90"], context="Finance Cost for the year")[1] == 1


# ---------------------------------------------------------------- highlight_terms


def test_figure_present_in_the_chunk_is_a_term_as_printed():
    text = "Finance Cost 94.90 171.40\nRevenue 1,097.36 615.30"
    assert highlight_terms("Finance cost was ₹94.90 million.", [], text) == ["94.90"]
    assert highlight_terms("Revenue was 1097.36.", [], text) == ["1,097.36"]  # the way the chunk prints it


def test_a_figure_not_in_the_chunk_is_not_a_term():
    assert highlight_terms("Finance cost was 94.90.", [], "Other Income 500.49") == []


def test_computed_figure_contributes_its_operands():
    text = "Revenue 1,097.36 615.30"
    got = highlight_terms("The increase was 482.06.", ["482.06 = 1097.36 − 615.30"], text)
    assert got == ["1,097.36", "615.30"]


def test_ratio_operands_skip_the_hundred():
    text = "Revenue 110.00 120.00 note 100"
    got = highlight_terms("Growth was 9.09%.", ["9.09 = (120.00 − 110.00) / 110.00 × 100"], text)
    assert got == ["120.00", "110.00"]


def test_citation_with_most_terms_is_primary():
    a = chunk(1, "Finance Cost 94.90 and Revenue 1,097.36", page=34)
    b = chunk(2, "Finance Cost 94.90", page=35)
    answer = "Finance cost was 94.90 and revenue 1,097.36."
    llm = FakeLLM(ok(answer, ("S2", "S1")))
    out = DocumentAnswerer(FakeRetriever(result([b, a])), llm, load_settings()).answer("q")
    by_page = {c.pdf_page: c for c in out.citations}
    assert by_page[34].highlight_terms == ["94.90", "1,097.36"] and by_page[34].primary
    assert by_page[35].highlight_terms == ["94.90"] and not by_page[35].primary


# ---------------------------------------------------------------- endpoint


def test_endpoint(make_client, embedder, rows_pdf):
    with make_client(embedder) as client:
        doc = upload(client, rows_pdf, "rows.pdf").json()
        wait_for(client, doc["id"], ("READY", "PARTIAL"))
        base = f"/documents/{doc['id']}/pages/1/highlight"

        r = client.get(base, params={"term": ["94.90", "500.49"]})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.headers["x-highlight-matches"] == "2" and r.content.startswith(PNG)

        assert client.get("/documents/nope/pages/1/highlight").status_code == 404
        assert client.get(f"/documents/{doc['id']}/pages/9/highlight").status_code == 422
        assert client.get(f"/documents/{doc['id']}/pages/0/highlight").status_code == 422
        assert client.get(base, params={"term": "<script>"}).status_code == 422
        assert client.get(base, params={"term": ["1.50"] * 13}).status_code == 422


# ---------------------------------------------------------------- UI client and caption


def test_client_fetches_the_image_and_the_match_count():
    from ui.api_client import ApiClient
    from ui.formatting import highlight_caption

    class Img:
        status_code, content, headers = 200, b"\x89PNGdata", {"X-Highlight-Matches": "2"}

    class Session:
        def request(self, method, url, **kw):
            self.call = (method, url, kw)
            return Img()

    s = Session()
    assert ApiClient("http://api", s).page_highlight("d1", 34, ["94.90", "1,097.36"], "Finance Cost") == (
        b"\x89PNGdata",
        2,
    )
    assert s.call[1] == "http://api/documents/d1/pages/34/highlight"
    assert s.call[2]["params"] == {"term": ["94.90", "1,097.36"], "ctx": "Finance Cost"}
    assert "highlighted" in highlight_caption(2, "table")
    assert "not found" in highlight_caption(0, "text")
    assert "Scanned" in highlight_caption(0, "ocr")
