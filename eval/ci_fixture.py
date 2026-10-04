"""Synthetic corpus for the CI retrieval gate.

The real reports are private, so CI cannot use them. This module generates two small, fictional annual
reports (Halden Power Cables, FY25 and FY26) as PDFs. They are generated, not committed (`*.pdf` is
git-ignored, and there is no licence question). Generation is deterministic: the same code always gives
the same pages, so gold pages in `eval/fixtures/ci_questions.jsonl` stay valid.

The two reports are deliberately near-twins (same sections and wording, different figures, FY25 figures
repeated as comparatives in FY26), so retrieval has to pick the right *year* as well as the right page.
Layout: pdf pages 1-2 are front matter (labels i, ii), body pages are pdf 3..17 with printed labels 1..15,
and every page has a running footer (exercises the header/footer stripper).

    python -m eval.ci_fixture <out_dir>      # write the PDFs, print the doc map
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "fixtures" / "ci_questions.jsonl"
DOCS_PATH = ROOT / "fixtures" / "ci_docs.yaml"
BASELINE_PATH = ROOT / "baselines" / "ci_retrieval.json"

# eval doc key -> generated filename (the filename becomes the chunk title, e.g. "Halden AR FY25")
DOCS = {"ci_fy25": "Halden_AR_FY25.pdf", "ci_fy26": "Halden_AR_FY26.pdf"}
FRONT_MATTER_PAGES = 2   # body page n is pdf page n + 2

FACTS: dict[str, dict[str, str]] = {
    "fy24": {
        "revenue": "4,280.1", "ebitda": "598.9", "pat": "330.5", "eps": "22.03", "net_debt": "784.6",
        "cables": "2,603.5", "transformers": "1,106.2", "switchgear": "570.4",
        "assets": "4,512.8", "net_worth": "2,046.9", "borrowings": "1,102.7", "cash": "318.1",
    },
    "fy25": {
        "year": "FY25", "revenue": "4,826.4", "growth": "12.8", "ebitda": "702.3", "margin": "14.6",
        "pat": "391.8", "eps": "26.12", "net_debt": "612.0", "dps": "6.50", "record_date": "18 July 2025",
        "employees": "3,420", "plants": "Hosur, Vadodara and Nashik",
        "cables": "2,910.2", "transformers": "1,244.7", "switchgear": "671.5",
        "cables_m": "16.2", "transformers_m": "12.1", "switchgear_m": "9.8",
        "copper": "8,210", "capex": "318.0", "capex_next": "520",
        "ltifr": "0.42", "emissions": "61,400", "water": "1.9", "csr": "8.9",
        "directors": "nine", "independent": "five", "meetings": "six",
        "attrition": "11.8", "training": "28",
        "kam": "revenue recognition on long-term transformer contracts",
        "assets": "4,980.5", "net_worth": "2,305.4", "borrowings": "934.2", "cash": "322.2",
        "promoters": "58.4", "fii": "14.2", "dii": "17.1", "public": "10.3",
        "focus": "grid modernisation orders and the new extra-high-voltage cable line at Vadodara",
        "milestone": "commissioned a 400 kV cable line at Vadodara",
        "transformer_driver": "orders from state transmission utilities for 220 kV power transformers",
        "switchgear_driver": "demand for gas-insulated switchgear from metro rail projects",
        "risk_fx": "a 3% depreciation of the rupee against the dollar raised our imported input costs",
    },
    "fy26": {
        "year": "FY26", "revenue": "5,412.9", "growth": "12.2", "ebitda": "835.7", "margin": "15.4",
        "pat": "468.2", "eps": "31.21", "net_debt": "438.5", "dps": "8.00", "record_date": "17 July 2026",
        "employees": "3,785", "plants": "Hosur, Vadodara, Nashik and Raipur",
        "cables": "3,198.0", "transformers": "1,432.6", "switchgear": "782.3",
        "cables_m": "16.9", "transformers_m": "13.4", "switchgear_m": "11.0",
        "copper": "9,040", "capex": "455.0", "capex_next": "610",
        "ltifr": "0.31", "emissions": "57,900", "water": "1.7", "csr": "10.4",
        "directors": "ten", "independent": "six", "meetings": "seven",
        "attrition": "10.2", "training": "34",
        "kam": "revenue recognition on long-term transformer contracts and valuation of copper inventory",
        "assets": "5,622.3", "net_worth": "2,701.0", "borrowings": "801.6", "cash": "363.1",
        "promoters": "58.4", "fii": "16.9", "dii": "15.8", "public": "8.9",
        "focus": "the commissioning of our Raipur plant and our first export orders from the Gulf region",
        "milestone": "commissioned the Raipur plant for medium-voltage cables",
        "transformer_driver": (
            "orders from renewable energy developers for 132 kV generator step-up transformers"
        ),
        "switchgear_driver": "demand for ring main units from data centre operators",
        "risk_fx": "a 2% depreciation of the rupee against the dollar raised our imported input costs",
    },
}

# ---- body pages: (kind, builder). Builders get (F, P, short year, F-key) and return text ------------------


def _chair(F, P, yy):
    return (
        "Chairman's Letter\n\n"
        "Dear Shareholders,\n\n"
        f"FY{yy} was a year of disciplined growth for Halden Power Cables. Revenue from operations rose "
        f"{F['growth']}% to Rs {F['revenue']} crore, helped by {F['focus']}. Our order book remains healthy "
        "and our customers in transmission, distribution, renewables and railways continue to invest.\n\n"
        "The Board remains committed to prudent capital allocation and to returning surplus cash to "
        "shareholders. I thank our employees, customers and partners for their trust. "
        "Ramesh Iyer, Chairman."
    )


def _overview(F, P, yy):
    return (
        "Business Overview\n\n"
        "Halden Power Cables Limited manufactures power cables, power transformers and switchgear for "
        "utilities and industry. We operate three segments: Cables, Transformers and Switchgear.\n\n"
        f"Our manufacturing facilities are located at {F['plants']}. At the end of FY{yy} the company "
        f"employed {F['employees']} people. The company is headquartered in Pune and its shares are "
        "listed on the National Stock Exchange and the BSE."
    )


def _highlights(F, P, yy):
    intro = (
        f"Financial Highlights\n\nThe table summarises our consolidated performance for FY{yy} against the "
        "previous year. Figures are in Rs crore unless stated otherwise."
    )
    rows = [
        ["Particulars", f"FY{yy}", f"FY{int(yy) - 1}"],
        ["Revenue from operations", F["revenue"], P["revenue"]],
        ["EBITDA", F["ebitda"], P["ebitda"]],
        ["Profit after tax", F["pat"], P["pat"]],
        ["Earnings per share (Rs)", F["eps"], P["eps"]],
        ["Net debt", F["net_debt"], P["net_debt"]],
    ]
    return intro, rows


def _cables(F, P, yy):
    return (
        "Management Discussion: Cables Segment\n\n"
        f"Cables is our largest segment, contributing Rs {F['cables']} crore of revenue in FY{yy}. "
        f"During the year we {F['milestone']}. Demand from utilities and the real estate sector stayed firm, "
        f"and the average copper price was USD {F['copper']} per tonne.\n\n"
        "We pass through most copper price movements to customers with a short lag, so the segment's "
        "margin is driven mainly by product mix and plant utilisation."
    )


def _transformers(F, P, yy):
    return (
        "Management Discussion: Transformers Segment\n\n"
        f"The Transformers segment earned Rs {F['transformers']} crore in FY{yy}, supported by "
        f"{F['transformer_driver']}. Long-term contracts are recognised over time as work progresses, and "
        "the segment carried a strong order book into the next year.\n\n"
        "Lead times for core steel and insulating oil improved during the year, which helped us deliver "
        "units on schedule and avoid liquidated damages."
    )


def _switchgear(F, P, yy):
    return (
        "Management Discussion: Switchgear Segment\n\n"
        f"Our Switchgear segment reported revenue of Rs {F['switchgear']} crore in FY{yy}, driven by "
        f"{F['switchgear_driver']}. It is the smallest segment but is growing quickly from a low base.\n\n"
        "We continue to invest in product certification and in a wider distributor network to improve "
        "reach in smaller cities."
    )


def _segments(F, P, yy):
    intro = (
        f"Segment Results\n\nRevenue and EBIT margin by business segment for FY{yy}. Revenue is in "
        "Rs crore; margin is segment EBIT as a percentage of segment revenue."
    )
    rows = [
        ["Segment", f"Revenue FY{yy}", f"Revenue FY{int(yy) - 1}", f"EBIT margin FY{yy} (%)"],
        ["Cables", F["cables"], P["cables"], F["cables_m"]],
        ["Transformers", F["transformers"], P["transformers"], F["transformers_m"]],
        ["Switchgear", F["switchgear"], P["switchgear"], F["switchgear_m"]],
    ]
    return intro, rows


def _balance(F, P, yy):
    intro = (
        f"Balance Sheet Summary\n\nKey balance sheet items as at 31 March 20{yy}, in Rs crore. "
        "Net debt is borrowings less cash and cash equivalents."
    )
    rows = [
        ["Particulars", f"31 Mar 20{yy}", f"31 Mar 20{int(yy) - 1}"],
        ["Total assets", F["assets"], P["assets"]],
        ["Net worth", F["net_worth"], P["net_worth"]],
        ["Borrowings", F["borrowings"], P["borrowings"]],
        ["Cash and cash equivalents", F["cash"], P["cash"]],
    ]
    return intro, rows


def _cashflow(F, P, yy):
    return (
        "Cash Flow and Capital Allocation\n\n"
        f"Capital expenditure in FY{yy} was Rs {F['capex']} crore, mostly for capacity expansion and "
        f"automation. We plan to spend about Rs {F['capex_next']} crore in the coming year, funded from "
        "internal accruals.\n\n"
        f"Net debt fell to Rs {F['net_debt']} crore by year end as operating cash flow comfortably covered "
        "investment and dividends. Working capital days were broadly stable."
    )


def _risk(F, P, yy):
    return (
        "Risk Management\n\n"
        "Our principal risks are volatility in copper and aluminium prices, currency movements, "
        "dependence on a few large utility customers, and execution delays on long-term contracts.\n\n"
        f"During FY{yy}, {F['risk_fx']}. We hedge a large part of our commodity exposure and review "
        "our customer concentration every quarter. The Risk Management Committee reports to the Board."
    )


def _governance(F, P, yy):
    return (
        "Corporate Governance\n\n"
        f"At the end of FY{yy} the Board had {F['directors']} directors, of whom {F['independent']} were "
        f"independent. The Board met {F['meetings']} times during the year and attendance was above 90%.\n\n"
        "The Audit Committee, the Nomination and Remuneration Committee and the Stakeholders' Relationship "
        "Committee are chaired by independent directors. Sunita Kapoor is the Managing Director."
    )


def _esg(F, P, yy):
    return (
        "Sustainability\n\n"
        f"Our lost-time injury frequency rate was {F['ltifr']} per million hours worked in FY{yy}. "
        f"Scope 1 and 2 greenhouse gas emissions were {F['emissions']} tonnes of CO2 equivalent, and "
        f"water withdrawal was {F['water']} million kilolitres.\n\n"
        f"We spent Rs {F['csr']} crore on corporate social responsibility, mainly on rural schools and "
        "primary healthcare near our plants."
    )


def _people(F, P, yy):
    return (
        "People and Culture\n\n"
        f"Employee attrition was {F['attrition']}% in FY{yy}. Each employee received an average of "
        f"{F['training']} hours of training during the year, with a focus on safety and technical skills.\n\n"
        "We run an apprenticeship programme with local technical institutes and promote from within "
        "wherever possible."
    )


def _audit(F, P, yy):
    return (
        "Independent Auditors' Report: Key Audit Matters\n\n"
        "Rao & Menon LLP, Chartered Accountants, have audited the financial statements and issued an "
        "unmodified opinion.\n\n"
        f"The key audit matters for FY{yy} were {F['kam']}. Our audit procedures included testing "
        "internal controls and examining a sample of contracts and inventory counts."
    )


def _dividend(F, P, yy):
    intro = (
        f"Dividend and Shareholder Information\n\nThe Board recommended a dividend of Rs {F['dps']} per "
        f"equity share for FY{yy}. The record date for the dividend is {F['record_date']}.\n\n"
        "The table shows the shareholding pattern as at 31 March."
    )
    rows = [
        ["Category", "Shareholding (%)"],
        ["Promoters", F["promoters"]],
        ["Foreign institutional investors", F["fii"]],
        ["Domestic institutions", F["dii"]],
        ["Public", F["public"]],
    ]
    return intro, rows


# Body page n (1-based printed label) -> builder. Page numbers are referenced by ci_questions.jsonl.
BODY: list[tuple[str, Callable]] = [
    ("text", _chair), ("text", _overview), ("table", _highlights), ("text", _cables),
    ("text", _transformers), ("text", _switchgear), ("table", _segments), ("table", _balance),
    ("text", _cashflow), ("text", _risk), ("text", _governance), ("text", _esg), ("text", _people),
    ("text", _audit), ("table", _dividend),
]

_FOOTER_Y = 815


def _draw_table(page: fitz.Page, rows: list[list[str]], y0: float) -> None:
    x0, rh = 72.0, 24.0
    widths = [210.0] + [(451.0 - 210.0) / (len(rows[0]) - 1)] * (len(rows[0]) - 1)
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            rect = fitz.Rect(x, y0 + r * rh, x + widths[c], y0 + (r + 1) * rh)
            page.draw_rect(rect, width=0.8)
            page.insert_text((rect.x0 + 5, rect.y0 + 16), cell, fontsize=9)
            x += widths[c]


def _heading(built) -> str:
    text = built[0] if isinstance(built, tuple) else built
    return text.split("\n", 1)[0]


def build_report(path: Path, year_key: str) -> None:
    """Write one report. `year_key` is "fy25" or "fy26"."""
    F = FACTS[year_key]
    P = FACTS["fy" + str(int(year_key[2:]) - 1)]
    yy = year_key[2:]
    title = f"Annual Report {F['year']}"

    doc = fitz.open()

    def new_page(footer_no: int) -> fitz.Page:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, _FOOTER_Y), f"Halden Power Cables Limited | {title} | {footer_no}", fontsize=8)
        return page

    cover = new_page(1)
    cover.insert_textbox(
        fitz.Rect(72, 250, 523, 450),
        f"HALDEN POWER CABLES LIMITED\n\n{title}\n\nPowering a connected India",
        fontsize=18,
    )
    contents = new_page(2)
    toc = "Contents\n\n" + "\n".join(
        f"{label}. {_heading(builder(F, P, yy))}" for label, (_, builder) in enumerate(BODY, start=1)
    )
    # A contents page names every section, so it is a realistic distractor for section-name queries.
    contents.insert_textbox(fitz.Rect(72, 80, 523, 700), toc, fontsize=10)

    for n, (kind, builder) in enumerate(BODY, start=1):
        page = new_page(FRONT_MATTER_PAGES + n)
        if kind == "text":
            page.insert_textbox(fitz.Rect(72, 80, 523, 700), builder(F, P, yy), fontsize=10.5)
        else:
            intro, rows = builder(F, P, yy)
            page.insert_textbox(fitz.Rect(72, 80, 523, 260), intro, fontsize=10.5)
            _draw_table(page, rows, 280)

    # Printed labels: i, ii for the front matter, then 1..15 for the body.
    doc.set_page_labels(
        [
            {"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1},
            {"startpage": FRONT_MATTER_PAGES, "prefix": "", "style": "D", "firstpagenum": 1},
        ]
    )
    doc.save(str(path), garbage=3, deflate=True)
    doc.close()


def build_corpus(out_dir: str | Path) -> dict[str, str]:
    """Write both reports into `out_dir`. Returns {doc key: filename}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, filename in DOCS.items():
        build_report(out / filename, key.removeprefix("ci_"))
    return dict(DOCS)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "ci_corpus")
    for k, f in build_corpus(target).items():
        print(f"{k}: {target / f}")
