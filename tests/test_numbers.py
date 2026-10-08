"""Number normaliser and numeric grounding check (all deterministic, no LLM)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.answering.numbers import check_numbers, extract_numbers, parse_number


@pytest.mark.parametrize(
    "token, expected",
    [
        ("12,563", "12563"),
        ("1,25,630", "125630"),  # Indian grouping
        ("12,34,56,789", "123456789"),
        ("1,234,567", "1234567"),  # Western grouping
        ("1,25,630.50", "125630.50"),
        ("4.5", "4.5"),
        ("0.75", "0.75"),
        ("100", "100"),
        ("12,563,", "12563"),  # trailing comma from prose
    ],
)
def test_parse_number(token, expected):
    assert parse_number(token) == Decimal(expected)


@pytest.mark.parametrize("token", ["1,2", "12,34", "1234,567", "1,2345"])
def test_parse_number_rejects_bad_grouping(token):
    assert parse_number(token) is None


def values(text: str, **kw) -> list[Decimal]:
    return [q.value for q in extract_numbers(text, **kw)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12,563", ["12563"]),
        ("₹1,25,630 crore", ["1256300000000"]),
        ("Rs. 1,25,630 Cr", ["1256300000000"]),
        ("4.5%", ["4.5"]),
        ("4.5 per cent", ["4.5"]),
        ("12.5 million", ["12500000"]),
        ("USD 3.2 billion", ["3200000000"]),
        ("2.5 lakh", ["250000"]),
        ("₹ 45 lakhs", ["4500000"]),
        ("1.2 bn and 300 mn", ["1200000000", "300000000"]),
        ("1,2,3", ["1", "2", "3"]),
    ],
)
def test_extract_values(text, expected):
    assert values(text) == [Decimal(e) for e in expected]


def test_extract_flags_and_raw():
    q = extract_numbers("growth of 4.5% and ₹1,25,630 crore")
    assert [x.raw for x in q] == ["4.5%", "1,25,630 crore"]
    assert q[0].percent and not q[1].percent
    assert q[1].mantissa == Decimal("125630") and q[1].exponent == 7


def test_unit_must_be_a_whole_word():
    # "crew" / "million-dollar" prefix tricks: only real scale words scale the number
    assert values("12 crew members") == [Decimal(12)]
    assert values("5 cr.") == [Decimal(50000000)]


@pytest.mark.parametrize(
    "answer",
    [
        "Revenue grew in FY25 and Q3 [S1].",  # codes and source markers
        "In the year 2024-25 the company grew.",  # fiscal-year range
        "As of 31 March 2025 and March 31, 2024.",  # dates
        "It operates in 3 segments.",  # single digits
        "He joined on the 25th.",  # ordinal
        "1. First point\n2. Second point",  # list numbering
        "The figures for 2025 are final.",  # bare year
    ],
)
def test_answer_side_noise_is_skipped(answer):
    assert extract_numbers(answer, for_answer=True) == []


def test_answer_side_keeps_real_figures_next_to_noise():
    got = extract_numbers("Revenue in FY2025 was ₹12,563 crore, up 4.5% from 2024-25 [S1].", for_answer=True)
    assert [q.raw for q in got] == ["12,563 crore", "4.5%"]


def test_chunk_side_keeps_everything():
    # chunks are not filtered: "FY25" contributes 25, a year contributes 2025
    assert values("FY25 results, year 2025, 3 segments") == [Decimal(25), Decimal(2025), Decimal(3)]


# ---- the check


def test_check_na_without_numbers():
    c = check_numbers("The company makes cables.", ["anything 123"])
    assert c.status == "na" and not c.warning


def test_check_pass_verbatim():
    c = check_numbers("Revenue was 12,563 in FY25.", ["Revenue from operations 12,563 crore (FY25)"])
    assert c.status == "pass" and c.missing == ()


def test_check_fail_when_number_not_in_chunks():
    c = check_numbers("Revenue was 12,999.", ["Revenue from operations 12,563"])
    assert c.status == "fail" and c.warning
    assert c.missing == ("12,999",)


def test_check_indian_vs_western_grouping_same_value():
    assert check_numbers("Profit was 1,25,630.", ["PAT 125630"]).status == "pass"
    assert check_numbers("Profit was 125,630.", ["PAT 1,25,630"]).status == "pass"


def test_check_unit_in_table_header_matches_bare_cell():
    chunk = "| Particulars (₹ crore) | FY25 |\n| Revenue | 1,25,630 |"
    assert check_numbers("Revenue was ₹1,25,630 crore.", [chunk]).status == "pass"


def test_check_scale_conversion_across_units():
    assert check_numbers("Sales were 12,563 crore.", ["Sales of 125.63 billion"]).status == "pass"
    assert check_numbers("Sales were 12.5 million.", ["Sales: 12,500,000"]).status == "pass"


def test_check_percent():
    assert check_numbers("Margin was 4.5%.", ["EBITDA margin 4.5 %"]).status == "pass"
    assert check_numbers("Margin was 4.6%.", ["EBITDA margin 4.5 %"]).status == "fail"


def test_check_is_exact_not_fuzzy():
    assert check_numbers("Revenue was 12,563.", ["Revenue 12,564"]).status == "fail"
    assert check_numbers("Revenue was 12.5 crore.", ["Revenue 12.50 crore"]).status == "pass"  # same value


def test_check_ignores_sign():
    assert check_numbers("A loss of 1,234 was booked.", ["Net result (1,234)"]).status == "pass"


def test_check_every_figure_must_be_found():
    c = check_numbers("Revenue 12,563 and profit 999.", ["Revenue 12,563"])
    assert c.status == "fail" and c.missing == ("999",) and c.checked == ("12,563", "999")


def test_check_looks_only_at_the_given_chunks():
    assert check_numbers("Revenue was 12,563.", []).status == "fail"


# ---- spaced period codes and computed figures


@pytest.mark.parametrize(
    "answer",
    [
        "Return on equity for FY 25 was 18.44%.",
        "Revenue in FY 2026 rose.",
        "Results for FY 2025-26 and FY25-26.",
        "In Q3 FY26 and Q 3 and H 1 and H1.",
    ],
)
def test_spaced_period_codes_are_not_figures(answer):
    assert [q.raw for q in extract_numbers(answer, for_answer=True)] in ([], ["18.44%"])


def test_period_code_does_not_swallow_a_real_figure():
    got = extract_numbers("FY 25 revenue was 3,415.82 and FY 2026 EBITDA 1,162.17", for_answer=True)
    assert [q.raw for q in got] == ["3,415.82", "1,162.17"]


def test_check_spaced_fy_code_passes():
    c = check_numbers("ROE for FY 25 was 18.44%.", ["Return on equity 18.44%"])
    assert c.status == "pass" and c.missing == () and c.computed == ()


EBITDA_CHUNK = "| EBITDA (₹ Mn) | FY2024 | FY2025 |\n| EBITDA | 615.30 | 1,097.36 |"


def test_check_difference_passes_as_computed():
    c = check_numbers("EBITDA increased by ₹482.06 million.", [EBITDA_CHUNK])
    assert c.status == "pass" and not c.warning and c.missing == ()
    assert c.computed == ("482.06 = 1097.36 − 615.30",)


def test_check_difference_works_in_either_order():
    chunk = "| Project Engineering | 200.28 | 75.40 |"
    assert check_numbers("A decline of 124.88.", [chunk]).computed == ("124.88 = 200.28 − 75.40",)


def test_check_wrong_difference_still_fails():
    c = check_numbers("EBITDA increased by ₹482.60 million.", [EBITDA_CHUNK])
    assert c.status == "fail" and c.missing == ("482.60 million",) and c.computed == ()


def test_check_rounded_share_passes_as_computed():
    chunk = "| Project Engineering | 75.40 |\n| Revenue from operations | 3,415.82 |"
    c = check_numbers("Project Engineering is about 2.2% of revenue.", [chunk])
    assert c.status == "pass" and c.computed == ("2.2 = 75.40 / 3415.82 × 100",)
    assert check_numbers("It is about 2.3% of revenue.", [chunk]).status == "fail"


def test_check_growth_and_decline_rates_pass_as_computed():
    chunk = "| Project Engineering | 200.28 | 75.40 |"
    assert check_numbers("down about 62.4%", [chunk]).computed == ("62.4 = (200.28 − 75.40) / 200.28 × 100",)
    assert check_numbers("up about 166%", [chunk]).status == "pass"  # 200.28 over 75.40
    assert check_numbers("down about 62%", [chunk]).status == "pass"  # rounded to its own decimals
    assert check_numbers("down about 63%", [chunk]).status == "fail"


def test_check_only_percentages_use_ratios():
    chunk = "| A | 75.40 |\n| B | 3,415.82 |"
    assert check_numbers("A is 0.02 of B", [chunk]).status == "fail"


def test_check_found_figures_are_not_labelled_computed():
    c = check_numbers("EBITDA was 1,097.36 and rose by 482.06.", [EBITDA_CHUNK])
    assert c.status == "pass" and c.computed == ("482.06 = 1097.36 − 615.30",)


def test_check_mixed_answer_fails_only_for_the_unexplained_figure():
    c = check_numbers("EBITDA rose by 482.06 to 1,097.36, with 999 new sites.", [EBITDA_CHUNK])
    assert c.status == "fail" and c.missing == ("999",) and len(c.computed) == 1


def test_check_years_are_not_operands():
    # 2025 - 1,500 = 525 would explain the claim if the bare year counted as a figure
    assert check_numbers("A change of 525.", ["Revenue 1,500 in 2025"]).status == "fail"


def test_check_computed_pass_is_skipped_for_a_huge_pool(monkeypatch):
    import app.answering.numbers as n

    assert check_numbers("EBITDA increased by 482.06.", [EBITDA_CHUNK]).status == "pass"
    monkeypatch.setattr(n, "MAX_COMPUTED_POOL", 1)
    assert check_numbers("EBITDA increased by 482.06.", [EBITDA_CHUNK]).status == "fail"


def test_check_ratios_are_skipped_for_a_large_pool(monkeypatch):
    import app.answering.numbers as n

    chunk = "| Project Engineering | 200.28 | 75.40 |"
    assert check_numbers("down about 62.4%", [chunk]).status == "pass"
    monkeypatch.setattr(n, "MAX_RATIO_POOL", 1)
    assert check_numbers("down about 62.4%", [chunk]).status == "fail"
    assert check_numbers("down by 124.88", [chunk]).status == "pass"  # differences still tried
