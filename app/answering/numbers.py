"""Deterministic number extraction and the numeric grounding check (design §11).

`extract_numbers` turns text into `Quantity` objects: the digits as written (commas removed, Western and
Indian grouping both understood), an optional scale word (crore, lakh, million, ...) and a percent flag.
`check_numbers` then asks, for every figure in an answer, whether the same figure appears in the cited
chunks. Matching is exact on Decimals; there is no tolerance, rounding or fuzzy matching.

A figure is found when its digits OR its fully scaled value equal the digits or the scaled value of some
number in the cited text. The digits-only route is what makes "₹12,563 crore" match a table cell "12,563"
whose "(₹ crore)" unit sits in the column header. Signs are ignored (a loss printed as "(1,234)" matches
"loss of 1,234").

Answer-side numbers that are not claims about quantities are skipped: source markers like [S1], bare years,
fiscal-year ranges (2024-25), dates, "FY25"/"FY 25"/"Q3"-style codes, ordinals, list numbering, single digits.

A figure that is not printed can still pass as *computed*: a difference of two figures in the cited text, or
(for a percentage) a share / growth rate of two of them. Operands are compared as printed (mantissas), the
difference exactly and the percentage rounded to the decimals the answer itself shows.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# Power-of-ten exponent for each scale word (matched case-insensitively).
SCALES: dict[str, int] = {
    "thousand": 3,
    "lakh": 5,
    "lakhs": 5,
    "lac": 5,
    "lacs": 5,
    "million": 6,
    "millions": 6,
    "mn": 6,
    "mln": 6,
    "crore": 7,
    "crores": 7,
    "cr": 7,
    "billion": 9,
    "billions": 9,
    "bn": 9,
    "trillion": 12,
    "trillions": 12,
    "tn": 12,
}

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_UNIT_AFTER = re.compile(r"\s*(" + "|".join(sorted(SCALES, key=len, reverse=True)) + r")\b", re.IGNORECASE)
_PERCENT_AFTER = re.compile(r"\s*(?:%|per\s?cent\b|percent\b|percentage points?\b)", re.IGNORECASE)
_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_NOISE = [
    re.compile(r"\[\s*S\d+\s*\]", re.IGNORECASE),  # [S1] source markers
    re.compile(
        r"\b(?:FY\s?)?(?:19|20)\d{2}\s*[-–—/]\s*(?:\d{4}|\d{2})\b", re.IGNORECASE
    ),  # 2024-25, FY2024/25
    re.compile(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\.?,?\s+(?:19|20)\d{{2}}\b", re.IGNORECASE
    ),  # 31 March 2025
    re.compile(
        rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b", re.IGNORECASE
    ),  # March 31, 2025
    re.compile(r"^\s*\d+[.)]\s", re.MULTILINE),  # "1. " list numbering
    re.compile(
        r"\b(?:FY|[QH])\s?\d{1,4}(?:\s*[-–—/]\s*\d{2,4})?(?![\d,]|\.\d)", re.IGNORECASE
    ),  # FY 25, FY 2026, FY25-26, Q 3, Q3 FY26, H 1 (a space slips past the "letter before" rule)
]

# Computed figures: skip the pairwise search when the cited text holds more numbers than this, so a huge
# table cannot slow /query.
MAX_COMPUTED_POOL = 400
# Percentages are matched only to their printed decimals, so with many operands some pair always rounds to
# the claim by chance (measured: 5 of 7 invented percentages "computed" from 399 random numbers). Ratios are
# therefore tried only on a small pool: a few cited passages, not a whole statement.
MAX_RATIO_POOL = 120


@dataclass(frozen=True)
class Quantity:
    raw: str  # as it appeared, with its unit word / % if any
    mantissa: Decimal  # the digits as written ("1,25,630" -> 125630)
    exponent: int = 0  # from a scale word (crore -> 7)
    percent: bool = False

    @property
    def value(self) -> Decimal:
        return self.mantissa.scaleb(self.exponent)


@dataclass(frozen=True)
class NumberCheck:
    status: str  # "pass" | "fail" | "na"
    checked: tuple[str, ...] = ()  # figures from the answer that were looked for
    missing: tuple[str, ...] = ()  # ... and neither found in the cited text nor computable from it
    computed: tuple[str, ...] = ()  # figures not printed but derived: "482.06 = 1097.36 − 615.30"

    @property
    def warning(self) -> bool:
        return self.status == "fail"


def _valid_grouping(int_part: str) -> bool:
    """True for 12,563 / 1,234,567 (Western) and 1,25,630 / 12,34,56,789 (Indian)."""
    parts = int_part.split(",")
    if len(parts) == 1:
        return True
    western = 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:])
    indian = 1 <= len(parts[0]) <= 2 and len(parts[-1]) == 3 and all(len(p) == 2 for p in parts[1:-1])
    return western or indian


def parse_number(token: str) -> Decimal | None:
    """'1,25,630.50' -> Decimal('125630.50'); None if the grouping is not a valid number."""
    token = token.rstrip(",")
    int_part, dot, frac = token.partition(".")
    if not _valid_grouping(int_part):
        return None
    try:
        return Decimal(int_part.replace(",", "") + dot + frac)
    except InvalidOperation:
        return None


def _candidates(text: str):
    """Yield (token, start, end) for every number-like run, splitting runs with invalid comma grouping
    ("1,2,3" -> 1, 2, 3)."""
    for m in _NUMBER.finditer(text):
        token = m.group().rstrip(",")
        if parse_number(token) is not None:
            yield token, m.start(), m.start() + len(token)
            continue
        offset = m.start()
        for piece in re.split(r"(,)", token):
            if piece and piece != "," and parse_number(piece) is not None:
                yield piece, offset, offset + len(piece)
            offset += len(piece)


def extract_numbers(text: str, *, for_answer: bool = False) -> list[Quantity]:
    """All numbers in `text`. With `for_answer=True`, skip numbers that are not quantity claims."""
    if for_answer:
        for pattern in _NOISE:
            text = pattern.sub(" ", text)
    out: list[Quantity] = []
    for token, start, end in _candidates(text):
        mantissa = parse_number(token)
        if mantissa is None:
            continue
        tail = text[end : end + 24]
        unit = _UNIT_AFTER.match(tail)
        percent = bool(_PERCENT_AFTER.match(tail))
        exponent = SCALES[unit.group(1).lower()] if unit else 0
        consumed = unit.end() if unit else (_PERCENT_AFTER.match(tail).end() if percent else 0)

        if for_answer:
            before = text[start - 1] if start > 0 else ""
            after = text[end + consumed] if end + consumed < len(text) else ""
            if before.isalpha():  # FY25, Q3, S1, H1
                continue
            if not unit and not percent and after.isalpha():  # 3rd, 25th, 2x
                continue
            plain_int = "." not in token and "," not in token
            if not unit and not percent and plain_int:
                if len(token) == 1:  # "3 segments": too noisy to check
                    continue
                if len(token) == 4 and 1900 <= int(token) <= 2100:  # bare year
                    continue
        out.append(
            Quantity(
                raw=text[start : end + consumed].strip(),
                mantissa=mantissa,
                exponent=exponent,
                percent=percent,
            )
        )
    return out


def _decimals(d: Decimal) -> int:
    return max(0, -d.as_tuple().exponent)


def _fmt(d: Decimal) -> str:
    return format(d, "f")


def _percent_matches(x: Decimal, claim: Decimal) -> bool:
    return x.quantize(Decimal(1).scaleb(-_decimals(claim)), rounding=ROUND_HALF_UP) == claim


def _ratio_partners(ops: list[Decimal], b: Decimal, lo: Decimal, hi: Decimal):
    """Operands `a` (from the sorted `ops`) with lo*b <= a <= hi*b."""
    return ops[bisect_left(ops, lo * b) : bisect_right(ops, hi * b)]


def _derive(claim: Quantity, ops: list[Decimal]) -> str | None:
    """How `claim` can be made from two sorted, distinct operands, or None."""
    c = claim.mantissa
    if c == 0:
        return None
    present = set(ops)
    for a in ops:  # difference a - b = c (either order: the sign is ignored)
        if a - c in present:
            return f"{_fmt(c)} = {_fmt(a)} − {_fmt(a - c)}"
    if not claim.percent or len(ops) > MAX_RATIO_POOL:
        return None
    half = Decimal(5).scaleb(-_decimals(c) - 1)  # rounding to the printed decimals: [c-half, c+half)
    lo, hi = (c - half) / 100, (c + half) / 100  # share a/b as a fraction
    for b in ops:
        for a in _ratio_partners(ops, b, lo, hi):  # share: a / b x 100
            if _percent_matches(a / b * 100, c):
                return f"{_fmt(c)} = {_fmt(a)} / {_fmt(b)} × 100"
        for a in _ratio_partners(ops, b, 1 + lo, 1 + hi):  # growth of a over base b
            if _percent_matches((a - b) / b * 100, c):
                return f"{_fmt(c)} = ({_fmt(a)} − {_fmt(b)}) / {_fmt(b)} × 100"
        for a in _ratio_partners(ops, b, 1 - hi, 1 - lo):  # decline of a from base b
            if _percent_matches((b - a) / b * 100, c):
                return f"{_fmt(c)} = ({_fmt(b)} − {_fmt(a)}) / {_fmt(b)} × 100"
    return None


def check_numbers(answer: str, cited_texts: list[str]) -> NumberCheck:
    """pass: every figure in the answer is in the cited chunks or computable from two of them. fail: at least
    one is neither. na: the answer has no checkable figure."""
    claims = extract_numbers(answer, for_answer=True)
    if not claims:
        return NumberCheck("na")

    pool: set[Decimal] = set()
    operands: set[Decimal] = set()
    for text in cited_texts:
        for q in extract_numbers(text):
            pool.add(q.mantissa)
            pool.add(q.value)
            bare_year = q.raw.isdigit() and len(q.raw) == 4 and 1900 <= int(q.raw) <= 2100
            if q.mantissa > 0 and not bare_year:
                operands.add(q.mantissa)

    ops = sorted(operands) if len(operands) <= MAX_COMPUTED_POOL else []
    missing: list[str] = []
    computed: list[str] = []
    for c in claims:
        if c.mantissa in pool or c.value in pool:
            continue
        how = _derive(c, ops) if ops else None
        if how:
            computed.append(how)
        else:
            missing.append(c.raw)
    return NumberCheck(
        status="fail" if missing else "pass",
        checked=tuple(c.raw for c in claims),
        missing=tuple(missing),
        computed=tuple(computed),
    )
