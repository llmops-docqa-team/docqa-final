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
fiscal-year ranges (2024-25), dates, "FY25"/"Q3"-style codes, ordinals, list numbering and single digits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

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
]


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
    missing: tuple[str, ...] = ()  # ... and not found in the cited text

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


def check_numbers(answer: str, cited_texts: list[str]) -> NumberCheck:
    """pass: every figure in the answer is in the cited chunks. fail: at least one is not.
    na: the answer has no checkable figure."""
    claims = extract_numbers(answer, for_answer=True)
    if not claims:
        return NumberCheck("na")

    pool: set[Decimal] = set()
    for text in cited_texts:
        for q in extract_numbers(text):
            pool.add(q.mantissa)
            pool.add(q.value)

    missing = [c.raw for c in claims if c.mantissa not in pool and c.value not in pool]
    return NumberCheck(
        status="fail" if missing else "pass",
        checked=tuple(c.raw for c in claims),
        missing=tuple(missing),
    )
