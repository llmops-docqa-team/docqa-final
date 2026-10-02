"""Cheap, conservative token estimate (no tokenizer dependency).

bge-small reads at most 512 WordPiece tokens. We count words and punctuation marks as one token each
and charge long words extra (they split into several pieces). It over-counts slightly on purpose, so a
chunk sized at 400 "tokens" stays safely under the model limit. Because whitespace separates the
counted units, estimate(a + " " + b) == estimate(a) + estimate(b).
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_LONG_WORD = 6
_CHARS_PER_PIECE = 5


def estimate_tokens(text: str) -> int:
    total = 0
    for tok in _TOKEN_RE.findall(text):
        n = len(tok)
        total += 1 if n <= _LONG_WORD else -(-n // _CHARS_PER_PIECE)
    return total
