"""A small in-memory BM25 over the chunks in the Chroma collection (no extra dependency).

Why it exists: dense embeddings (bge-small) match a number-heavy table chunk poorly against a question like
"What was current borrowings per the standalone balance sheet?", while the exact words ("borrowings",
"standalone", "balance sheet", and numbers such as "985.71") rank it first lexically. Built lazily from the
collection and rebuilt when the collection changes. Numbers keep their separators ("1,467.25" is one token).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = frozenset(
    "the of and in to a for was what is were per on by as at an or its it this that with from basis".split()
)


_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")


def tokenize(text: str) -> list[str]:
    """Lower-case word/number tokens without stop words. Ordinals fold to their number, so "31st March"
    and "March 31" share the token "31" (reports and questions write dates both ways)."""
    return [w for w in _TOKEN.findall(_ORDINAL.sub(r"\1", text.lower())) if w not in _STOP]


class BM25Index:
    def __init__(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        doc_ids: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not (len(ids) == len(texts) == len(doc_ids)):
            raise ValueError("ids, texts and doc_ids must have the same length")
        self.ids = list(ids)
        self.doc_ids = list(doc_ids)
        self.k1, self.b = k1, b
        counts = [Counter(tokenize(t)) for t in texts]
        self._len = [sum(c.values()) for c in counts]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(counts):
            for term, tf in c.items():
                self._postings[term].append((i, tf))

    def search(self, query: str, doc_ids: Collection[str] | None, n: int) -> list[tuple[str, float]]:
        """Best `n` (chunk id, score) with a positive score, best first, restricted to `doc_ids` if given."""
        n_chunks = len(self.ids)
        if not n_chunks or n < 1:
            return []
        allowed = None if doc_ids is None else set(doc_ids)
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n_chunks - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, tf in postings:
                if allowed is not None and self.doc_ids[i] not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg) if self._avg else self.k1
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        return [(self.ids[i], s) for i, s in best]


def rrf(rankings: Sequence[tuple[float, Sequence[str]]], k: int = 60) -> list[str]:
    """Reciprocal-rank fusion of weighted ranked id lists; best first. Ties keep the first list's order."""
    score: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for weight, ranked in rankings:
        for pos, cid in enumerate(ranked):
            score[cid] += weight / (k + pos + 1)
            first_seen.setdefault(cid, len(first_seen))
    return sorted(score, key=lambda c: (-score[c], first_seen[c]))
