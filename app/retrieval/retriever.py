"""Dense retrieval over the Chroma index, restricted to documents that are currently queryable.

`retrieve()` embeds the question (bge query-instruction prefix, done by the embedder), searches only
documents whose status is PARTIAL or READY, and returns ranked chunks with a cosine similarity score.
It makes no LLM call. The answer step (05) takes the first `top_k` (5) of the `fetch_k` (8) chunks.

Modes (`retrieval.mode`): `dense` (cosine only), `bm25` (lexical only) or `hybrid` (both lists fused with
weighted reciprocal-rank fusion). A chunk's `score` is always its cosine similarity to the question, whatever
ranked it, so the step 05 score gate means the same thing in every mode.
"""
from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from app.config import RetrievalConfig, TopicBoostConfig
from app.ingestion.chunking import title_from_filename
from app.ingestion.embedder import Embedder
from app.ingestion.index import VectorIndex
from app.ingestion.tokens import estimate_tokens
from app.retrieval.bm25 import BM25Index, rrf_scored
from app.storage import documents as st
from app.storage.documents import DocumentStore


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    doc_id: str
    filename: str
    page: int             # 1-based PDF page index
    page_label: str       # printed label (what a citation shows)
    source_kind: str      # "text" | "table" | "ocr"
    text: str             # the chunk as stored (without the embedding header)
    score: float          # cosine similarity, 1 - distance; higher is better
    rank: int             # 1-based


@dataclass(frozen=True)
class DocStatus:
    doc_id: str
    filename: str
    status: str
    pages_done: int
    pages_total: int | None
    error: str | None = None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    searched: list[DocStatus] = field(default_factory=list)     # queryable docs that were searched
    not_ready: list[DocStatus] = field(default_factory=list)    # QUEUED / PROCESSING / FAILED docs
    t_embed_ms: float = 0.0
    t_retrieve_ms: float = 0.0

    @property
    def top_score(self) -> float | None:
        """Best cosine similarity among the returned chunks (the first chunk, in dense mode)."""
        return max((c.score for c in self.chunks), default=None)

    @property
    def partial(self) -> list[DocStatus]:
        """Searched documents that are still being processed (so a miss may just mean 'not indexed yet')."""
        return [d for d in self.searched if d.status == st.PARTIAL]

    def coverage_note(self) -> str | None:
        """Text for step 05 to show when it abstains and some searched document is unfinished."""
        parts = [
            f"{d.filename}: searched {d.pages_done} of {d.pages_total or '?'} pages; "
            "the rest is still processing."
            for d in self.partial
        ]
        return " ".join(parts) or None

    def to_dict(self) -> dict:
        return {
            "chunks": [asdict(c) for c in self.chunks],
            "searched": [asdict(d) for d in self.searched],
            "not_ready": [asdict(d) for d in self.not_ready],
            "top_score": self.top_score,
            "coverage_note": self.coverage_note(),
            "t_embed_ms": round(self.t_embed_ms, 1),
            "t_retrieve_ms": round(self.t_retrieve_ms, 1),
        }


def _keep_lexical_top(order: list[str], lexical: list[str], n: int, limit: int) -> list[str]:
    """`order` with keyword search's best `n` chunks moved inside its first `limit` places (the passages the
    model sees). Chunks already there stay put; a missing one takes the last places, pushing the rest down."""
    if n <= 0 or limit <= 0:
        return order
    missing = [cid for cid in lexical[:n] if cid not in order[:limit]]
    if not missing:
        return order
    rest = [cid for cid in order if cid not in missing]
    slot = max(0, limit - len(missing))
    return rest[:slot] + missing + rest[slot:]


def topic_headings(question: str, cfg: TopicBoostConfig) -> list[str]:
    """Lower-cased headings that fit the question's topic (see `retrieval.topic_boost.topics`)."""
    q = question.lower()
    found: list[str] = []
    for rule in cfg.topics:
        if any(re.search(rf"(?<![a-z0-9]){re.escape(w.lower())}", q) for w in rule.when):
            found += [h.lower() for h in rule.headings]
    return list(dict.fromkeys(found))


def apply_topic_boost(
    fused: list[tuple[str, float]], text_of: Callable[[str], str], headings: list[str], cfg: TopicBoostConfig
) -> list[tuple[str, float]]:
    """`fused` (best first) with `cfg.bonus` x the best score added to each of the first `cfg.depth` chunks
    whose text holds one of `headings`, then re-sorted (stable). No model, no I/O: O(depth)."""
    if not fused or not headings:
        return fused
    bonus = cfg.bonus * fused[0][1]
    boosted = [
        (cid, score + bonus if i < cfg.depth and any(h in text_of(cid) for h in headings) else score)
        for i, (cid, score) in enumerate(fused)
    ]
    return sorted(boosted, key=lambda x: -x[1])


def _table_key(text: str) -> str:
    """A table piece's title and header row, which every piece of one table repeats ("" if not a table)."""
    head, sep, _ = text.partition("| --- |")
    return head.strip() if sep else ""


def fit_context(
    chunks: Sequence[RetrievedChunk],
    max_tokens: int,
    siblings: Callable[[RetrievedChunk], list[RetrievedChunk]] | None = None,
) -> list[RetrievedChunk]:
    """The passages the model sees, within `max_tokens`: `chunks` best first, then, in the room left, the
    other pieces of any split table among them (small-to-big: the row that answers may sit in the piece that
    ranked lower), each placed right after its own piece. The retrieved passages come first, so a sibling
    never pushes one out. A passage that does not fit is skipped; the first one is always kept."""
    kept: list[RetrievedChunk] = []
    seen: set[str] = set()
    used = 0
    for c in chunks:
        t = estimate_tokens(c.text)
        if c.id in seen or (kept and used + t > max_tokens):
            continue
        kept.append(c)
        seen.add(c.id)
        used += t
    if siblings is None:
        return kept
    out: list[RetrievedChunk] = []
    for c in kept:
        out.append(c)
        if c.source_kind != "table":
            continue
        for s in siblings(c):
            t = estimate_tokens(s.text)
            if s.id in seen or used + t > max_tokens:
                continue
            out.append(s)
            seen.add(s.id)
            used += t
    return out


class Retriever:
    def __init__(self, index: VectorIndex, embedder: Embedder, store: DocumentStore, cfg: RetrievalConfig):
        self.index = index
        self.embedder = embedder
        self.store = store
        self.cfg = cfg
        self._bm25: BM25Index | None = None
        self._bm25_key: tuple | None = None

    def retrieve(
        self, question: str, top_k: int | None = None, doc_ids: Sequence[str] | None = None
    ) -> RetrievalResult:
        """Top `top_k` (default `retrieval.fetch_k`) chunks, best first.

        `doc_ids` narrows the search further (the eval uses it); documents that are not queryable are
        never searched, whatever is passed.
        """
        k = top_k if top_k is not None else self.cfg.fetch_k
        if k < 1:
            raise ValueError("top_k must be >= 1")

        searched: list[DocStatus] = []
        not_ready: list[DocStatus] = []
        wanted = None if doc_ids is None else set(doc_ids)
        for doc in self.store.list():
            if wanted is not None and doc["id"] not in wanted:
                continue
            info = DocStatus(
                doc_id=doc["id"], filename=doc["filename"], status=doc["status"],
                pages_done=doc.get("pages_done") or 0, pages_total=doc.get("pages_total"),
                error=doc.get("error"),
            )
            (searched if doc["status"] in st.QUERYABLE else not_ready).append(info)
        if not searched:
            return RetrievalResult(chunks=[], searched=searched, not_ready=not_ready)

        t0 = time.perf_counter()
        vector = self.embedder.embed_query(question)
        t1 = time.perf_counter()
        chunks = self._search(question, vector, k, [d.doc_id for d in searched])
        t2 = time.perf_counter()
        return RetrievalResult(
            chunks=chunks, searched=searched, not_ready=not_ready,
            t_embed_ms=(t1 - t0) * 1000, t_retrieve_ms=(t2 - t1) * 1000,
        )

    def _search(
        self, question: str, vector: Sequence[float], k: int, doc_ids: list[str]
    ) -> list[RetrievedChunk]:
        if self.cfg.mode == "dense":
            return self._dense(vector, k, doc_ids)
        pool = max(self.cfg.pool, k)
        lexical = self._bm25_index().search(question, doc_ids, pool)
        if self.cfg.mode == "bm25":
            order = [cid for cid, _ in lexical][:k]
            dense: dict[str, RetrievedChunk] = {}
        else:
            dense = {c.id: c for c in self._dense(vector, pool, doc_ids)}
            fused = rrf_scored(
                [(1.0, list(dense)), (self.cfg.bm25_weight, [cid for cid, _ in lexical])], self.cfg.rrf_k
            )
            boost = self.cfg.topic_boost
            if boost.enabled:
                fused = apply_topic_boost(
                    fused, self._bm25_index().text_lower, topic_headings(question, boost), boost
                )
            order = [cid for cid, _ in fused]
            order = _keep_lexical_top(order, [cid for cid, _ in lexical], self.cfg.bm25_keep_top,
                                      min(k, self.cfg.top_k))[:k]
        by_id = {**dense, **self._by_id([cid for cid in order if cid not in dense], vector)}
        return [
            RetrievedChunk(**{**by_id[cid].__dict__, "rank": rank})
            for rank, cid in enumerate(order, start=1)
            if cid in by_id
        ]

    def _dense(self, vector: Sequence[float], k: int, doc_ids: list[str]) -> list[RetrievedChunk]:
        where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        res = self.index.collection.query(
            query_embeddings=[list(vector)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        for i, (cid, text, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True),
            start=1,
        ):
            out.append(self._chunk(cid, text, meta, round(1.0 - float(dist), 6), i))
        return out

    def _by_id(self, ids: list[str], vector: Sequence[float]) -> dict[str, RetrievedChunk]:
        """Chunks the dense search did not return (found by BM25), scored by cosine similarity."""
        if not ids:
            return {}
        res = self.index.collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        q_norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        out: dict[str, RetrievedChunk] = {}
        rows = zip(res["ids"], res["documents"], res["metadatas"], res["embeddings"], strict=True)
        for cid, text, meta, emb in rows:
            e_norm = math.sqrt(sum(float(x) * float(x) for x in emb)) or 1.0
            cos = sum(float(a) * float(b) for a, b in zip(vector, emb, strict=True)) / (q_norm * e_norm)
            out[cid] = self._chunk(cid, text, meta, round(cos, 6), 0)
        return out

    def table_siblings(self, chunk: RetrievedChunk) -> list[RetrievedChunk]:
        """The other pieces of the table `chunk` is a piece of, in table order (same page, same title and
        header row). They carry `chunk`'s score and rank. Empty for a table that was not split."""
        key = _table_key(chunk.text)
        if not key:
            return []
        res = self.index.collection.get(
            where={"$and": [{"doc_id": chunk.doc_id}, {"page": chunk.page}, {"source_kind": "table"}]},
            include=["documents", "metadatas"],
        )
        found = [
            (int(meta.get("chunk_idx", 0)), self._chunk(cid, text, meta, chunk.score, chunk.rank))
            for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
            if cid != chunk.id and _table_key(text) == key
        ]
        return [c for _, c in sorted(found, key=lambda x: x[0])]

    @staticmethod
    def _chunk(cid: str, text: str, meta: dict, score: float, rank: int) -> RetrievedChunk:
        return RetrievedChunk(
            id=cid, doc_id=meta["doc_id"], filename=meta["filename"], page=int(meta["page"]),
            page_label=str(meta["page_label"]), source_kind=meta["source_kind"], text=text,
            score=score, rank=rank,
        )

    def _bm25_index(self) -> BM25Index:
        """Built from the collection on first use and rebuilt after any write to it.

        Each chunk is indexed with its document's title in front ("HPCL AR FY25"), the same header the
        embeddings get, so a question that names the company or report matches that report's chunks:
        a balance-sheet chunk itself rarely contains the company's name."""
        key = (getattr(self.index, "version", 0), self.index.collection.count())
        if self._bm25 is None or key != self._bm25_key:
            data = self.index.collection.get(include=["documents", "metadatas"])
            texts = [
                f"{title_from_filename(m.get('filename', ''))}\n{doc}"
                for doc, m in zip(data["documents"], data["metadatas"], strict=True)
            ]
            self._bm25 = BM25Index(data["ids"], texts, [m["doc_id"] for m in data["metadatas"]])
            self._bm25_key = key
        return self._bm25