"""Embedder interface plus two implementations: Model2Vec static embeddings (default, fast) and fastembed
(ONNX bge-small, slower but better at matching meaning). Tests use a fake with the same shape."""
from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.config import Settings

# bge-*-en-v1.5 is trained to see this instruction in front of *queries* (not passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 (384-d) via fastembed. The model loads on first use, not at import/startup."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
        query_prefix: str = BGE_QUERY_PREFIX,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.model_name, cache_dir=self._cache_dir)
            return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in self._load().embed(list(texts), batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        """Embeds a question with the bge query-instruction prefix.

        fastembed's own `query_embed` does NOT add this prefix for bge models (checked: its output is
        identical to plain `embed`), so it is applied here. Passages are embedded without it.
        """
        return next(iter(self._load().embed([self.query_prefix + text]))).tolist()


class StaticEmbedder:
    """Model2Vec static embeddings (default: minishlab/potion-retrieval-32M, 512-d).

    A static model is a lookup table of token vectors averaged per text: no transformer runs per chunk, so
    embedding is ~2,000x faster than bge-small on a CPU (measured: 1,576 chunks of a 427-page annual report
    in 0.45 s vs ~16 min). It matches meaning less well than bge-small; in this app the keyword (BM25) side of
    hybrid search carries exact figures and names, which is what annual-report questions mostly need.
    The model downloads once to the Hugging Face cache and loads on first use, not at import/startup.
    """

    def __init__(self, model_name: str, batch_size: int = 1024, query_prefix: str = ""):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._batch_size = max(1, batch_size)
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from model2vec import StaticModel

                # force_download=False: use the cached copy; the library default re-downloads on every load.
                self._model = StaticModel.from_pretrained(self.model_name, force_download=False)
            return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(list(texts), batch_size=self._batch_size, use_multiprocessing=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._load().encode([self.query_prefix + text], use_multiprocessing=False)[0].tolist()


def embedder_from_settings(settings: Settings) -> FastEmbedder | StaticEmbedder:
    if settings.embedding.backend == "model2vec":
        return StaticEmbedder(
            settings.embedding.model,
            batch_size=settings.embedding.batch_size,
            query_prefix=settings.embedding.query_prefix,
        )
    return FastEmbedder(
        settings.embedding.model,
        cache_dir=settings.model_cache_dir,
        batch_size=settings.embedding.batch_size,
        query_prefix=settings.embedding.query_prefix,
    )
