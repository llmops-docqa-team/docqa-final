"""Chroma persistent collection (cosine) for chunk vectors, with the embedding-model guard."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.ingestion.chunking import Chunk

COLLECTION = "chunks"
# v3: text pages are indexed before OCR pages; ₹ drawn as a backtick is restored.
# v4: every chunk starts with its page heading ("Standalone Balance Sheet as at ...").
INGEST_VERSION = 6   # bump when parsing/chunking output changes; stored on every chunk
_UPSERT_BATCH = 500


class EmbeddingModelMismatch(RuntimeError):
    """The configured embedding model differs from the one the index was built with."""


class VectorIndex:
    def __init__(self, path: str | Path, embedding_model: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # get_or_create keeps the metadata of an existing collection, so the model recorded on first
        # creation is what we compare against on every later start.
        self.collection = self._client.get_or_create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
            embedding_function=None,
        )
        self.embedding_model = embedding_model
        self.version = 0  # bumped on every write; the retriever's BM25 index is rebuilt when it changes
        stored = (self.collection.metadata or {}).get("embedding_model")
        if stored != embedding_model:
            raise EmbeddingModelMismatch(
                f"The vector index in '{path}' was built with embedding model '{stored}', but config.yaml "
                f"says '{embedding_model}'. Vectors from different models are not comparable. Either set "
                f"embedding.model back to '{stored}', or delete '{path}' and re-upload the documents."
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Deterministic chunk IDs: re-running a job overwrites rather than duplicates."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.version += 1
        for i in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[i : i + _UPSERT_BATCH]
            self.collection.upsert(
                ids=[c.id for c in batch],
                embeddings=[list(v) for v in vectors[i : i + _UPSERT_BATCH]],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "filename": c.filename,
                        "page": c.page,
                        "page_label": c.page_label,
                        "chunk_idx": c.chunk_idx,
                        "source_kind": c.source_kind,
                        "char_len": c.char_len,
                        "embedding_model": self.embedding_model,
                        "ingest_version": INGEST_VERSION,
                    }
                    for c in batch
                ],
            )

    def delete_doc(self, doc_id: str) -> None:
        self.version += 1
        self.collection.delete(where={"doc_id": doc_id})

    def count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return self.collection.count()
        return len(self.collection.get(where={"doc_id": doc_id}, include=[])["ids"])