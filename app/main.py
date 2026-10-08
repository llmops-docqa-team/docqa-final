"""FastAPI entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.answering.document import DocumentAnswerer
from app.answering.general import GeneralAnswerer
from app.answering.query import QueryService
from app.api.catalog import router as catalog_router
from app.api.debug import router as debug_router
from app.api.documents import router as documents_router
from app.api.feedback import router as feedback_router
from app.api.middleware import RequestIDMiddleware
from app.api.query import router as query_router
from app.config import get_settings
from app.ingestion.embedder import Embedder, embedder_from_settings
from app.ingestion.index import VectorIndex
from app.ingestion.pdf_parse import OcrFn
from app.ingestion.worker import IngestionWorker
from app.llm.client import LLMClient, llm_client_from_settings
from app.observability.logging import configure_logging, get_logger
from app.observability.tracing import build_tracer, get_tracer, set_tracer
from app.observability.version import get_app_version
from app.retrieval.retriever import Retriever
from app.routing.rewriter import QueryRewriter
from app.routing.router import Router
from app.storage.db import init_db
from app.storage.documents import DocumentStore
from app.storage.requests import RequestStore


def create_app(
    embedder: Embedder | None = None, ocr_fn: OcrFn | None = None, llm: LLMClient | None = None
) -> FastAPI:
    """`embedder` / `ocr_fn` / `llm` are injection points for tests; production uses fastembed, tesseract
    and the OpenAI-compatible client (Groq by default)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        configure_logging()
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        init_db(settings.sqlite_path)
        # Langfuse tracing: a no-op unless LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set.
        set_tracer(build_tracer(capture_content=settings.tracing.capture_content, release=get_app_version()))

        emb = embedder or embedder_from_settings(settings)
        # Raises EmbeddingModelMismatch (with instructions) if config and index disagree: we refuse to serve.
        index = VectorIndex(settings.chroma_dir, emb.model_name)
        store = DocumentStore(settings.sqlite_path)
        worker = IngestionWorker(settings, store, index, emb, ocr_fn=ocr_fn)

        app.state.settings, app.state.store, app.state.index = settings, store, index
        app.state.worker, app.state.embedder = worker, emb
        app.state.request_store = RequestStore(settings.sqlite_path)
        app.state.retriever = Retriever(index, emb, store, settings.retrieval)
        app.state.llm = llm or llm_client_from_settings(settings)
        app.state.doc_answerer = DocumentAnswerer(app.state.retriever, app.state.llm, settings)
        app.state.query_service = QueryService(
            store,
            Router(app.state.llm, settings),
            app.state.doc_answerer,
            GeneralAnswerer(app.state.llm, settings),
            settings,
            request_store=app.state.request_store,
            rewriter=QueryRewriter(app.state.llm, settings),
        )
        if not settings.groq_api_key and "groq.com" in settings.llm.base_url:
            get_logger().warning("llm_key_missing", detail="GROQ_API_KEY is not set; answers will fail")
        worker.start()
        worker.requeue_unfinished()
        get_logger().info("startup", sqlite=str(settings.sqlite_path), embedding_model=emb.model_name)
        try:
            yield
        finally:
            worker.stop()
            get_tracer().flush()

    app = FastAPI(title="FinChat", lifespan=lifespan)
    app.add_middleware(RequestIDMiddleware)
    app.include_router(documents_router)
    app.include_router(catalog_router)
    app.include_router(query_router)
    app.include_router(feedback_router)
    app.include_router(debug_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
