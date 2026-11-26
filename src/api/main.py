"""FastAPI application entry point for FinSight AI."""

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import chat, documents, health, ingest
from src.config import settings
from src.embeddings.embedding_model import get_embedding_model
from src.ingestion.ingest import sync_approved_documents_if_empty
from src.vectorstore.chroma_store import initialize_collections


logger = logging.getLogger(__name__)
API_V1_PREFIX = "/api/v1"


def _normalize_cors_origins(origins: Sequence[str]) -> list[str]:
    """Validate exact HTTP(S) origins and reject wildcard or path entries."""

    normalized: list[str] = []
    for raw_origin in origins:
        origin = raw_origin.strip().rstrip("/")
        if not origin:
            continue
        parsed = urlsplit(origin)
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError(
                "CORS origins must be exact HTTP(S) origins."
            ) from exc
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or not parsed.hostname
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("CORS origins must be exact HTTP(S) origins.")
        if origin not in normalized:
            normalized.append(origin)
    return normalized


def _configured_cors_origins() -> list[str]:
    """Parse the comma-separated server-side CORS allowlist."""

    return _normalize_cors_origins(settings.API_CORS_ORIGINS.split(","))


def initialize_api_resources() -> dict[str, Any]:
    """Initialize the configured embeddings for Supabase pgvector retrieval."""

    settings.validate_api_configuration()
    embedding_model = get_embedding_model()
    return {
        "backend": "pgvector",
        "embedding_model": embedding_model,
    }


def create_app(
    *,
    resources_factory: Callable[[], Mapping[str, Any]] | None = None,
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    """Create the API app with injectable resources for isolated tests."""

    initialize = (
        initialize_api_resources
        if resources_factory is None
        else resources_factory
    )
    allowed_origins = _normalize_cors_origins(
        _configured_cors_origins() if cors_origins is None else cors_origins
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            collections = initialize()
            if not isinstance(collections, Mapping):
                raise TypeError("Collection initialization returned invalid data.")
        except Exception:
            logger.error("API resource initialization failed.")
            raise RuntimeError("API resource initialization failed.") from None
        app.state.collections = collections
        yield
        app.state.collections = None

    application = FastAPI(
        title="FinSight AI RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    application.include_router(health.router)
    application.include_router(chat.router)
    application.include_router(ingest.router)
    application.include_router(documents.router)

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request,
        _exception: RequestValidationError,
    ) -> JSONResponse:
        del _request, _exception
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid request."},
        )

    @application.exception_handler(Exception)
    async def unexpected_error(
        _request: Request,
        _exception: Exception,
    ) -> JSONResponse:
        del _request, _exception
        logger.error("Unhandled API request failure.")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    return application


app = create_app()
