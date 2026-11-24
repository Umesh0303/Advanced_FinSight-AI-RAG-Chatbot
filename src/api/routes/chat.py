"""Authenticated chat endpoint backed by the existing secure RAG chain."""

import logging
from collections.abc import Mapping
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.dependencies.auth import (
    AuthenticatedPrincipal,
    get_authenticated_principal,
)
from src.api.schemas.chat import ChatRequest, ChatResponse, SourceResponse
from src.api.services import chat_service


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(
    payload: ChatRequest,
    request: Request,
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
) -> ChatResponse:
    """Answer one authenticated question without trusting request-supplied roles."""

    collections = getattr(request.app.state, "collections", None)
    if not isinstance(collections, Mapping):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The RAG service is not ready.",
        )

    try:
        response = chat_service.answer_question(
            payload.message,
            principal.role,
            collections,
        )
    except chat_service.ChatServiceError:
        logger.warning("RAG chat operation failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The RAG service is temporarily unavailable.",
        ) from None

    return ChatResponse(
        success=True,
        answer=response.answer,
        session_id=payload.session_id or str(uuid4()),
        sources=[
            SourceResponse.model_validate(source)
            for source in response.sources
        ],
    )
