"""Adapt the existing RAG chain for API requests."""

from collections.abc import Mapping
from typing import Any

from src.chains.rag_chain import RAGResponse, ask_question


class ChatServiceError(RuntimeError):
    """Safe public error for a failed or malformed RAG operation."""


def answer_question(
    message: str,
    role: str,
    collections: Mapping[str, Any],
) -> RAGResponse:
    """Call the existing RAG chain and reject malformed responses."""

    try:
        response = ask_question(message, role, collections)
    except Exception:
        raise ChatServiceError from None

    if not isinstance(response, RAGResponse):
        raise ChatServiceError
    return response
