"""Role-aware Retrieval-Augmented Generation orchestration."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from src.chains.prompts import get_rag_prompt
from src.llm.groq_model import get_groq_model
from src.retrieval.retriever import (
    retrieve_documents,
    retrieve_documents_pgvector,
)
from src.security.access_control import validate_role


_INSUFFICIENT_CONTEXT_ANSWER = (
    "The available documents do not contain enough information to answer "
    "this question."
)


@dataclass(frozen=True)
class SourceReference:
    """Provenance fields available for one retrieved source."""

    source: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None


@dataclass
class RAGResponse:
    """Generated answer with validated role and retrieved-source metadata."""

    answer: str
    sources: list[SourceReference]
    role: str
    retrieved_count: int


def _metadata_text(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _metadata_page_number(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("page_number")
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    ):
        return value
    return None


def _document_metadata(document: Document) -> Mapping[str, Any]:
    metadata = document.metadata
    return metadata if isinstance(metadata, Mapping) else {}


def _prepare_context(documents: list[Document]) -> str:
    context_chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        if not isinstance(document.page_content, str):
            raise RuntimeError(
                "The secure retriever returned a document with invalid text."
            )
        metadata = _document_metadata(document)
        provenance: dict[str, str | int] = {}
        for field in ("source", "section_title", "document_id", "chunk_id"):
            value = _metadata_text(metadata, field)
            if value is not None:
                provenance[field] = value
        page_number = _metadata_page_number(metadata)
        if page_number is not None:
            provenance["page_number"] = page_number

        metadata_json = json.dumps(
            provenance,
            ensure_ascii=False,
            sort_keys=True,
        )
        context_chunks.append(
            f"--- Retrieved chunk {index} (untrusted reference data) ---\n"
            f"Metadata: {metadata_json}\n"
            "Untrusted document content follows:\n"
            f"{document.page_content}\n"
            f"--- End retrieved chunk {index} ---"
        )
    return "\n\n".join(context_chunks)


def _build_source_references(
    documents: list[Document],
) -> list[SourceReference]:
    references: list[SourceReference] = []
    seen: set[tuple[str | int | None, ...]] = set()

    for document in documents:
        metadata = _document_metadata(document)
        reference = SourceReference(
            source=_metadata_text(metadata, "source"),
            page_number=_metadata_page_number(metadata),
            section_title=_metadata_text(metadata, "section_title"),
            document_id=_metadata_text(metadata, "document_id"),
            chunk_id=_metadata_text(metadata, "chunk_id"),
        )
        if not any((
            reference.source,
            reference.page_number,
            reference.section_title,
            reference.document_id,
            reference.chunk_id,
        )):
            continue

        if reference.document_id and reference.chunk_id:
            identity: tuple[str | int | None, ...] = (
                reference.document_id,
                reference.chunk_id,
            )
        else:
            identity = (
                reference.source,
                reference.page_number,
                reference.section_title,
                reference.document_id,
                reference.chunk_id,
            )
        if identity not in seen:
            seen.add(identity)
            references.append(reference)
    return references


def _answer_text(response: Any) -> str:
    if isinstance(response, BaseMessage):
        response = response.content
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(
            "The language model returned an unexpected or empty response."
        )
    return response.strip()


def ask_question(
    question: str,
    role: str,
    collections: Mapping[str, Any],
    top_k: int = 5,
    llm: Runnable[Any, Any] | None = None,
) -> RAGResponse:
    """Retrieve authorized evidence and generate a grounded answer.

    ``role`` must be supplied by a trusted caller. This function validates the
    role but does not authenticate users or initialize ChromaDB.
    """

    validated_role = validate_role(role)
    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    if not question.strip():
        raise ValueError("question must not be empty or whitespace.")
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer.")

    try:
        if isinstance(collections, Mapping) and collections.get("backend") == "pgvector":
            documents = retrieve_documents_pgvector(
                question,
                validated_role,
                embedding_model=collections.get("embedding_model"),
                top_k=top_k,
            )
        else:
            documents = retrieve_documents(
                question,
                validated_role,
                collections,
                top_k=top_k,
            )
    except Exception as exc:
        raise RuntimeError("Secure document retrieval failed.") from exc

    if not isinstance(documents, list) or any(
        not isinstance(document, Document)
        for document in documents
    ):
        raise RuntimeError("The secure retriever returned malformed documents.")

    if not documents:
        return RAGResponse(
            answer=_INSUFFICIENT_CONTEXT_ANSWER,
            sources=[],
            role=validated_role,
            retrieved_count=0,
        )

    context = _prepare_context(documents)
    sources = _build_source_references(documents)

    try:
        prompt = get_rag_prompt(validated_role)
        prompt_value = prompt.invoke({
            "question": question,
            "context": context,
        })
    except Exception as exc:
        raise RuntimeError("Unable to construct the role-aware RAG prompt.") from exc

    try:
        model = llm if llm is not None else get_groq_model()
    except Exception as exc:
        raise RuntimeError("Unable to initialize the Groq language model.") from exc

    try:
        model_response = model.invoke(prompt_value)
    except Exception as exc:
        raise RuntimeError("Unable to generate a RAG answer.") from exc

    answer = _answer_text(model_response)
    return RAGResponse(
        answer=answer,
        sources=sources,
        role=validated_role,
        retrieved_count=len(documents),
    )