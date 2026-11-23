"""Secure role-scoped similarity retrieval from initialized Chroma collections."""

import json
import math
from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document

from src.config.settings import ALLOWED_ROLES
from src.retrieval.role_router import get_role_collections
from src.security.access_control import validate_role


_ADMIN_ROLE = "admin"
_REQUIRED_METADATA_FIELDS = (
    "chunk_id",
    "document_id",
    "role",
    "allowed_roles",
    "source",
    "section_title",
)


def _valid_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_chunk_metadata(
    document: Document,
    collection_name: str,
    requesting_role: str,
) -> bool:
    metadata = document.metadata
    if not isinstance(metadata, Mapping):
        return False
    if any(
        not _valid_non_empty_string(metadata.get(field))
        for field in _REQUIRED_METADATA_FIELDS
    ):
        return False

    destination_role = metadata["role"]
    if destination_role not in ALLOWED_ROLES or destination_role != collection_name:
        return False

    try:
        allowed_roles = json.loads(metadata["allowed_roles"])
    except (TypeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(allowed_roles, list)
        or not allowed_roles
        or any(not isinstance(allowed_role, str) for allowed_role in allowed_roles)
        or any(allowed_role not in ALLOWED_ROLES for allowed_role in allowed_roles)
        or len(set(allowed_roles)) != len(allowed_roles)
    ):
        return False
    if requesting_role != _ADMIN_ROLE and requesting_role not in allowed_roles:
        return False

    if "page_number" in metadata:
        page_number = metadata["page_number"]
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number <= 0
        ):
            return False
    return True


def retrieve_documents(
    query: str,
    role: str,
    collections: Mapping[str, Any],
    top_k: int = 5,
) -> list[Document]:
    """Return the top unique, authorized chunks for a role and query.

    The collection mapping is supplied by the caller; this function neither
    initializes ChromaDB nor authenticates the requested role.
    """

    validated_role = validate_role(role)
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query.strip():
        raise ValueError("query must not be empty or whitespace.")
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer.")

    routed_collections = get_role_collections(validated_role, collections)
    # A chunk may occur in every permitted collection, so fetch beyond top_k
    # before deduplicating the merged results.
    fetch_k = top_k * len(routed_collections)
    best_by_chunk_id: dict[str, tuple[float, Document]] = {}

    for collection_name, collection in routed_collections.items():
        try:
            results = collection.similarity_search_with_relevance_scores(
                query,
                k=fetch_k,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Similarity search failed for collection "
                f"{collection_name!r}."
            ) from exc

        if not isinstance(results, list):
            raise RuntimeError(
                f"Collection {collection_name!r} returned malformed search "
                "results."
            )
        for result in results:
            if (
                not isinstance(result, tuple)
                or len(result) != 2
                or not isinstance(result[0], Document)
            ):
                raise RuntimeError(
                    f"Collection {collection_name!r} returned a malformed "
                    "search result."
                )
            document, raw_score = result
            if not isinstance(raw_score, (int, float)) or isinstance(
                raw_score,
                bool,
            ):
                raise RuntimeError(
                    f"Collection {collection_name!r} returned an invalid "
                    "relevance score."
                )
            try:
                score = float(raw_score)
            except (OverflowError, ValueError) as exc:
                raise RuntimeError(
                    f"Collection {collection_name!r} returned an invalid "
                    "relevance score."
                ) from exc
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise RuntimeError(
                    f"Collection {collection_name!r} returned an invalid "
                    "relevance score."
                )
            if not _valid_chunk_metadata(
                document,
                collection_name,
                validated_role,
            ):
                continue

            chunk_id = document.metadata["chunk_id"]
            existing = best_by_chunk_id.get(chunk_id)
            if existing is None or score > existing[0]:
                copied_metadata = dict(document.metadata)
                copied_metadata["relevance_score"] = score
                best_by_chunk_id[chunk_id] = (
                    score,
                    Document(
                        page_content=document.page_content,
                        metadata=copied_metadata,
                    ),
                )

    ranked = sorted(
        best_by_chunk_id.values(),
        key=lambda item: item[0],
        reverse=True,
    )
    return [document for _, document in ranked[:top_k]]