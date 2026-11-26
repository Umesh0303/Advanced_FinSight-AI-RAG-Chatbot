"""Persistent ChromaDB collection management for FinSight AI."""

import json
from typing import Any, Mapping

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from src.config.settings import ALLOWED_ROLES, CHROMA_DB_DIR, COLLECTION_NAMES
from src.embeddings.embedding_model import get_embedding_model


SECURITY_METADATA_FIELDS: tuple[str, ...] = (
    "document_id",
    "section_id",
    "source_id",
    "chunk_id",
    "source",
    "page_number",
    "section_title",
    "role",
    "allowed_roles",
    "classification_method",
    "classification_confidence",
)


def normalize_chroma_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, str | int | float | bool]:
    """Convert supplied metadata to scalar values supported by ChromaDB."""

    if metadata is None:
        return {}

    normalized: dict[str, str | int | float | bool] = {}

    for field_name in SECURITY_METADATA_FIELDS:
        if field_name not in metadata:
            continue

        value = metadata[field_name]
        if value is None:
            continue

        if field_name == "allowed_roles":
            if isinstance(value, list) or isinstance(value, tuple):
                normalized[field_name] = json.dumps(list(value))
            elif isinstance(value, str):
                normalized[field_name] = value
            else:
                raise ValueError(
                    "allowed_roles must be a list, tuple, or JSON string."
                )
        elif isinstance(value, (str, int, float, bool)):
            normalized[field_name] = value
        else:
            raise ValueError(
                f"Unsupported ChromaDB metadata type for '{field_name}'. "
                "Use a string, number, or boolean."
            )

    return normalized


def initialize_collections(
    embedding_model: Embeddings | None = None,
) -> dict[str, Chroma]:
    """Create or retrieve all four logical collections in one persistent database."""

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    embedding = embedding_model or get_embedding_model()

    collections: dict[str, Chroma] = {}
    for collection_name in COLLECTION_NAMES:
        collections[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=str(CHROMA_DB_DIR),
        )

    return collections


def get_collection(
    role: str,
    embedding_model: Embeddings | None = None,
) -> Chroma:
    """Retrieve the Chroma collection authorized for one allowed role."""

    if role not in ALLOWED_ROLES:
        raise ValueError(
            f"Invalid role '{role}'. Allowed roles: {', '.join(ALLOWED_ROLES)}"
        )

    collections = initialize_collections(embedding_model=embedding_model)
    return collections[role]