"""Split one already authorized document section into LangChain chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.role_classifier import ClassificationResult


CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150


def split_classified_section(
    section: ClassificationResult,
) -> list[Document]:
    """Split an authorized section while preserving its security metadata.

    The input must have readable text, at least one authorized role, and
    manual review must not be required. The section is processed as one
    independent unit, so no chunk can mix text from another section.

    Args:
        section: A previously classified section.

    Returns:
        LangChain Document objects containing the section text chunks.

    Raises:
        TypeError: If section is not a ClassificationResult.
        ValueError: If text is empty, no role is authorized, or review is required.
    """

    if not isinstance(section, ClassificationResult):
        raise TypeError("section must be a ClassificationResult")

    if not section.section_text.strip():
        raise ValueError("Cannot split a section with empty section text.")

    if not section.allowed_roles:
        raise ValueError(
            "Cannot split an unauthorized section: allowed_roles is empty."
        )

    if section.review_required:
        raise ValueError(
            "Cannot split a section that requires manual review."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    base_metadata = {
        "section_title": section.section_title,
        "allowed_roles": list(section.allowed_roles),
        "classification_method": section.classification_method,
        "classification_confidence": section.classification_confidence,
        "review_required": section.review_required,
    }

    chunks = splitter.create_documents(
        [section.section_text],
        metadatas=[base_metadata],
    )

    indexed_chunks: list[Document] = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_metadata = dict(chunk.metadata)
        chunk_metadata["chunk_index"] = chunk_index
        indexed_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=chunk_metadata,
            )
        )

    return indexed_chunks