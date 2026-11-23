"""Load supported documents from the configured uploads directory."""

from pathlib import Path
from typing import Final

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

from src.config.settings import SUPPORTED_DOCUMENT_EXTENSIONS, UPLOADS_DIR


def _resolve_upload_path(file_path: Path) -> Path:
    """Resolve a path and verify that it is inside the uploads directory."""

    try:
        resolved_uploads_dir = UPLOADS_DIR.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"The configured uploads directory does not exist: {UPLOADS_DIR}"
        ) from exc

    try:
        resolved_file_path = file_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Document does not exist: {file_path}"
        ) from exc

    if not resolved_file_path.is_file():
        raise ValueError(
            f"Document path is not a regular file: {resolved_file_path}"
        )

    try:
        resolved_file_path.relative_to(resolved_uploads_dir)
    except ValueError as exc:
        raise ValueError(
            "Document loading is restricted to the configured uploads directory."
        ) from exc

    return resolved_file_path


def _get_loader(resolved_file_path: Path):
    """Return the appropriate LangChain loader for a supported file."""

    extension: Final[str] = resolved_file_path.suffix.lower()

    if extension == ".pdf":
        return PyPDFLoader(str(resolved_file_path))
    if extension == ".docx":
        return Docx2txtLoader(str(resolved_file_path))
    if extension == ".txt":
        return TextLoader(
            str(resolved_file_path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    raise ValueError(
        f"Unsupported document extension '{resolved_file_path.suffix}'. "
        f"Supported extensions: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}"
    )


def load_document(file_path: Path) -> list[Document]:
    """Load one PDF, DOCX, or TXT document as LangChain Document objects."""

    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a pathlib.Path")

    resolved_file_path = _resolve_upload_path(file_path)
    extension = resolved_file_path.suffix.lower()

    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Unsupported document extension '{resolved_file_path.suffix}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}"
        )

    try:
        loader = _get_loader(resolved_file_path)
        documents = loader.load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load document '{resolved_file_path.name}'. "
            "The file may be unreadable or invalid."
        ) from exc

    if not documents:
        raise ValueError(
            f"The document is empty or contains no readable content: "
            f"{resolved_file_path.name}"
        )

    populated_documents: list[Document] = []
    for document in documents:
        if not isinstance(document.page_content, str):
            continue
        if not document.page_content.strip():
            continue

        metadata = dict(document.metadata)
        metadata["source"] = resolved_file_path.name
        metadata["file_name"] = resolved_file_path.name
        metadata["file_path"] = str(resolved_file_path)
        populated_documents.append(
            Document(
                page_content=document.page_content,
                metadata=metadata,
            )
        )

    if not populated_documents:
        raise ValueError(
            f"The document is empty or contains no readable content: "
            f"{resolved_file_path.name}"
        )

    return populated_documents