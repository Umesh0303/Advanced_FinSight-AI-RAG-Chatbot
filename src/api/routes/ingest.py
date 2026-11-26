"""Admin-only document upload, classification, and Supabase registration endpoint."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import zipfile
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from src.api.dependencies.auth import (
    AuthenticatedPrincipal,
    get_authenticated_principal,
)
from src.api.schemas.ingest import IngestResponse
from src.api.services import local_rag_store
from src.config.settings import SUPPORTED_DOCUMENT_EXTENSIONS, UPLOADS_DIR
from src.ingestion.ingest import classify_document_sections


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB
_STREAM_CHUNK_BYTES = 256 * 1024


def require_admin(
    principal: AuthenticatedPrincipal = Depends(get_authenticated_principal),
) -> AuthenticatedPrincipal:
    """Require an authenticated active Admin principal."""

    if principal.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role is required for document ingestion.",
        )
    return principal


def _validate_file_content(file_path: Path, extension: str) -> None:
    """Verify file magic bytes / structure match the declared extension."""

    if extension == ".pdf":
        with file_path.open("rb") as handle:
            header = handle.read(5)
        if header != b"%PDF-":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded file is not a valid PDF document.",
            )
        return

    if extension == ".docx":
        if not zipfile.is_zipfile(file_path):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded file is not a valid DOCX archive.",
            )
        try:
            with zipfile.ZipFile(file_path, "r") as archive:
                if "word/document.xml" not in archive.namelist():
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail="Uploaded DOCX archive is missing document content.",
                    )
        except zipfile.BadZipFile as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded file is not a valid DOCX archive.",
            ) from exc
        return

    if extension == ".txt":
        with file_path.open("rb") as handle:
            raw_bytes = handle.read(64 * 1024)
        if b"\x00" in raw_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded TXT file contains invalid binary data.",
            )
        try:
            raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded TXT file must be valid UTF-8 text.",
            ) from exc
        return

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="Unsupported document format.",
    )


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_upload(
    response: Response,
    file: UploadFile = File(...),
    principal: AuthenticatedPrincipal = Depends(require_admin),
) -> IngestResponse:
    """Upload, validate, classify, and register a document as pending in Supabase."""

    raw_filename = file.filename or ""
    sanitized_name = Path(raw_filename).name.strip()
    if not sanitized_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A valid filename is required.",
        )

    extension = Path(sanitized_name).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unsupported file type. Supported types: .pdf, .docx, .txt.",
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    total_bytes = 0
    digest = hashlib.sha256()

    try:
        fd, raw_temp = tempfile.mkstemp(
            dir=str(UPLOADS_DIR),
            prefix=".upload_",
            suffix=extension,
        )
        temp_path = Path(raw_temp)
        with os.fdopen(fd, "wb") as temp_file:
            while True:
                chunk = await file.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="Uploaded file exceeds the 10 MiB size limit.",
                    )
                digest.update(chunk)
                temp_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded file is empty.",
            )

        _validate_file_content(temp_path, extension)
        file_hash_sha256 = digest.hexdigest().lower()
        source_id = hashlib.sha256(
            f"upload:{sanitized_name.casefold()}".encode("utf-8")
        ).hexdigest()

        try:
            existing = local_rag_store.get_document_by_hash(file_hash_sha256)
        except local_rag_store.LocalRagStoreError:
            logger.warning("Supabase duplicate lookup failed during /ingest.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document registry service is temporarily unavailable.",
            ) from None

        if (
            existing is not None
            and existing.get("source_id") == source_id
            and existing.get("approval_status") in {"pending", "approved"}
        ):
            existing_sections = local_rag_store.get_document_sections(existing["id"])
            response.status_code = status.HTTP_200_OK
            return IngestResponse(
                success=True,
                document_id=str(existing["id"]),
                filename=str(existing.get("filename") or sanitized_name),
                file_hash_sha256=file_hash_sha256,
                approval_status=str(existing["approval_status"]),
                indexing_status=str(existing["indexing_status"]),
                approval_version=int(existing["approval_version"]),
                extracted_text_length=0,
                section_count=len(existing_sections),
                quarantined_section_count=sum(
                    1
                    for sec in existing_sections
                    if bool(sec.get("classifier_quarantined"))
                ),
                duplicate=True,
            )

        try:
            classified = classify_document_sections(
                temp_path,
                logical_filename=sanitized_name,
            )
        except (ValueError, RuntimeError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Unable to extract readable text from the uploaded document.",
            ) from None
        except Exception:
            logger.warning("Document classification failed during /ingest.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document classification service is temporarily unavailable.",
            ) from None

        try:
            registered = local_rag_store.register_document(
                file_hash_sha256=classified["file_hash_sha256"],
                source_id=classified["source_id"],
                filename=classified["filename"],
                sections=classified["sections"],
                actor_id=principal.user_id,
            )
        except local_rag_store.LocalRagStoreConflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document registration conflicted with existing state.",
            ) from None
        except local_rag_store.LocalRagStoreError:
            logger.warning("Supabase registration failed during /ingest.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document registry service is temporarily unavailable.",
            ) from None

        response.status_code = status.HTTP_201_CREATED
        return IngestResponse(
            success=True,
            document_id=str(registered["id"]),
            filename=str(registered["filename"]),
            file_hash_sha256=str(registered["file_hash_sha256"]),
            approval_status=str(registered["approval_status"]),
            indexing_status=str(registered["indexing_status"]),
            approval_version=int(registered["approval_version"]),
            extracted_text_length=int(classified["extracted_text_length"]),
            section_count=len(classified["sections"]),
            quarantined_section_count=int(classified["quarantined_section_count"]),
            duplicate=False,
        )
    finally:
        await file.close()
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Failed to remove temporary upload file.")
