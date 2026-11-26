"""Admin-only endpoints for document review, approval, revocation, and indexing."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from src.api.dependencies.auth import AuthenticatedPrincipal
from src.api.routes.ingest import MAX_UPLOAD_BYTES, require_admin
from src.api.schemas.documents import (
    ApproveDocumentRequest,
    RejectOrRevokeRequest,
)
from src.api.services import local_rag_store
from src.config.settings import UPLOADS_DIR
from src.ingestion.ingest import (
    index_approved_document,
    remove_document_chunks_from_chroma,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def _handle_store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, local_rag_store.LocalRagStoreConflict):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document operation conflicted with current state or validation rules.",
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Document registry service is temporarily unavailable.",
    )


@router.get("")
def list_documents(
    _principal: AuthenticatedPrincipal = Depends(require_admin),
) -> dict[str, Any]:
    """List registered local RAG documents for Admin review."""

    try:
        documents = local_rag_store.list_documents()
    except local_rag_store.LocalRagStoreError as exc:
        raise _handle_store_error(exc) from None
    return {"documents": documents}


@router.get("/{document_id}")
def get_document_detail(
    document_id: str,
    _principal: AuthenticatedPrincipal = Depends(require_admin),
) -> dict[str, Any]:
    """Return one document and all of its classified/reviewed sections."""

    try:
        document = local_rag_store.get_document_by_id(document_id)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )
        sections = local_rag_store.get_document_sections(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid document ID.",
        ) from None
    except local_rag_store.LocalRagStoreError as exc:
        raise _handle_store_error(exc) from None

    return {"document": document, "sections": sections}


@router.post("/{document_id}/approve")
def approve_document(
    document_id: str,
    payload: ApproveDocumentRequest,
    principal: AuthenticatedPrincipal = Depends(require_admin),
) -> dict[str, Any]:
    """Approve or exclude every section of a pending document atomically."""

    try:
        updated = local_rag_store.approve_document(
            document_id=document_id,
            actor_id=principal.user_id,
            section_decisions=[
                decision.model_dump() for decision in payload.section_decisions
            ],
            decision_reason=payload.decision_reason,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid document or section identifier.",
        ) from None
    except local_rag_store.LocalRagStoreError as exc:
        raise _handle_store_error(exc) from None

    return {"success": True, "document": updated}


@router.post("/{document_id}/reject")
def reject_document(
    document_id: str,
    payload: RejectOrRevokeRequest,
    principal: AuthenticatedPrincipal = Depends(require_admin),
) -> dict[str, Any]:
    """Reject a pending document with a mandatory reason."""

    try:
        updated = local_rag_store.reject_or_revoke_document(
            document_id=document_id,
            action="rejected",
            actor_id=principal.user_id,
            decision_reason=payload.decision_reason,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid document ID or decision reason.",
        ) from None
    except local_rag_store.LocalRagStoreError as exc:
        raise _handle_store_error(exc) from None

    return {"success": True, "document": updated}


@router.post("/{document_id}/revoke")
def revoke_document(
    document_id: str,
    payload: RejectOrRevokeRequest,
    request: Request,
    principal: AuthenticatedPrincipal = Depends(require_admin),
) -> dict[str, Any]:
    """Revoke an approved document in Supabase and clean up ChromaDB chunks."""

    try:
        updated = local_rag_store.reject_or_revoke_document(
            document_id=document_id,
            action="revoked",
            actor_id=principal.user_id,
            decision_reason=payload.decision_reason,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid document ID or decision reason.",
        ) from None
    except local_rag_store.LocalRagStoreError as exc:
        raise _handle_store_error(exc) from None

    removed_chunks = 0
    collections = getattr(request.app.state, "collections", None)
    if isinstance(collections, Mapping):
        try:
            removed_chunks = remove_document_chunks_from_chroma(
                document_id,
                collections,
            )
        except Exception:
            logger.warning(
                "ChromaDB cleanup failed after revocation; Supabase revocation still blocks retrieval."
            )

    return {
        "success": True,
        "document": updated,
        "removed_chunks": removed_chunks,
    }


@router.post("/{document_id}/index")
async def index_document_endpoint(
    document_id: str,
    request: Request,
    file: UploadFile | None = File(default=None),
    principal: AuthenticatedPrincipal = Depends(require_admin),
) -> dict[str, Any]:
    """Index an approved document into ChromaDB and attest verification in Supabase."""

    collections = getattr(request.app.state, "collections", None)
    if not isinstance(collections, Mapping):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vectorstore collections are not ready.",
        )

    try:
        doc = local_rag_store.get_document_by_id(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid document ID.",
        ) from None
    except local_rag_store.LocalRagStoreError as exc:
        raise _handle_store_error(exc) from None

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    temp_path: Path | None = None
    try:
        if file is not None:
            filename = Path(file.filename or doc["filename"]).name
            suffix = Path(filename).suffix.lower() or ".txt"
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            fd, raw_temp = tempfile.mkstemp(
                dir=str(UPLOADS_DIR),
                prefix=".index_",
                suffix=suffix,
            )
            temp_path = Path(raw_temp)
            total_bytes = 0
            with os.fdopen(fd, "wb") as handle:
                while True:
                    chunk = await file.read(256 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                            detail="Uploaded file exceeds the 10 MiB size limit.",
                        )
                    handle.write(chunk)
            target_path = temp_path
            logical_name = filename
        else:
            candidate = (UPLOADS_DIR / Path(str(doc["filename"])).name).resolve()
            uploads_resolved = UPLOADS_DIR.resolve()
            try:
                candidate.relative_to(uploads_resolved)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Invalid stored filename.",
                ) from None
            if not candidate.is_file():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Source file is not present on disk; upload it with this request.",
                )
            target_path = candidate
            logical_name = candidate.name

        summary = index_approved_document(
            target_path,
            document_id,
            collections=collections,
            actor_id=principal.user_id,
            logical_filename=logical_name,
        )
        return {"success": True, **summary}
    except HTTPException:
        raise
    except local_rag_store.LocalRagStoreError as exc:
        raise _handle_store_error(exc) from None
    except Exception:
        logger.warning("Approved document indexing failed.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document could not be indexed in its current state.",
        ) from None
    finally:
        if file is not None:
            await file.close()
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Failed to remove temporary indexing file.")
