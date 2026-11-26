"""Backend-only PostgREST client for local_rag_* Supabase tables and RPCs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import httpx

from src.api.services.identity_service import _service_key_kind
from src.config import settings


_REQUEST_TIMEOUT_SECONDS = 10.0


class LocalRagStoreError(RuntimeError):
    """Base error for Supabase local_rag_* store operations."""


class LocalRagStoreNotConfigured(LocalRagStoreError):
    """Required server-side Supabase configuration is missing or invalid."""


class LocalRagStoreUnavailable(LocalRagStoreError):
    """Supabase PostgREST could not be reached or returned an unexpected error."""


class LocalRagStoreConflict(LocalRagStoreError):
    """Supabase rejected an operation due to a state or constraint conflict."""


def _supabase_headers_and_base() -> tuple[str, dict[str, str]]:
    project_url = settings.SUPABASE_URL
    service_role_key = settings.SUPABASE_SERVICE_ROLE_KEY
    if (
        not isinstance(project_url, str)
        or not project_url.strip()
        or not isinstance(service_role_key, str)
        or not service_role_key.strip()
    ):
        raise LocalRagStoreNotConfigured(
            "Supabase local RAG store is not configured."
        )

    key_kind = _service_key_kind(service_role_key.strip())
    if key_kind is None:
        raise LocalRagStoreNotConfigured(
            "A Supabase service-role credential is required."
        )

    key = service_role_key.strip()
    headers: dict[str, str] = {
        "apikey": key,
        "Content-Type": "application/json",
    }
    if key_kind == "legacy":
        headers["Authorization"] = f"Bearer {key}"
    else:
        headers["Authorization"] = f"Bearer {key}"

    return project_url.rstrip("/"), headers


def _normalize_uuid(value: object, field_name: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID.") from exc


def _call_rpc(
    rpc_name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    base_url, headers = _supabase_headers_and_base()
    endpoint = f"{base_url}/rest/v1/rpc/{rpc_name}"
    try:
        response = httpx.post(
            endpoint,
            json=dict(payload),
            headers=headers,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=False,
        )
    except httpx.RequestError as exc:
        raise LocalRagStoreUnavailable(
            f"Supabase RPC '{rpc_name}' request failed."
        ) from exc

    if response.status_code in {400, 404, 409, 422}:
        raise LocalRagStoreConflict(
            f"Supabase RPC '{rpc_name}' rejected the request (HTTP {response.status_code})."
        )
    if response.status_code != 200:
        raise LocalRagStoreUnavailable(
            f"Supabase RPC '{rpc_name}' failed with HTTP {response.status_code}."
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise LocalRagStoreUnavailable(
            f"Supabase RPC '{rpc_name}' returned invalid JSON."
        ) from exc

    if not isinstance(data, Mapping):
        raise LocalRagStoreUnavailable(
            f"Supabase RPC '{rpc_name}' returned an unexpected payload."
        )
    return dict(data)


def _select_rows(
    table_name: str,
    params: Mapping[str, str],
) -> list[dict[str, Any]]:
    base_url, headers = _supabase_headers_and_base()
    endpoint = f"{base_url}/rest/v1/{table_name}"
    try:
        response = httpx.get(
            endpoint,
            params=dict(params),
            headers=headers,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=False,
        )
    except httpx.RequestError as exc:
        raise LocalRagStoreUnavailable(
            f"Supabase query on '{table_name}' failed."
        ) from exc

    if response.status_code != 200:
        raise LocalRagStoreUnavailable(
            f"Supabase query on '{table_name}' failed with HTTP {response.status_code}."
        )

    try:
        rows = response.json()
    except ValueError as exc:
        raise LocalRagStoreUnavailable(
            f"Supabase query on '{table_name}' returned invalid JSON."
        ) from exc

    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise LocalRagStoreUnavailable(
            f"Supabase query on '{table_name}' returned malformed rows."
        )
    return [dict(row) for row in rows]


def get_document_by_hash(file_hash_sha256: str) -> dict[str, Any] | None:
    """Return one local_rag_documents row by canonical lowercase SHA-256 hash."""

    normalized_hash = file_hash_sha256.strip().lower()
    rows = _select_rows(
        "local_rag_documents",
        {
            "select": (
                "id,file_hash_sha256,source_id,filename,approval_status,"
                "indexing_status,approval_version,created_at,updated_at"
            ),
            "file_hash_sha256": f"eq.{normalized_hash}",
            "limit": "2",
        },
    )
    if not rows:
        return None
    if len(rows) > 1:
        raise LocalRagStoreUnavailable("Duplicate document rows returned for hash.")
    return rows[0]


def get_document_by_id(document_id: str) -> dict[str, Any] | None:
    """Return one local_rag_documents row by canonical UUID."""

    normalized_id = _normalize_uuid(document_id, "document_id")
    rows = _select_rows(
        "local_rag_documents",
        {
            "select": (
                "id,file_hash_sha256,source_id,filename,legacy_chroma_document_id,"
                "approval_status,indexing_status,approval_version,uploaded_by,"
                "approved_by,approved_at,rejected_by,rejected_at,revoked_by,"
                "revoked_at,superseded_by_document_id,superseded_at,"
                "decision_reason,created_at,updated_at"
            ),
            "id": f"eq.{normalized_id}",
            "limit": "2",
        },
    )
    if not rows:
        return None
    if len(rows) > 1:
        raise LocalRagStoreUnavailable("Duplicate document rows returned for ID.")
    return rows[0]


def list_documents(*, limit: int = 100) -> list[dict[str, Any]]:
    """Return local_rag_documents rows ordered by most recently updated."""

    bounded_limit = max(1, min(limit, 500))
    return _select_rows(
        "local_rag_documents",
        {
            "select": (
                "id,file_hash_sha256,source_id,filename,approval_status,"
                "indexing_status,approval_version,approved_by,approved_at,"
                "rejected_by,rejected_at,revoked_by,revoked_at,"
                "decision_reason,created_at,updated_at"
            ),
            "order": "updated_at.desc",
            "limit": str(bounded_limit),
        },
    )


def get_document_sections(document_id: str) -> list[dict[str, Any]]:
    """Return all sections for one local_rag_documents UUID."""

    normalized_id = _normalize_uuid(document_id, "document_id")
    return _select_rows(
        "local_rag_document_sections",
        {
            "select": (
                "id,document_id,section_index,section_title,page_number,"
                "section_sha256,proposed_roles,approved_roles,classification_method,"
                "classification_confidence,classifier_quarantined,review_status,"
                "quarantine_resolution_reason,exclusion_reason,chunk_ids_by_role"
            ),
            "document_id": f"eq.{normalized_id}",
            "order": "section_index.asc",
        },
    )


def register_document(
    *,
    file_hash_sha256: str,
    source_id: str,
    filename: str,
    sections: Sequence[Mapping[str, Any]],
    actor_id: str | None = None,
    legacy_chroma_document_id: str | None = None,
) -> dict[str, Any]:
    """Register or re-register a document and its classified sections as pending."""

    payload: dict[str, Any] = {
        "p_file_hash_sha256": file_hash_sha256.strip().lower(),
        "p_source_id": source_id,
        "p_filename": filename,
        "p_sections": [dict(section) for section in sections],
        "p_actor_id": (
            _normalize_uuid(actor_id, "actor_id")
            if actor_id is not None
            else None
        ),
        "p_legacy_chroma_document_id": legacy_chroma_document_id,
    }
    return _call_rpc("local_rag_register_document", payload)


def approve_document(
    *,
    document_id: str,
    actor_id: str,
    section_decisions: Sequence[Mapping[str, Any]],
    decision_reason: str | None = None,
) -> dict[str, Any]:
    """Atomically approve or exclude every section of a pending document."""

    return _call_rpc(
        "local_rag_approve_document",
        {
            "p_document_id": _normalize_uuid(document_id, "document_id"),
            "p_actor_id": _normalize_uuid(actor_id, "actor_id"),
            "p_section_decisions": [dict(item) for item in section_decisions],
            "p_decision_reason": decision_reason,
        },
    )


def reject_or_revoke_document(
    *,
    document_id: str,
    action: str,
    actor_id: str,
    decision_reason: str,
) -> dict[str, Any]:
    """Reject a pending document or revoke an approved document."""

    if action not in {"rejected", "revoked"}:
        raise ValueError("action must be 'rejected' or 'revoked'.")
    return _call_rpc(
        "local_rag_reject_or_revoke_document",
        {
            "p_document_id": _normalize_uuid(document_id, "document_id"),
            "p_action": action,
            "p_actor_id": _normalize_uuid(actor_id, "actor_id"),
            "p_decision_reason": decision_reason,
        },
    )


def start_indexing(
    *,
    document_id: str,
    expected_approval_version: int,
    expected_file_hash: str,
    actor_id: str | None = None,
) -> dict[str, Any]:
    """Transition an approved document into indexing state for a bound job."""

    return _call_rpc(
        "local_rag_start_indexing",
        {
            "p_document_id": _normalize_uuid(document_id, "document_id"),
            "p_expected_approval_version": expected_approval_version,
            "p_expected_file_hash": expected_file_hash.strip().lower(),
            "p_actor_id": (
                _normalize_uuid(actor_id, "actor_id")
                if actor_id is not None
                else None
            ),
        },
    )


def complete_indexing(
    *,
    document_id: str,
    expected_approval_version: int,
    expected_file_hash: str,
    section_chunks: Sequence[Mapping[str, Any]],
    details: Mapping[str, Any],
    actor_id: str | None = None,
) -> dict[str, Any]:
    """Record backend-attested ChromaDB indexing verification in Supabase."""

    return _call_rpc(
        "local_rag_complete_indexing",
        {
            "p_document_id": _normalize_uuid(document_id, "document_id"),
            "p_expected_approval_version": expected_approval_version,
            "p_expected_file_hash": expected_file_hash.strip().lower(),
            "p_section_chunks": [dict(item) for item in section_chunks],
            "p_details": dict(details),
            "p_actor_id": (
                _normalize_uuid(actor_id, "actor_id")
                if actor_id is not None
                else None
            ),
        },
    )


def fail_indexing(
    *,
    document_id: str,
    expected_approval_version: int,
    expected_file_hash: str,
    error_details: Mapping[str, Any],
    actor_id: str | None = None,
) -> dict[str, Any]:
    """Mark an in-progress indexing job as failed with sanitized error details."""

    return _call_rpc(
        "local_rag_fail_indexing",
        {
            "p_document_id": _normalize_uuid(document_id, "document_id"),
            "p_expected_approval_version": expected_approval_version,
            "p_expected_file_hash": expected_file_hash.strip().lower(),
            "p_error_details": dict(error_details),
            "p_actor_id": (
                _normalize_uuid(actor_id, "actor_id")
                if actor_id is not None
                else None
            ),
        },
    )


def verify_chunks_for_retrieval(
    chunk_refs: Sequence[tuple[str, str]],
    requesting_role: str,
) -> set[tuple[str, str]]:
    """Batch-verify candidate (document_id, section_id) pairs against Supabase.

    Fails closed (raises LocalRagStoreError) if Supabase is unconfigured,
    unavailable, or returns malformed data. Admin bypasses role-membership
    restrictions only, never approval, indexing, or section-review gates.
    """

    from src.security.access_control import validate_role

    validated_role = validate_role(requesting_role)
    if not chunk_refs:
        return set()

    valid_pairs: list[tuple[str, str]] = []
    for raw_doc_id, raw_sec_id in chunk_refs:
        try:
            doc_uuid = str(UUID(str(raw_doc_id)))
            sec_uuid = str(UUID(str(raw_sec_id)))
        except (TypeError, ValueError, AttributeError):
            continue
        valid_pairs.append((doc_uuid, sec_uuid))

    if not valid_pairs:
        return set()

    unique_doc_ids = sorted({doc_id for doc_id, _ in valid_pairs})
    unique_sec_ids = sorted({sec_id for _, sec_id in valid_pairs})

    docs_by_id: dict[str, dict[str, Any]] = {}
    for start in range(0, len(unique_doc_ids), 100):
        batch = unique_doc_ids[start : start + 100]
        rows = _select_rows(
            "local_rag_documents",
            {
                "select": "id,approval_status,indexing_status",
                "id": f"in.({','.join(batch)})",
            },
        )
        for row in rows:
            row_id = _normalize_uuid(row.get("id"), "local_rag_documents.id")
            docs_by_id[row_id] = row

    secs_by_id: dict[str, dict[str, Any]] = {}
    for start in range(0, len(unique_sec_ids), 100):
        batch = unique_sec_ids[start : start + 100]
        rows = _select_rows(
            "local_rag_document_sections",
            {
                "select": "id,document_id,review_status,approved_roles",
                "id": f"in.({','.join(batch)})",
            },
        )
        for row in rows:
            sec_id = _normalize_uuid(row.get("id"), "local_rag_document_sections.id")
            secs_by_id[sec_id] = row

    authorized: set[tuple[str, str]] = set()
    for doc_id, sec_id in valid_pairs:
        doc = docs_by_id.get(doc_id)
        sec = secs_by_id.get(sec_id)
        if doc is None or sec is None:
            continue
        if (
            doc.get("approval_status") != "approved"
            or doc.get("indexing_status") != "indexed"
        ):
            continue
        try:
            sec_doc_id = str(UUID(str(sec.get("document_id"))))
        except (TypeError, ValueError, AttributeError):
            continue
        if sec_doc_id != doc_id:
            continue
        if sec.get("review_status") != "approved":
            continue
        approved_roles = sec.get("approved_roles")
        if not isinstance(approved_roles, list) or not approved_roles:
            continue
        if validated_role != "admin" and validated_role not in approved_roles:
            continue
        authorized.add((doc_id, sec_id))

    return authorized
