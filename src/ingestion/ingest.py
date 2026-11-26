"""Document ingestion for classified FinSight AI sections."""

import hashlib
import hmac
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from langchain_core.documents import Document

from src.config.settings import (
    ALLOWED_ROLES,
    INGESTION_MANIFEST_HMAC_KEY,
    PROJECT_ROOT,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    UNCLASSIFIED_DIR,
    UPLOADS_DIR,
)
from src.ingestion.document_loader import load_document
from src.ingestion.role_classifier import (
    ClassificationResult,
    MINIMUM_CONFIDENCE_THRESHOLD,
    classify_section,
)
from src.ingestion.text_splitter import split_classified_section
from src.vectorstore.chroma_store import (
    initialize_collections,
    normalize_chroma_metadata,
)


MANIFEST_PATH: Path = PROJECT_ROOT / "data" / "ingestion_manifest.json"
MANIFEST_SIGNING_KEY = INGESTION_MANIFEST_HMAC_KEY
_SUPPORTED_CLASSIFICATION_METHODS = frozenset({"groq_llm"})

_HEADING_PATTERN = re.compile(
    r"(?:^#{1,6}[ \t]+\S.*$)"
    r"|(?:^\d+(?:\.\d+)*[.)][ \t]+\S.*$)",
    re.MULTILINE,
)


def _document_id(resolved_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_source_id(resolved_path).encode("ascii"))
    digest.update(b"\0")
    with resolved_path.open("rb") as document_file:
        for file_chunk in iter(
            lambda: document_file.read(1024 * 1024),
            b"",
        ):
            digest.update(file_chunk)
    return digest.hexdigest()


def _source_id(resolved_path: Path) -> str:
    source_identity = str(resolved_path).casefold().encode("utf-8")
    return hashlib.sha256(source_identity).hexdigest()


def _chunk_id(
    document_id: str,
    section_index: int,
    chunk_index: int,
) -> str:
    identity = f"{document_id}|section={section_index}|chunk={chunk_index}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"documents": {}}

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"The ingestion manifest is malformed: {MANIFEST_PATH}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Unable to read the ingestion manifest: {MANIFEST_PATH}"
        ) from exc

    if not isinstance(manifest, dict):
        raise RuntimeError("The ingestion manifest must contain a JSON object.")
    if not isinstance(manifest.get("documents"), dict):
        manifest["documents"] = {}
    return manifest


def _save_manifest(manifest: Mapping[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MANIFEST_PATH.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(MANIFEST_PATH)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to update the ingestion manifest: {MANIFEST_PATH}"
        ) from exc


def _one_based_page(page_value: Any) -> int:
    if isinstance(page_value, bool):
        return 1
    if isinstance(page_value, int):
        return page_value + 1 if page_value >= 0 else 1
    if isinstance(page_value, str):
        digits = re.findall(r"\d+", page_value)
        if digits:
            parsed = int(digits[-1])
            return parsed if parsed >= 1 else 1
    return 1


def _identify_sections(
    loaded_documents: list[Document],
    source_name: str,
) -> tuple[list[dict[str, Any]], str, list[tuple[int, int, int]]]:
    """Build combined text and exact offsets together with page boundaries."""

    combined_parts: list[str] = []
    page_boundaries: list[tuple[int, int, int]] = []
    combined_offset = 0

    for loaded_document in loaded_documents:
        text = loaded_document.page_content
        if not text:
            continue

        if combined_parts:
            combined_parts.append("\n\n")
            combined_offset += 2

        start_offset = combined_offset
        combined_parts.append(text)
        combined_offset += len(text)
        end_offset = combined_offset

        metadata = loaded_document.metadata
        raw_page = metadata.get(
            "page",
            metadata.get("page_number", 0),
        )
        page_boundaries.append(
            (start_offset, end_offset, _one_based_page(raw_page))
        )

    combined_text = "".join(combined_parts)
    if not combined_text.strip():
        raise ValueError("The loaded document contains no readable text.")

    heading_matches = list(_HEADING_PATTERN.finditer(combined_text))
    if not heading_matches:
        return (
            [{
                "title": f"Fallback: entire {source_name}",
                "text": combined_text,
                "start_offset": 0,
                "end_offset": len(combined_text),
                "page_number": (
                    page_boundaries[0][2] if page_boundaries else 1
                ),
            }],
            "document_fallback_no_reliable_headings",
            page_boundaries,
        )

    sections: list[dict[str, Any]] = []
    first_heading_start = heading_matches[0].start()

    if first_heading_start > 0:
        raw_pre_heading = combined_text[:first_heading_start]
        leading_trim = len(raw_pre_heading) - len(raw_pre_heading.lstrip())
        trailing_trim = len(raw_pre_heading) - len(raw_pre_heading.rstrip())
        pre_heading_start = leading_trim
        pre_heading_end = first_heading_start - trailing_trim
        pre_heading_text = combined_text[pre_heading_start:pre_heading_end]
        if pre_heading_text:
            sections.append({
                "title": f"Pre-heading content: {source_name}",
                "text": pre_heading_text,
                "start_offset": pre_heading_start,
                "end_offset": pre_heading_end,
                "page_number": _page_number_at_offset(
                    pre_heading_start,
                    page_boundaries,
                ),
            })

    for match_index, heading_match in enumerate(heading_matches):
        raw_start = heading_match.start()
        section_end = (
            heading_matches[match_index + 1].start()
            if match_index + 1 < len(heading_matches)
            else len(combined_text)
        )

        raw_section = combined_text[raw_start:section_end]
        section_left_trimmed = raw_section.lstrip()
        start_offset = raw_start + (
            len(raw_section) - len(section_left_trimmed)
        )
        section_text = section_left_trimmed.rstrip()

        if not section_text:
            continue

        title = heading_match.group(0).strip()
        title = re.sub(r"^#{1,6}[ \t]+", "", title).strip()

        sections.append({
            "title": title,
            "text": section_text,
            "start_offset": start_offset,
            "end_offset": start_offset + len(section_text),
            "page_number": _page_number_at_offset(
                start_offset,
                page_boundaries,
            ),
        })

    if not sections:
        return (
            [{
                "title": f"Fallback: entire {source_name}",
                "text": combined_text,
                "start_offset": 0,
                "end_offset": len(combined_text),
                "page_number": (
                    page_boundaries[0][2] if page_boundaries else 1
                ),
            }],
            "document_fallback_no_reliable_headings",
            page_boundaries,
        )

    return sections, "reliable_heading_boundary", page_boundaries


def _page_number_at_offset(
    offset: int,
    page_boundaries: list[tuple[int, int, int]],
) -> int:
    for start, end, page_number in page_boundaries:
        if start <= offset < end:
            return page_number
        if offset < start:
            return page_number
    if page_boundaries:
        return page_boundaries[-1][2]
    return 1


def _quarantine_section(
    document_id: str,
    section_index: int,
    section_title: str,
    section_text: str,
    page_number: int,
    reason: str,
) -> str:
    UNCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    quarantine_path = UNCLASSIFIED_DIR / (
        f"{document_id}-section-{section_index}.json"
    )
    quarantine_data = {
        "document_id": document_id,
        "section_index": section_index,
        "section_title": section_title,
        "page_number": page_number,
        "section_text": section_text,
        "reason": reason,
    }
    try:
        quarantine_path.write_text(
            json.dumps(quarantine_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(
            f"Failed to save quarantined section {section_index}: {exc}"
        ) from exc
    return str(quarantine_path)


def _collection_already_has_id(
    collection: Any,
    chunk_id: str,
) -> bool:
    if not hasattr(collection, "get"):
        raise RuntimeError(
            "Unable to verify duplicate chunk IDs: collection has no get()."
        )
    try:
        existing = collection.get(ids=[chunk_id])
    except Exception as exc:
        raise RuntimeError(
            f"Unable to verify existing chunk ID '{chunk_id}' in ChromaDB."
        ) from exc
    if not isinstance(existing, dict) or "ids" not in existing:
        raise RuntimeError(
            f"ChromaDB returned an invalid result for chunk ID '{chunk_id}'."
        )
    ids = existing["ids"]
    if not isinstance(ids, (list, tuple)):
        raise RuntimeError(
            f"ChromaDB returned invalid IDs for chunk ID '{chunk_id}'."
        )
    return chunk_id in ids


def _validate_classification(
    classification: Any,
) -> ClassificationResult | None:
    if not isinstance(classification, ClassificationResult):
        return None
    if not isinstance(classification.review_required, bool):
        return None
    if classification.review_required:
        return None

    confidence = classification.classification_confidence
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or confidence < MINIMUM_CONFIDENCE_THRESHOLD
        or confidence > 1.0
        or not math.isfinite(confidence)
    ):
        return None
    if not isinstance(classification.allowed_roles, list):
        return None
    if not isinstance(classification.section_title, str):
        return None
    if not classification.section_title.strip():
        return None
    if not isinstance(classification.classification_method, str):
        return None
    if classification.classification_method not in (
        _SUPPORTED_CLASSIFICATION_METHODS
    ):
        return None
    if not isinstance(classification.section_text, str):
        return None
    if not classification.section_text.strip():
        return None

    roles = classification.allowed_roles
    if not roles:
        return None
    if any(not isinstance(role, str) or not role.strip() for role in roles):
        return None
    if len(set(roles)) != len(roles):
        return None
    if any(role not in ALLOWED_ROLES for role in roles):
        return None

    return classification


def _section_text_sha256(section_text: str) -> str:
    return hashlib.sha256(section_text.encode("utf-8")).hexdigest()


def _manifest_signing_key() -> bytes:
    if (
        not isinstance(MANIFEST_SIGNING_KEY, str)
        or len(MANIFEST_SIGNING_KEY) < 32
    ):
        raise RuntimeError(
            "INGESTION_MANIFEST_HMAC_KEY must contain at least 32 characters."
        )
    return MANIFEST_SIGNING_KEY.encode("utf-8")


def _decision_signature(
    document_id: str,
    section_index: int,
    section_record: Mapping[str, Any],
) -> str:
    signed_fields = {
        "document_id": document_id,
        "section_index": section_index,
        "allowed_roles": section_record.get("allowed_roles"),
        "classification_method": section_record.get(
            "classification_method"
        ),
        "classification_confidence": section_record.get(
            "classification_confidence"
        ),
        "classification_decision_status": section_record.get(
            "classification_decision_status"
        ),
        "section_text_sha256": section_record.get("section_text_sha256"),
    }
    try:
        payload = json.dumps(
            signed_fields,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Unable to authenticate the saved classification decision."
        ) from exc

    return hmac.new(
        _manifest_signing_key(),
        payload,
        hashlib.sha256,
    ).hexdigest()


def _saved_classification(
    section_record: Mapping[str, Any],
    document_id: str,
    section_index: int,
    section_title: str,
    section_text: str,
) -> tuple[bool, ClassificationResult | None, str]:
    """Validate and reconstruct a previously persisted authorization decision."""

    decision_fields = {
        "allowed_roles",
        "classification_method",
        "classification_confidence",
        "classification_decision_status",
        "section_text_sha256",
        "classification_decision_signature",
    }
    if not any(field in section_record for field in decision_fields):
        return False, None, ""

    if not all(field in section_record for field in decision_fields):
        return True, None, "saved_classification_invalid: incomplete decision"
    if section_record.get("classification_decision_status") != "validated":
        return True, None, "saved_classification_invalid: invalid decision status"
    if section_record.get("section_text_sha256") != _section_text_sha256(
        section_text
    ):
        return True, None, "saved_classification_text_hash_mismatch"

    classification = ClassificationResult(
        section_title=section_title,
        section_text=section_text,
        allowed_roles=section_record.get("allowed_roles"),
        classification_method=section_record.get("classification_method"),
        classification_confidence=section_record.get(
            "classification_confidence"
        ),
        review_required=False,
    )
    validated = _validate_classification(classification)
    if validated is None:
        return True, None, "saved_classification_invalid: invalid fields"

    try:
        expected_signature = _decision_signature(
            document_id,
            section_index,
            section_record,
        )
    except RuntimeError as exc:
        return True, None, f"saved_classification_invalid: {exc}"
    saved_signature = section_record.get("classification_decision_signature")
    if (
        not isinstance(saved_signature, str)
        or not hmac.compare_digest(saved_signature, expected_signature)
    ):
        return True, None, "saved_classification_invalid: signature mismatch"

    return True, validated, ""


def _save_classification_decision(
    section_record: dict[str, Any],
    classification: ClassificationResult,
    document_id: str,
    section_index: int,
    section_text: str,
) -> None:
    section_record.update({
        "allowed_roles": list(classification.allowed_roles),
        "classification_method": classification.classification_method,
        "classification_confidence": classification.classification_confidence,
        "classification_decision_status": "validated",
        "section_text_sha256": _section_text_sha256(section_text),
    })
    section_record["classification_decision_signature"] = (
        _decision_signature(document_id, section_index, section_record)
    )
    section_record.pop("retryable_error", None)


def _section_has_recorded_chunks(section_record: Mapping[str, Any]) -> bool:
    chunk_ids_by_role = section_record.get("chunk_ids_by_role")
    return (
        isinstance(chunk_ids_by_role, dict)
        and any(
            isinstance(chunk_ids, list) and chunk_ids
            for chunk_ids in chunk_ids_by_role.values()
        )
    )


def _manifest_entry(
    manifest: dict[str, Any],
    document_id: str,
    source_name: str,
    source_id: str,
) -> dict[str, Any]:
    documents = manifest["documents"]
    entry = documents.get(document_id)
    if not isinstance(entry, dict):
        entry = {
            "document_id": document_id,
            "source": source_name,
            "source_id": source_id,
            "status": "in_progress",
            "retryable": True,
            "sections_processed": 0,
            "chunks_stored_per_role": {
                role: 0 for role in ALLOWED_ROLES
            },
            "chunk_ids_by_role": {
                role: [] for role in ALLOWED_ROLES
            },
            "quarantined_sections": [],
            "sections": {},
        }
        documents[document_id] = entry
    elif entry.get("source_id", source_id) != source_id:
        raise RuntimeError(
            "The ingestion manifest source identity does not match the file."
        )

    if entry.get("status") not in {
        "in_progress",
        "completed",
        "needs_review",
        "partial",
    }:
        entry["status"] = "partial"
        entry["retryable"] = True

    raw_ids_by_role = entry.get("chunk_ids_by_role")
    if not isinstance(raw_ids_by_role, dict):
        raw_ids_by_role = {}
    entry["chunk_ids_by_role"] = {}
    for role in ALLOWED_ROLES:
        role_ids = raw_ids_by_role.get(role)
        if not isinstance(role_ids, list):
            role_ids = []
        entry["chunk_ids_by_role"][role] = list(dict.fromkeys(
            chunk_id
            for chunk_id in role_ids
            if isinstance(chunk_id, str)
        ))

    if not isinstance(entry.get("quarantined_sections"), list):
        entry["quarantined_sections"] = []
    entry["quarantined_sections"] = [
        item
        for item in entry["quarantined_sections"]
        if isinstance(item, dict)
    ]
    if not isinstance(entry.get("sections"), dict):
        entry["sections"] = {}
    for section_record in entry["sections"].values():
        if not isinstance(section_record, dict):
            continue
        role_chunk_ids = section_record.get("chunk_ids_by_role")
        if not isinstance(role_chunk_ids, dict):
            role_chunk_ids = {}
        section_record["chunk_ids_by_role"] = {
            role: list(dict.fromkeys(
                chunk_id
                for chunk_id in role_chunk_ids.get(role, [])
                if isinstance(chunk_id, str)
            ))
            if isinstance(role_chunk_ids.get(role, []), list)
            else []
            for role in ALLOWED_ROLES
        }

    processed = entry.get("sections_processed")
    if not isinstance(processed, int) or isinstance(processed, bool) or processed < 0:
        entry["sections_processed"] = 0
    return entry


def _reconcile_manifest_entry(
    collections: Mapping[str, Any],
    entry: dict[str, Any],
) -> None:
    """Reconcile stored chunk counts against IDs actually present."""

    chunk_ids_by_role = entry["chunk_ids_by_role"]
    counts: dict[str, int] = {}

    for role in ALLOWED_ROLES:
        stored_ids: list[str] = []
        recorded_ids = chunk_ids_by_role.get(role, [])
        if not isinstance(recorded_ids, list):
            recorded_ids = []

        for chunk_id in recorded_ids:
            if not isinstance(chunk_id, str):
                continue
            if _collection_already_has_id(collections[role], chunk_id):
                stored_ids.append(chunk_id)

        chunk_ids_by_role[role] = list(dict.fromkeys(stored_ids))
        counts[role] = len(stored_ids)

    entry["chunks_stored_per_role"] = counts


def _delete_collection_chunks(
    collection: Any,
    where: Mapping[str, Any],
    keep_document_id: str | None = None,
) -> bool:
    if not hasattr(collection, "get") or not hasattr(collection, "delete"):
        raise RuntimeError(
            "Unable to remove stale chunks: collection must support get() "
            "and delete()."
        )
    try:
        result = collection.get(where=dict(where), include=["metadatas"])
    except Exception as exc:
        raise RuntimeError("Unable to find stale chunks in ChromaDB.") from exc

    if (
        not isinstance(result, dict)
        or not isinstance(result.get("ids"), list)
        or not isinstance(result.get("metadatas"), list)
        or len(result["ids"]) != len(result["metadatas"])
    ):
        raise RuntimeError("ChromaDB returned invalid stale-chunk data.")

    chunk_ids = []
    for chunk_id, metadata in zip(result["ids"], result["metadatas"]):
        if not isinstance(chunk_id, str):
            raise RuntimeError("ChromaDB returned an invalid stale chunk ID.")
        if (
            keep_document_id is not None
            and isinstance(metadata, dict)
            and metadata.get("document_id") == keep_document_id
        ):
            continue
        chunk_ids.append(chunk_id)

    if not chunk_ids:
        return False
    try:
        collection.delete(ids=chunk_ids)
    except Exception as exc:
        raise RuntimeError("Unable to remove stale chunks from ChromaDB.") from exc
    return True


def _remove_superseded_source_documents(
    manifest: dict[str, Any],
    collections: Mapping[str, Any],
    document_id: str,
    source_id: str,
    source_name: str,
) -> None:
    documents = manifest["documents"]
    superseded_ids = [
        old_document_id
        for old_document_id, old_entry in documents.items()
        if (
            old_document_id != document_id
            and isinstance(old_entry, dict)
            and (
                old_entry.get("source_id") == source_id
                or (
                    not old_entry.get("source_id")
                    and old_entry.get("source") == source_name
                )
            )
        )
    ]

    removed_chunks = False
    for role in ALLOWED_ROLES:
        removed_chunks = _delete_collection_chunks(
            collections[role],
            {"source_id": source_id},
            keep_document_id=document_id,
        ) or removed_chunks
        for old_document_id in superseded_ids:
            removed_chunks = _delete_collection_chunks(
                collections[role],
                {"document_id": old_document_id},
            ) or removed_chunks

    for old_document_id in superseded_ids:
        documents.pop(old_document_id, None)
    if superseded_ids or removed_chunks:
        _save_manifest(manifest)


def _ingestion_summary(
    document_id: str,
    entry: dict[str, Any],
    duplicate: bool,
    boundary_method: str,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "sections_processed": entry["sections_processed"],
        "chunks_stored_per_role": entry["chunks_stored_per_role"],
        "quarantined_sections": entry["quarantined_sections"],
        "duplicate": duplicate,
        "section_boundary_method": boundary_method,
    }


def ingest_document(
    document_path: Path,
    retry_needs_review: bool = False,
) -> dict[str, Any]:
    """Load, classify, split, quarantine, and route one uploaded document."""

    if not isinstance(document_path, Path):
        raise TypeError("document_path must be a pathlib.Path")

    _manifest_signing_key()
    resolved_path = document_path.resolve(strict=True)
    loaded_documents = load_document(resolved_path)
    source_id = _source_id(resolved_path)
    document_id = _document_id(resolved_path)
    source_name = resolved_path.name

    manifest = _read_manifest()
    entry = _manifest_entry(manifest, document_id, source_name, source_id)
    status = entry["status"]
    collections = initialize_collections()
    try:
        _remove_superseded_source_documents(
            manifest,
            collections,
            document_id,
            source_id,
            source_name,
        )
    except RuntimeError as exc:
        entry["status"] = "partial"
        entry["retryable"] = True
        entry["retryable_error"] = f"stale_chunk_cleanup_failed: {exc}"
        _save_manifest(manifest)
        raise

    if status == "completed":
        _reconcile_manifest_entry(collections, entry)
        _save_manifest(manifest)
        return _ingestion_summary(
            document_id,
            entry,
            True,
            entry.get("section_boundary_method", "manifest_duplicate"),
        )

    if status == "needs_review" and not retry_needs_review:
        _reconcile_manifest_entry(collections, entry)
        _save_manifest(manifest)
        summary = _ingestion_summary(
            document_id,
            entry,
            True,
            entry.get("section_boundary_method", "manifest_needs_review"),
        )
        summary["needs_review"] = True
        return summary

    sections, boundary_method, page_boundaries = _identify_sections(
        loaded_documents,
        source_name,
    )
    entry["status"] = "in_progress"
    entry["retryable"] = True
    entry["section_boundary_method"] = boundary_method
    entry["section_count"] = len(sections)
    retried_section_keys: set[str] = set()
    if retry_needs_review:
        retried_section_keys = {
            str(section_index)
            for section_index in range(len(sections))
            if isinstance(entry["sections"].get(str(section_index)), dict)
            and entry["sections"][str(section_index)].get("status")
            == "quarantined"
        }
    for section_key in list(entry["sections"].keys()):
        section_record = entry["sections"].get(section_key)
        if not isinstance(section_record, dict):
            continue
        if section_key in retried_section_keys:
            section_record["status"] = "in_progress"
            section_record["quarantine_path"] = None

    if retry_needs_review:
        entry["quarantined_sections"] = [
            item
            for item in entry["quarantined_sections"]
            if str(item.get("section_index")) not in retried_section_keys
        ]

    _save_manifest(manifest)

    current_section_record: dict[str, Any] | None = None
    try:
        _reconcile_manifest_entry(collections, entry)
        _save_manifest(manifest)

        for section_index, section_data in enumerate(sections):
            section_key = str(section_index)
            section_record = entry["sections"].get(section_key)
            if not isinstance(section_record, dict):
                section_record = {
                    "status": "in_progress",
                    "title": section_data["title"],
                    "page_number": section_data["page_number"],
                    "chunk_ids_by_role": {},
                    "quarantine_path": None,
                }
                entry["sections"][section_key] = section_record
                _save_manifest(manifest)
            current_section_record = section_record

            is_review_retry = section_key in retried_section_keys
            if (
                section_record.get("status") in {"completed", "quarantined"}
                and not is_review_retry
            ):
                continue

            section_title = section_data["title"]
            section_text = section_data["text"]
            section_page = int(section_data["page_number"])

            has_saved_decision, classification, decision_error = (
                _saved_classification(
                    section_record,
                    document_id,
                    section_index,
                    section_title,
                    section_text,
                )
            )
            if is_review_retry and (
                has_saved_decision or _section_has_recorded_chunks(section_record)
            ):
                decision_error = (
                    "saved_classification_invalid: refusing to replace an "
                    "authorization decision for an indexed section"
                )
                section_record["retryable_error"] = decision_error
                _save_manifest(manifest)
                raise RuntimeError(decision_error)

            if has_saved_decision and classification is None:
                section_record["retryable_error"] = decision_error
                _save_manifest(manifest)
                raise RuntimeError(decision_error)

            if not has_saved_decision:
                if _section_has_recorded_chunks(section_record):
                    decision_error = (
                        "saved_classification_invalid: authorization decision "
                        "is missing for an indexed section"
                    )
                    section_record["retryable_error"] = decision_error
                    _save_manifest(manifest)
                    raise RuntimeError(decision_error)

                try:
                    classification = classify_section(
                        section_title,
                        section_text,
                    )
                except ValueError as exc:
                    classification = None
                    classification_error = str(exc)
                else:
                    classification_error = ""

                classification = _validate_classification(classification)
            if classification is None:
                reason = "invalid_uncertain_low_confidence"
                if not has_saved_decision and classification_error:
                    reason = f"invalid_classification: {classification_error}"
                quarantine_path = _quarantine_section(
                    document_id,
                    section_index,
                    section_title,
                    section_text,
                    section_page,
                    reason,
                )
                quarantine_entry = {
                    "section_index": section_index,
                    "section_title": section_title,
                    "page_number": section_page,
                    "path": quarantine_path,
                    "reason": reason,
                }
                if quarantine_entry not in entry["quarantined_sections"]:
                    entry["quarantined_sections"].append(quarantine_entry)
                section_record["status"] = "quarantined"
                section_record["quarantine_path"] = quarantine_path
                _save_manifest(manifest)
                continue

            if not has_saved_decision:
                _save_classification_decision(
                    section_record,
                    classification,
                    document_id,
                    section_index,
                    section_text,
                )
                _save_manifest(manifest)

            chunks = split_classified_section(classification)

            for role in classification.allowed_roles:
                role_chunks: list[Document] = []
                role_ids: list[str] = []

                for local_chunk_index, chunk in enumerate(chunks):
                    chunk_id = _chunk_id(
                        document_id,
                        section_index,
                        local_chunk_index,
                    )
                    start_index = chunk.metadata.get("start_index")
                    if (
                        not isinstance(start_index, int)
                        or isinstance(start_index, bool)
                        or start_index < 0
                    ):
                        raise RuntimeError(
                            "A text chunk is missing its integer start_index."
                        )
                    absolute_start = int(section_data["start_offset"]) + start_index
                    chunk_page = _page_number_at_offset(
                        absolute_start,
                        page_boundaries,
                    )

                    metadata = {
                        "document_id": document_id,
                        "source_id": source_id,
                        "chunk_id": chunk_id,
                        "source": source_name,
                        "page_number": chunk_page,
                        "section_title": section_title,
                        "role": role,
                        "allowed_roles": classification.allowed_roles,
                        "classification_method": (
                            classification.classification_method
                        ),
                        "classification_confidence": (
                            classification.classification_confidence
                        ),
                    }
                    role_chunks.append(
                        Document(
                            page_content=chunk.page_content,
                            metadata=normalize_chroma_metadata(metadata),
                        )
                    )
                    role_ids.append(chunk_id)

                collection = collections[role]
                documents_to_add: list[Document] = []
                ids_to_add: list[str] = []

                for chunk, chunk_id in zip(role_chunks, role_ids):
                    if _collection_already_has_id(collection, chunk_id):
                        continue
                    documents_to_add.append(chunk)
                    ids_to_add.append(chunk_id)

                if documents_to_add:
                    try:
                        collection.add_documents(
                            documents=documents_to_add,
                            ids=ids_to_add,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to store section {section_index} "
                            f"in role collection '{role}'."
                        ) from exc

                section_chunk_ids = section_record["chunk_ids_by_role"]
                section_chunk_ids[role] = role_ids
                entry_role_ids = entry["chunk_ids_by_role"][role]
                for role_chunk_id in role_ids:
                    if role_chunk_id not in entry_role_ids:
                        entry_role_ids.append(role_chunk_id)
                _reconcile_manifest_entry(collections, entry)
                _save_manifest(manifest)

            section_record["status"] = "completed"
            _save_manifest(manifest)
    except Exception:
        entry["status"] = "partial"
        entry["retryable"] = True
        if (
            current_section_record is not None
            and current_section_record.get("status")
            not in {"completed", "quarantined"}
        ):
            current_section_record["status"] = "in_progress"
        entry["sections_processed"] = sum(
            1
            for section_index in range(len(sections))
            if isinstance(
                entry["sections"].get(str(section_index)),
                dict,
            )
            and entry["sections"][str(section_index)].get("status")
            in {"completed", "quarantined"}
        )
        _save_manifest(manifest)
        raise

    _reconcile_manifest_entry(collections, entry)

    entry["sections_processed"] = sum(
        1
        for record in entry["sections"].values()
        if isinstance(record, dict)
        and record.get("status") in {"completed", "quarantined"}
    )

    statuses = [
        entry["sections"].get(str(section_index), {}).get("status")
        for section_index in range(len(sections))
        if isinstance(
            entry["sections"].get(str(section_index)),
            dict,
        )
    ]
    if statuses and all(status == "completed" for status in statuses):
        entry["status"] = "completed"
        entry["retryable"] = False
    elif (
        statuses
        and all(status in {"completed", "quarantined"} for status in statuses)
        and any(status == "quarantined" for status in statuses)
    ):
        entry["status"] = "needs_review"
        entry["retryable"] = False
    else:
        entry["status"] = "partial"
        entry["retryable"] = True

    _save_manifest(manifest)

    return _ingestion_summary(
        document_id,
        entry,
        False,
        boundary_method,
    )


def _collection_is_empty(collection: Any) -> bool:
    """Return True when a Chroma collection contains no stored chunks."""

    result = collection.get(limit=1, include=[])
    ids = result.get("ids") if isinstance(result, Mapping) else None
    return not ids


def ingest_uploads_if_empty(
    collections: Mapping[str, Any],
    uploads_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Ingest supported upload documents when all vector collections are empty."""

    if not collections or not all(
        _collection_is_empty(collection) for collection in collections.values()
    ):
        return []

    target_dir = UPLOADS_DIR if uploads_dir is None else uploads_dir
    if not target_dir.is_dir():
        return []

    summaries: list[dict[str, Any]] = []
    for candidate in sorted(target_dir.iterdir(), key=lambda path: path.name):
        if (
            candidate.is_file()
            and not candidate.name.startswith(".")
            and candidate.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        ):
            summaries.append(ingest_document(candidate))
    return summaries


def sync_approved_documents_if_empty(
    collections: Mapping[str, Any],
    uploads_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Index Supabase-approved documents when all ChromaDB collections are empty."""

    if not collections or not all(
        _collection_is_empty(collection) for collection in collections.values()
    ):
        return []

    from src.api.services import local_rag_store

    target_dir = UPLOADS_DIR if uploads_dir is None else uploads_dir
    if not target_dir.is_dir():
        return []

    docs = local_rag_store.list_documents(limit=500)
    summaries: list[dict[str, Any]] = []
    for doc in docs:
        if doc.get("approval_status") != "approved":
            continue
        filename = Path(str(doc.get("filename") or "")).name
        if not filename:
            continue
        candidate = (target_dir / filename).resolve()
        try:
            candidate.relative_to(target_dir.resolve())
        except ValueError:
            continue
        if not candidate.is_file():
            continue
        if doc.get("indexing_status") == "indexed":
            summaries.append(
                rehydrate_indexed_document(
                    candidate,
                    str(doc["id"]),
                    doc=doc,
                    collections=collections,
                    logical_filename=filename,
                )
            )
        else:
            summaries.append(
                index_approved_document(
                    candidate,
                    str(doc["id"]),
                    collections=collections,
                    logical_filename=filename,
                )
            )
    return summaries


def classify_document_sections(
    document_path: Path,
    logical_filename: str | None = None,
) -> dict[str, Any]:
    """Load and classify one document without writing to ChromaDB or manifest."""

    if not isinstance(document_path, Path):
        raise TypeError("document_path must be a pathlib.Path")

    resolved_path = document_path.resolve(strict=True)
    source_name = (
        Path(logical_filename).name
        if logical_filename and logical_filename.strip()
        else resolved_path.name
    )
    loaded_documents = load_document(resolved_path)

    file_hash = hashlib.sha256()
    with resolved_path.open("rb") as document_file:
        for file_chunk in iter(lambda: document_file.read(1024 * 1024), b""):
            file_hash.update(file_chunk)
    file_hash_sha256 = file_hash.hexdigest().lower()

    source_id = hashlib.sha256(
        f"upload:{source_name.casefold()}".encode("utf-8")
    ).hexdigest()

    sections, boundary_method, _ = _identify_sections(
        loaded_documents,
        source_name,
    )

    extracted_text_length = sum(
        len(doc.page_content)
        for doc in loaded_documents
        if isinstance(doc.page_content, str)
    )

    classified_sections: list[dict[str, Any]] = []
    quarantined_count = 0

    for section_index, section_data in enumerate(sections):
        section_title = section_data["title"]
        section_text = section_data["text"]
        section_page = int(section_data["page_number"])
        section_sha256 = hashlib.sha256(
            section_text.encode("utf-8")
        ).hexdigest().lower()

        try:
            raw_classification = classify_section(section_title, section_text)
        except ValueError:
            raw_classification = None

        validated = _validate_classification(raw_classification)
        if validated is None:
            quarantined_count += 1
            classified_sections.append({
                "section_index": section_index,
                "section_title": section_title,
                "page_number": section_page,
                "section_sha256": section_sha256,
                "proposed_roles": [],
                "classification_method": (
                    raw_classification.classification_method
                    if raw_classification is not None
                    else "groq_llm"
                ),
                "classification_confidence": (
                    raw_classification.classification_confidence
                    if raw_classification is not None
                    else None
                ),
                "classifier_quarantined": True,
            })
        else:
            classified_sections.append({
                "section_index": section_index,
                "section_title": section_title,
                "page_number": section_page,
                "section_sha256": section_sha256,
                "proposed_roles": list(validated.allowed_roles),
                "classification_method": validated.classification_method,
                "classification_confidence": validated.classification_confidence,
                "classifier_quarantined": False,
            })

    return {
        "filename": source_name,
        "file_hash_sha256": file_hash_sha256,
        "source_id": source_id,
        "extracted_text_length": extracted_text_length,
        "section_boundary_method": boundary_method,
        "sections": classified_sections,
        "quarantined_section_count": quarantined_count,
    }


def remove_document_chunks_from_chroma(
    document_id: str,
    collections: Mapping[str, Any],
) -> int:
    """Remove all ChromaDB chunks for ``document_id`` across all role collections."""

    removed = 0
    for role in ALLOWED_ROLES:
        collection = collections.get(role)
        if collection is None:
            continue
        result = collection.get(where={"document_id": document_id}, include=[])
        ids = result.get("ids") if isinstance(result, Mapping) else None
        if isinstance(ids, list) and ids:
            collection.delete(ids=ids)
            removed += len(ids)
    return removed


def _write_approved_sections_to_chroma(
    resolved_path: Path,
    document_id: str,
    source_id: str,
    source_name: str,
    target_collections: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str], int]:
    """Chunk, embed, write, and verify approved sections in ChromaDB."""

    from src.api.services import local_rag_store

    loaded_documents = load_document(resolved_path)
    parsed_sections, _, page_boundaries = _identify_sections(
        loaded_documents,
        source_name,
    )
    db_sections = local_rag_store.get_document_sections(document_id)
    approved_sections = [
        sec for sec in db_sections if sec.get("review_status") == "approved"
    ]
    if not approved_sections:
        raise RuntimeError("No approved sections exist to index.")

    section_chunks_payload: list[dict[str, Any]] = []
    verified_collections: set[str] = set()
    total_verified_chunks = 0

    for db_sec in approved_sections:
        sec_index = int(db_sec["section_index"])
        if sec_index < 0 or sec_index >= len(parsed_sections):
            raise RuntimeError(
                f"Approved section index {sec_index} is missing from document."
            )
        parsed_sec = parsed_sections[sec_index]
        parsed_hash = hashlib.sha256(
            parsed_sec["text"].encode("utf-8")
        ).hexdigest().lower()
        if parsed_hash != db_sec.get("section_sha256"):
            raise RuntimeError(
                f"Section {sec_index} content hash mismatch against approved record."
            )

        approved_roles = list(db_sec.get("approved_roles") or [])
        if not approved_roles:
            raise RuntimeError(
                f"Approved section {sec_index} has no approved_roles."
            )

        classification = ClassificationResult(
            section_title=str(db_sec["section_title"]),
            section_text=parsed_sec["text"],
            allowed_roles=approved_roles,
            classification_method=str(
                db_sec.get("classification_method") or "groq_llm"
            ),
            classification_confidence=float(
                db_sec.get("classification_confidence")
                if db_sec.get("classification_confidence") is not None
                else 1.0
            ),
            review_required=False,
        )
        split_chunks = split_classified_section(classification)
        section_uuid = str(db_sec["id"])
        chunk_ids_by_role: dict[str, list[str]] = {}

        for role in approved_roles:
            collection = target_collections[role]
            role_docs: list[Document] = []
            role_ids: list[str] = []
            for chunk_idx, chunk in enumerate(split_chunks):
                chunk_id = _chunk_id(document_id, sec_index, chunk_idx)
                start_offset = parsed_sec["start_offset"] + int(
                    chunk.metadata.get("start_index", 0)
                )
                page_number = _page_number_at_offset(
                    start_offset,
                    page_boundaries,
                )
                metadata = {
                    "document_id": document_id,
                    "section_id": section_uuid,
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "page_number": page_number,
                    "section_title": str(db_sec["section_title"]),
                    "role": role,
                    "allowed_roles": approved_roles,
                    "classification_method": classification.classification_method,
                    "classification_confidence": classification.classification_confidence,
                }
                role_docs.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata=normalize_chroma_metadata(metadata),
                    )
                )
                role_ids.append(chunk_id)

            missing_ids = [
                cid
                for cid in role_ids
                if not _collection_already_has_id(collection, cid)
            ]
            if missing_ids:
                docs_to_add = [
                    doc_obj
                    for doc_obj, cid in zip(role_docs, role_ids)
                    if cid in set(missing_ids)
                ]
                collection.add_documents(documents=docs_to_add, ids=missing_ids)

            verify_result = collection.get(
                ids=role_ids,
                include=["metadatas"],
            )
            stored_ids = (
                verify_result.get("ids")
                if isinstance(verify_result, Mapping)
                else None
            )
            stored_metas = (
                verify_result.get("metadatas")
                if isinstance(verify_result, Mapping)
                else None
            )
            if (
                not isinstance(stored_ids, list)
                or not isinstance(stored_metas, list)
                or set(stored_ids) != set(role_ids)
                or len(stored_metas) != len(stored_ids)
            ):
                raise RuntimeError(
                    f"ChromaDB verification failed for role '{role}', section {sec_index}."
                )
            for meta in stored_metas:
                if (
                    not isinstance(meta, Mapping)
                    or meta.get("document_id") != document_id
                    or meta.get("section_id") != section_uuid
                    or meta.get("role") != role
                ):
                    raise RuntimeError(
                        f"ChromaDB metadata ownership mismatch for role '{role}'."
                    )

            chunk_ids_by_role[role] = role_ids
            verified_collections.add(role)
            total_verified_chunks += len(role_ids)

        section_chunks_payload.append({
            "section_id": section_uuid,
            "chunk_ids_by_role": chunk_ids_by_role,
        })

    return section_chunks_payload, verified_collections, total_verified_chunks


def rehydrate_indexed_document(
    document_path: Path,
    document_id: str,
    *,
    doc: Mapping[str, Any] | None = None,
    collections: Mapping[str, Any] | None = None,
    logical_filename: str | None = None,
) -> dict[str, Any]:
    """Re-populate ephemeral ChromaDB for an already approved & indexed document."""

    from src.api.services import local_rag_store

    if not isinstance(document_path, Path):
        raise TypeError("document_path must be a pathlib.Path")

    resolved_path = document_path.resolve(strict=True)
    record = doc if doc is not None else local_rag_store.get_document_by_id(document_id)
    if record is None:
        raise RuntimeError(f"Document {document_id} not found in Supabase.")
    if (
        record.get("approval_status") != "approved"
        or record.get("indexing_status") != "indexed"
    ):
        raise RuntimeError("Only approved and indexed documents can be rehydrated.")

    file_hash = hashlib.sha256()
    with resolved_path.open("rb") as document_file:
        for file_chunk in iter(lambda: document_file.read(1024 * 1024), b""):
            file_hash.update(file_chunk)
    actual_hash = file_hash.hexdigest().lower()
    if actual_hash != record.get("file_hash_sha256"):
        raise RuntimeError(
            "Document file hash does not match the indexed Supabase record."
        )

    source_id = str(record["source_id"])
    source_name = (
        Path(logical_filename).name
        if logical_filename and logical_filename.strip()
        else str(record.get("filename") or resolved_path.name)
    )
    target_collections = (
        dict(collections) if collections is not None else initialize_collections()
    )
    _, verified_collections, total_verified_chunks = _write_approved_sections_to_chroma(
        resolved_path,
        document_id,
        source_id,
        source_name,
        target_collections,
    )
    return {
        "document_id": document_id,
        "approval_status": "approved",
        "indexing_status": "indexed",
        "verified_chunk_count": total_verified_chunks,
        "verified_collections": sorted(verified_collections),
    }


def index_approved_document(
    document_path: Path,
    document_id: str,
    *,
    collections: Mapping[str, Any] | None = None,
    actor_id: str | None = None,
    logical_filename: str | None = None,
) -> dict[str, Any]:
    """Index only Supabase-approved sections into ChromaDB and attest verification."""

    from src.api.services import local_rag_store

    if not isinstance(document_path, Path):
        raise TypeError("document_path must be a pathlib.Path")

    resolved_path = document_path.resolve(strict=True)
    doc = local_rag_store.get_document_by_id(document_id)
    if doc is None:
        raise RuntimeError(f"Document {document_id} not found in Supabase.")
    if doc.get("approval_status") != "approved":
        raise RuntimeError(
            f"Document {document_id} is not approved for indexing "
            f"(status: {doc.get('approval_status')})."
        )

    file_hash = hashlib.sha256()
    with resolved_path.open("rb") as document_file:
        for file_chunk in iter(lambda: document_file.read(1024 * 1024), b""):
            file_hash.update(file_chunk)
    actual_hash = file_hash.hexdigest().lower()
    if actual_hash != doc.get("file_hash_sha256"):
        raise RuntimeError(
            "Document file hash does not match the approved Supabase record."
        )

    approval_version = int(doc["approval_version"])
    source_id = str(doc["source_id"])
    source_name = (
        Path(logical_filename).name
        if logical_filename and logical_filename.strip()
        else str(doc.get("filename") or resolved_path.name)
    )

    local_rag_store.start_indexing(
        document_id=document_id,
        expected_approval_version=approval_version,
        expected_file_hash=actual_hash,
        actor_id=actor_id,
    )

    target_collections = (
        dict(collections) if collections is not None else initialize_collections()
    )

    try:
        (
            section_chunks_payload,
            verified_collections,
            total_verified_chunks,
        ) = _write_approved_sections_to_chroma(
            resolved_path,
            document_id,
            source_id,
            source_name,
            target_collections,
        )

        updated_doc = local_rag_store.complete_indexing(
            document_id=document_id,
            expected_approval_version=approval_version,
            expected_file_hash=actual_hash,
            section_chunks=section_chunks_payload,
            details={
                "verified_chunk_count": total_verified_chunks,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "verification_source": "chromadb_backend_attestation",
                "verified_collections": sorted(verified_collections),
            },
            actor_id=actor_id,
        )
        return {
            "document_id": document_id,
            "approval_status": updated_doc.get("approval_status"),
            "indexing_status": updated_doc.get("indexing_status"),
            "approval_version": updated_doc.get("approval_version"),
            "verified_chunk_count": total_verified_chunks,
            "verified_collections": sorted(verified_collections),
        }
    except Exception as exc:
        try:
            local_rag_store.fail_indexing(
                document_id=document_id,
                expected_approval_version=approval_version,
                expected_file_hash=actual_hash,
                error_details={"error_type": type(exc).__name__},
                actor_id=actor_id,
            )
        except Exception:
            pass
        raise
