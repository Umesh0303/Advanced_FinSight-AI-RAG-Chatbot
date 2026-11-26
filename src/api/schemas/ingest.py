"""Response schemas for Admin document upload and registration."""

from pydantic import BaseModel


class IngestResponse(BaseModel):
    """Result of uploading, extracting, classifying, and registering a document."""

    success: bool
    document_id: str
    filename: str
    file_hash_sha256: str
    approval_status: str
    indexing_status: str
    approval_version: int
    extracted_text_length: int
    section_count: int
    quarantined_section_count: int
    duplicate: bool
