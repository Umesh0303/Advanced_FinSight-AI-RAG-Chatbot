"""Schemas for Admin document review, approval, rejection, revocation, and indexing."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SectionDecisionRequest(BaseModel):
    """Explicit Admin review decision for one document section."""

    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    review_status: Literal["approved", "excluded"]
    approved_roles: list[str] = Field(default_factory=list)
    quarantine_resolution_reason: str | None = None
    exclusion_reason: str | None = None


class ApproveDocumentRequest(BaseModel):
    """Payload for approving a pending document and all of its sections."""

    model_config = ConfigDict(extra="forbid")

    section_decisions: list[SectionDecisionRequest] = Field(min_length=1)
    decision_reason: str | None = None


class RejectOrRevokeRequest(BaseModel):
    """Payload for rejecting a pending document or revoking an approved document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision_reason: str = Field(min_length=1, max_length=1000)
