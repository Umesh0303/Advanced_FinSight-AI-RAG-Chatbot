"""Validated chat request and response models."""

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """A user question and optional client correlation identifier."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    message: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class SourceResponse(BaseModel):
    """Source metadata returned by the existing RAG chain."""

    model_config = ConfigDict(from_attributes=True)

    source: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None


class ChatResponse(BaseModel):
    """Successful answer and provenance from one stateless RAG request."""

    success: bool
    answer: str
    session_id: str
    sources: list[SourceResponse]
