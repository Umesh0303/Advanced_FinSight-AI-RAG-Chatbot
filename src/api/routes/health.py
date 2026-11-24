"""Unauthenticated process health endpoint."""

from fastapi import APIRouter


router = APIRouter()


@router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Return a minimal liveness response without exposing configuration."""

    return {"status": "ok"}
