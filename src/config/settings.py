"""Configuration settings for the FinSight AI application."""

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv


# The project root is three levels above this file's containing directory:
# src/config/settings.py -> config -> src -> project root.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# Load local environment values without overriding values already set by the system.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# API configuration is read when the module loads, but it is only validated
# when validate_api_configuration() is called.
GROQ_API_KEY: Final[str | None] = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY: Final[str | None] = os.getenv("GOOGLE_API_KEY")
GROQ_MODEL: Final[str | None] = os.getenv("GROQ_MODEL")
INGESTION_MANIFEST_HMAC_KEY: Final[str | None] = os.getenv(
    "INGESTION_MANIFEST_HMAC_KEY"
)
SUPABASE_URL: Final[str | None] = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY: Final[str | None] = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)
SUPABASE_JWT_AUDIENCE: Final[str] = os.getenv(
    "SUPABASE_JWT_AUDIENCE",
    "authenticated",
)
API_CORS_ORIGINS: Final[str] = os.getenv("API_CORS_ORIGINS", "")

# Project data and storage paths.
UPLOADS_DIR: Final[Path] = PROJECT_ROOT / "data" / "uploads"
UNCLASSIFIED_DIR: Final[Path] = PROJECT_ROOT / "data" / "unclassified"
_chroma_db_dir = os.getenv("CHROMA_DB_DIR")
CHROMA_DB_DIR: Final[Path] = (
    Path(_chroma_db_dir)
    if _chroma_db_dir and _chroma_db_dir.strip()
    else PROJECT_ROOT / "vectorstore" / "chroma_db"
)

# Roles and their matching logical ChromaDB collections.
ALLOWED_ROLES: Final[tuple[str, ...]] = (
    "admin",
    "business_analyst",
    "risk_analyst",
    "customer",
)

COLLECTION_NAMES: Final[tuple[str, ...]] = (
    "admin",
    "business_analyst",
    "risk_analyst",
    "customer",
)

ROLE_COLLECTIONS: Final[dict[str, str]] = {
    "admin": "admin",
    "business_analyst": "business_analyst",
    "risk_analyst": "risk_analyst",
    "customer": "customer",
}

# Document formats accepted by the application.
SUPPORTED_DOCUMENT_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".pdf", ".docx", ".txt"}
)


def validate_api_configuration() -> None:
    """Raise a clear error when a required API setting is missing."""

    required_settings: dict[str, str | None] = {
        "GROQ_API_KEY": GROQ_API_KEY,
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "GROQ_MODEL": GROQ_MODEL,
    }
    missing_settings = [
        setting_name
        for setting_name, setting_value in required_settings.items()
        if not setting_value
    ]

    if missing_settings:
        missing_names = ", ".join(missing_settings)
        raise ValueError(
            f"Missing required environment settings: {missing_names}. "
            "Configure them in the project root .env file."
        )