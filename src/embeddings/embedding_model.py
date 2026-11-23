"""Centralized configuration for Google Gemini embeddings."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config.settings import GOOGLE_API_KEY


EMBEDDING_MODEL_NAME: str = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS: int = 768


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Create and return the configured Gemini embedding model.

    The model is initialized only when this function is called. It does not
    download a model or make an embedding API request.

    Returns:
        A configured GoogleGenerativeAIEmbeddings instance.

    Raises:
        ValueError: If GOOGLE_API_KEY is missing.
    """

    if not GOOGLE_API_KEY:
        raise ValueError(
            "Missing required Google API configuration: GOOGLE_API_KEY. "
            "Set GOOGLE_API_KEY in the project root .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY,
        model=EMBEDDING_MODEL_NAME,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )