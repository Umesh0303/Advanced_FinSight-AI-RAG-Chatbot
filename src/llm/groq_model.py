"""Centralized configuration for the Groq chat model."""

from langchain_groq import ChatGroq

from src.config.settings import GROQ_API_KEY, GROQ_MODEL


def get_groq_model() -> ChatGroq:
    """Create and return a configured Groq chat model.

    The function validates Groq configuration before creating the model.
    It does not send a request to Groq.

    Returns:
        A configured ChatGroq model instance.

    Raises:
        ValueError: If the Groq API key or model name is missing.
    """

    missing_settings: list[str] = []

    if not GROQ_API_KEY:
        missing_settings.append("GROQ_API_KEY")
    if not GROQ_MODEL:
        missing_settings.append("GROQ_MODEL")

    if missing_settings:
        missing_names = " and ".join(missing_settings)
        raise ValueError(
            f"Missing required Groq configuration: {missing_names}. "
            "Set the value(s) in the project root .env file."
        )

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.0,
    )