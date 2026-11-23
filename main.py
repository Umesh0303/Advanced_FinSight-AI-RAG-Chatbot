"""Local command-line interface for the FinSight AI RAG chatbot."""

from collections.abc import Callable, Mapping
from typing import Any

from src.chains.rag_chain import RAGResponse, ask_question
from src.config import settings
from src.embeddings.embedding_model import get_embedding_model
from src.security.access_control import validate_role
from src.vectorstore.chroma_store import initialize_collections


_ROLE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", "admin", "Admin"),
    ("2", "business_analyst", "Business Analyst"),
    ("3", "risk_analyst", "Risk Analyst"),
    ("4", "customer", "Customer"),
)


def initialize_resources() -> dict[str, Any]:
    """Validate configuration and initialize embeddings and persistent collections.

    The collection factory uses the project's configured ChromaDB directory.
    """

    try:
        settings.validate_api_configuration()
    except Exception as exc:
        raise RuntimeError(
            "Required API configuration is missing or invalid. Check "
            "GROQ_API_KEY, GOOGLE_API_KEY, and GROQ_MODEL in the project .env."
        ) from exc

    try:
        embedding_model = get_embedding_model()
    except Exception as exc:
        raise RuntimeError("Gemini embedding initialization failed.") from exc

    try:
        return initialize_collections(embedding_model=embedding_model)
    except Exception as exc:
        raise RuntimeError("ChromaDB collection initialization failed.") from exc


def select_role(
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> str | None:
    """Prompt until the user selects a validated role or exits the menu."""

    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn

    while True:
        write_output("\nSelect a role (local testing only; this does not authenticate you):")
        for number, role, label in _ROLE_OPTIONS:
            write_output(f"{number}. {label} ({role})")
        write_output("5. Exit")

        selection = read_input("Enter a number: ").strip()
        if selection == "5":
            return None

        selected_role = next(
            (option[1] for option in _ROLE_OPTIONS if option[0] == selection),
            None,
        )
        if selected_role is None:
            write_output("Invalid selection. Choose a number from 1 to 5.")
            continue

        try:
            return validate_role(selected_role)
        except (TypeError, ValueError):
            write_output("Invalid role selection. Please choose another option.")


def display_response(
    response: RAGResponse,
    output_fn: Callable[[str], None] | None = None,
) -> None:
    """Print an answer and only the source metadata actually provided."""

    write_output = print if output_fn is None else output_fn
    write_output(f"\nAnswer:\n{response.answer}")

    source_lines: list[str] = []
    for source in response.sources:
        details: list[str] = []
        if source.source:
            details.append(source.source)
        if source.page_number is not None:
            details.append(f"Page {source.page_number}")
        if source.section_title:
            details.append(f"Section: {source.section_title}")
        if details:
            source_lines.append("  - " + " | ".join(details))

    write_output("Sources:")
    if source_lines:
        for line in source_lines:
            write_output(line)
    else:
        write_output("  No source metadata available.")


def run_chat_session(
    role: str,
    collections: Mapping[str, Any],
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
    question_handler: Callable[..., RAGResponse] | None = None,
) -> bool:
    """Run one role's chat; return True to return to the role menu."""

    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn
    ask = ask_question if question_handler is None else question_handler
    validated_role = validate_role(role)

    write_output(
        f"\nSelected role: {validated_role}. "
        "Role selection is for local testing and is not authentication."
    )
    write_output("Enter 'menu' to return to role selection or 'exit' to quit.")

    while True:
        question = read_input("Question: ").strip()
        command = question.casefold()
        if command in {"exit", "quit"}:
            return False
        if command in {"menu", "back"}:
            return True
        if not question:
            write_output("Question cannot be empty. Please enter a question.")
            continue

        try:
            response = ask(question, validated_role, collections)
        except Exception:
            write_output(
                "Unable to answer this question because a retrieval or language "
                "model operation failed. Check the service configuration and try again."
            )
            continue

        if not isinstance(response, RAGResponse):
            write_output(
                "Unable to display the answer because the RAG system returned "
                "an unexpected response."
            )
            continue

        display_response(response, output_fn=write_output)


def run_cli(
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> None:
    """Initialize shared resources and run the local role-based chat menu."""

    read_input = input if input_fn is None else input_fn
    write_output = print if output_fn is None else output_fn

    try:
        collections = initialize_resources()
    except RuntimeError:
        write_output(
            "Startup failed. Check the required API configuration and local "
            "resource setup, then try again."
        )
        return
    except Exception:
        write_output(
            "Startup failed due to an unexpected error. Exception details "
            "were withheld."
        )
        return

    write_output(
        "FinSight AI local RAG chatbot. Role selection is for testing only "
        "and does not authenticate users."
    )
    try:
        while True:
            role = select_role(input_fn=read_input, output_fn=write_output)
            if role is None:
                write_output("Goodbye.")
                return

            should_return_to_menu = run_chat_session(
                role,
                collections,
                input_fn=read_input,
                output_fn=write_output,
            )
            if not should_return_to_menu:
                write_output("Goodbye.")
                return
    except KeyboardInterrupt:
        write_output("\nInterrupted. Goodbye.")
    except EOFError:
        write_output("\nInput closed. Goodbye.")


if __name__ == "__main__":
    run_cli()