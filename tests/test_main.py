"""Tests for the local FinSight AI command-line interface."""

from collections.abc import Callable

import pytest

import main
from src.chains.rag_chain import RAGResponse, SourceReference
from src.config.settings import ALLOWED_ROLES


def make_input(*values: str) -> Callable[[str], str]:
    remaining = iter(values)

    def read(prompt: str) -> str:
        assert isinstance(prompt, str)
        return next(remaining)

    return read


def configure_initialization(monkeypatch, collections=None):
    embedding = object()
    calls = []

    monkeypatch.setattr(main.settings, "validate_api_configuration", lambda: None)
    monkeypatch.setattr(main, "get_embedding_model", lambda: embedding)

    def initialize_collections(*, embedding_model):
        calls.append(embedding_model)
        return collections if collections is not None else {"customer": object()}

    monkeypatch.setattr(main, "initialize_collections", initialize_collections)
    return embedding, calls


def test_successful_application_initialization_uses_shared_embedding(
    monkeypatch,
):
    collections = {"admin": object(), "customer": object()}
    embedding, calls = configure_initialization(monkeypatch, collections)

    result = main.initialize_resources()

    assert result is collections
    assert calls == [embedding]


def test_missing_api_configuration_is_reported_without_secrets(monkeypatch):
    monkeypatch.setattr(
        main.settings,
        "validate_api_configuration",
        lambda: (_ for _ in ()).throw(ValueError("missing configuration")),
    )

    with pytest.raises(RuntimeError, match="Required API configuration"):
        main.initialize_resources()


def test_embedding_initialization_failure_is_clear_and_chained(monkeypatch):
    configure_initialization(monkeypatch)
    failure = RuntimeError("provider detail containing credentials")
    monkeypatch.setattr(
        main,
        "get_embedding_model",
        lambda: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini embedding initialization failed",
    ) as exc:
        main.initialize_resources()

    assert exc.value.__cause__ is failure


def test_chromadb_initialization_failure_is_clear_and_chained(monkeypatch):
    configure_initialization(monkeypatch)
    failure = OSError("database detail")

    def fail_initialization(**kwargs):
        assert kwargs["embedding_model"] is not None
        raise failure

    monkeypatch.setattr(main, "initialize_collections", fail_initialization)

    with pytest.raises(
        RuntimeError,
        match="ChromaDB collection initialization failed",
    ) as exc:
        main.initialize_resources()

    assert exc.value.__cause__ is failure


@pytest.mark.parametrize(
    ("selection", "expected_role"),
    tuple(zip(("1", "2", "3", "4"), ALLOWED_ROLES, strict=True)),
)
def test_every_role_selection_is_validated(selection, expected_role):
    output: list[str] = []

    role = main.select_role(
        input_fn=make_input(selection),
        output_fn=output.append,
    )

    assert role == expected_role
    assert any("local testing only" in line for line in output)


def test_invalid_role_selection_prompts_again():
    output: list[str] = []

    role = main.select_role(
        input_fn=make_input("not-a-role", "4"),
        output_fn=output.append,
    )

    assert role == "customer"
    assert any("Invalid selection" in line for line in output)


def test_successful_question_and_answer_flow_displays_sources():
    response = RAGResponse(
        answer="Revenue increased based on the retrieved report.",
        sources=[
            SourceReference(
                source="annual_report.pdf",
                page_number=12,
                section_title="Revenue",
            )
        ],
        role="customer",
        retrieved_count=1,
    )
    calls = []
    output: list[str] = []

    def answer_question(question, role, collections):
        calls.append((question, role, collections))
        return response

    collections = {"customer": object()}
    should_return = main.run_chat_session(
        "customer",
        collections,
        input_fn=make_input("What happened to revenue?", "exit"),
        output_fn=output.append,
        question_handler=answer_question,
    )

    assert should_return is False
    assert calls == [("What happened to revenue?", "customer", collections)]
    assert any(response.answer in line for line in output)
    assert any("annual_report.pdf" in line for line in output)
    assert any("Page 12" in line for line in output)
    assert any("Section: Revenue" in line for line in output)


def test_answer_without_optional_source_metadata_is_displayed_gracefully():
    response = RAGResponse(
        answer="An answer without citation metadata.",
        sources=[SourceReference()],
        role="customer",
        retrieved_count=1,
    )
    output: list[str] = []

    main.display_response(response, output_fn=output.append)

    assert any(response.answer in line for line in output)
    assert "  No source metadata available." in output


def test_invalid_rag_response_is_rejected_safely():
    output: list[str] = []

    def return_invalid_response(question, role, collections):
        assert question
        assert role == "customer"
        assert collections == {}
        return {"answer": "not a RAGResponse", "secret": "do-not-print"}

    main.run_chat_session(
        "customer",
        {},
        input_fn=make_input("What is the balance?", "exit"),
        output_fn=output.append,
        question_handler=return_invalid_response,
    )

    rendered = "\n".join(output)
    assert (
        "Unable to display the answer because the RAG system returned "
        "an unexpected response."
    ) in rendered
    assert "do-not-print" not in rendered


def test_chat_can_return_to_role_menu_and_exit(monkeypatch):
    configure_initialization(monkeypatch)
    output: list[str] = []

    main.run_cli(
        input_fn=make_input("4", "menu", "5"),
        output_fn=output.append,
    )

    assert sum(line.strip().startswith("Select a role") for line in output) == 2
    assert any("Selected role: customer" in line for line in output)
    assert output[-1] == "Goodbye."


def test_exit_menu_option_stops_cli(monkeypatch):
    configure_initialization(monkeypatch)
    output: list[str] = []

    main.run_cli(input_fn=make_input("5"), output_fn=output.append)

    assert output[-1] == "Goodbye."


def test_empty_question_is_rejected_without_calling_rag():
    output: list[str] = []
    calls = []

    def answer_question(*args):
        calls.append(args)
        raise AssertionError("RAG must not be called for an empty question")

    main.run_chat_session(
        "customer",
        {},
        input_fn=make_input("   ", "exit"),
        output_fn=output.append,
        question_handler=answer_question,
    )

    assert calls == []
    assert any("Question cannot be empty" in line for line in output)


@pytest.mark.parametrize(
    ("raised", "expected_text"),
    [
        (KeyboardInterrupt(), "Interrupted. Goodbye."),
        (EOFError(), "Input closed. Goodbye."),
    ],
)
def test_cli_handles_keyboard_interrupt_and_eof(
    monkeypatch,
    raised,
    expected_text,
):
    configure_initialization(monkeypatch)
    output: list[str] = []

    def interrupted_input(_prompt):
        assert isinstance(_prompt, str)
        raise raised

    main.run_cli(input_fn=interrupted_input, output_fn=output.append)

    assert any(expected_text in line for line in output)


def test_rag_error_message_does_not_expose_secrets():
    output: list[str] = []

    def fail_with_secret(question, role, collections):
        assert question
        assert role == "customer"
        assert collections == {}
        raise RuntimeError("GROQ_API_KEY=do-not-print")

    main.run_chat_session(
        "customer",
        {},
        input_fn=make_input("What is the balance?", "exit"),
        output_fn=output.append,
        question_handler=fail_with_secret,
    )

    rendered = "\n".join(output)
    assert "Unable to answer this question" in rendered
    assert "do-not-print" not in rendered


@pytest.mark.parametrize(
    "startup_error",
    [
        RuntimeError("GROQ_API_KEY=runtime-secret"),
        ValueError("GOOGLE_API_KEY=unexpected-secret"),
    ],
)
def test_startup_errors_do_not_expose_exception_details(
    monkeypatch,
    startup_error,
):
    output: list[str] = []

    def fail_startup():
        raise startup_error

    monkeypatch.setattr(main, "initialize_resources", fail_startup)

    main.run_cli(input_fn=make_input(), output_fn=output.append)

    rendered = "\n".join(output)
    assert "Startup failed" in rendered
    assert "runtime-secret" not in rendered
    assert "unexpected-secret" not in rendered
    assert "GROQ_API_KEY=" not in rendered
    assert "GOOGLE_API_KEY=" not in rendered
