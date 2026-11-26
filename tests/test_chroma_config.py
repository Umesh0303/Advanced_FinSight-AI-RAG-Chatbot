"""Tests for configurable persistent ChromaDB storage."""

import importlib.util
from pathlib import Path

import dotenv
import pytest

import src.api.main as api_main
import src.ingestion.ingest as ingest_module
import src.vectorstore.chroma_store as chroma_store


_SETTINGS_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "config" / "settings.py"
)


def load_isolated_settings():
    spec = importlib.util.spec_from_file_location(
        "_isolated_finsight_settings",
        _SETTINGS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    settings_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings_module)
    return settings_module


def test_chroma_path_defaults_to_project_local_directory(monkeypatch):
    monkeypatch.delenv("CHROMA_DB_DIR", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **_kwargs: None)

    settings_module = load_isolated_settings()

    assert settings_module.CHROMA_DB_DIR == (
        settings_module.PROJECT_ROOT / "vectorstore" / "chroma_db"
    )


def test_chroma_path_uses_environment_value(monkeypatch, tmp_path):
    configured_path = tmp_path / "persistent" / "chroma_db"
    monkeypatch.setenv("CHROMA_DB_DIR", str(configured_path))
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **_kwargs: None)

    settings_module = load_isolated_settings()

    assert isinstance(settings_module.CHROMA_DB_DIR, Path)
    assert settings_module.CHROMA_DB_DIR == configured_path


@pytest.mark.parametrize("configured_value", ["", " \t "])
def test_empty_chroma_path_uses_project_local_directory(
    monkeypatch,
    configured_value,
):
    monkeypatch.setenv("CHROMA_DB_DIR", configured_value)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **_kwargs: None)

    settings_module = load_isolated_settings()

    assert settings_module.CHROMA_DB_DIR == (
        settings_module.PROJECT_ROOT / "vectorstore" / "chroma_db"
    )


def test_api_and_ingestion_use_configured_chroma_path(
    monkeypatch,
    tmp_path,
):
    configured_path = tmp_path / "configured_chroma"
    initialized_paths = []

    class RecordingChroma:
        def __init__(
            self,
            *,
            collection_name,
            embedding_function,
            persist_directory,
        ):
            initialized_paths.append(Path(persist_directory))

    monkeypatch.setattr(chroma_store, "CHROMA_DB_DIR", configured_path)
    monkeypatch.setattr(chroma_store, "Chroma", RecordingChroma)
    monkeypatch.setattr(chroma_store, "get_embedding_model", object)
    monkeypatch.setattr(api_main.settings, "validate_api_configuration", lambda: None)
    monkeypatch.setattr(api_main, "get_embedding_model", object)

    api_main.initialize_api_resources()

    document = tmp_path / "synthetic.txt"
    document.write_text("test", encoding="utf-8")
    entry = {
        "status": "completed",
        "sections_processed": 0,
        "chunks_stored_per_role": {},
        "quarantined_sections": [],
        "section_boundary_method": "test",
    }
    monkeypatch.setattr(ingest_module, "_manifest_signing_key", lambda: "test-key")
    monkeypatch.setattr(ingest_module, "_source_id", lambda _path: "source-id")
    monkeypatch.setattr(ingest_module, "_document_id", lambda _path: "document-id")
    monkeypatch.setattr(ingest_module, "load_document", lambda _path: [])
    monkeypatch.setattr(ingest_module, "_read_manifest", lambda: {"documents": {}})
    monkeypatch.setattr(
        ingest_module,
        "_manifest_entry",
        lambda *_args: entry,
    )
    monkeypatch.setattr(
        ingest_module,
        "_remove_superseded_source_documents",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        ingest_module,
        "_reconcile_manifest_entry",
        lambda *_args: None,
    )
    monkeypatch.setattr(ingest_module, "_save_manifest", lambda _manifest: None)

    result = ingest_module.ingest_document(document)

    assert result["duplicate"] is True
    assert initialized_paths == [configured_path] * 4
