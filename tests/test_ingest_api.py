"""Isolated tests for POST /api/v1/ingest and local_rag_store."""

import io
import zipfile
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

import src.api.routes.ingest as ingest_route
import src.api.services.local_rag_store as store
from src.api.main import create_app
from src.config import settings
from tests.test_api import bearer, configure_auth, make_token


@pytest.fixture
def ingest_client(tmp_path, monkeypatch):
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    monkeypatch.setattr(ingest_route, "UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr("src.ingestion.document_loader.UPLOADS_DIR", uploads_dir)

    app = create_app(
        resources_factory=lambda: {"admin": object()},
        cors_origins=["https://finsight.example"],
    )
    with TestClient(app) as client:
        yield client, uploads_dir


def _sample_classified(filename: str = "policy.txt") -> dict:
    return {
        "filename": filename,
        "file_hash_sha256": "a" * 64,
        "source_id": "b" * 64,
        "extracted_text_length": 120,
        "section_boundary_method": "single_section_fallback",
        "sections": [
            {
                "section_index": 0,
                "section_title": "1. Policy",
                "page_number": 1,
                "section_sha256": "c" * 64,
                "proposed_roles": ["customer"],
                "classification_method": "groq_llm",
                "classification_confidence": 0.95,
                "classifier_quarantined": False,
            }
        ],
        "quarantined_section_count": 0,
    }


def test_ingest_requires_authentication(ingest_client):
    client, _ = ingest_client
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("policy.txt", b"Customer policy text.", "text/plain")},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "app_role",
    ["Customer", "Business Analyst", "Risk Analyst"],
)
def test_ingest_rejects_non_admin_roles(ingest_client, monkeypatch, app_role):
    client, _ = ingest_client
    private_key = configure_auth(monkeypatch, application_role=app_role)

    response = client.post(
        "/api/v1/ingest",
        headers=bearer(make_token(private_key)),
        files={"file": ("policy.txt", b"Customer policy text.", "text/plain")},
    )
    assert response.status_code == 403


def test_ingest_admin_valid_txt_succeeds_and_cleans_temp_file(ingest_client, monkeypatch):
    client, uploads_dir = ingest_client
    private_key = configure_auth(monkeypatch, application_role="Admin")
    doc_uuid = str(uuid4())

    monkeypatch.setattr(store, "get_document_by_hash", lambda h: None)
    monkeypatch.setattr(
        ingest_route,
        "classify_document_sections",
        lambda path, logical_filename=None: _sample_classified(logical_filename),
    )
    captured = {}

    def fake_register(**kwargs):
        captured.update(kwargs)
        return {
            "id": doc_uuid,
            "filename": kwargs["filename"],
            "file_hash_sha256": kwargs["file_hash_sha256"],
            "approval_status": "pending",
            "indexing_status": "not_indexed",
            "approval_version": 1,
        }

    monkeypatch.setattr(store, "register_document", fake_register)

    response = client.post(
        "/api/v1/ingest",
        headers=bearer(make_token(private_key)),
        files={"file": ("policy.txt", b"Customer may read this policy.", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["document_id"] == doc_uuid
    assert body["approval_status"] == "pending"
    assert body["indexing_status"] == "not_indexed"
    assert body["duplicate"] is False
    assert body["section_count"] == 1
    assert captured["actor_id"] == "13f6e5bc-22a9-4e8b-8c5a-d5d363c3ec47"
    assert list(uploads_dir.iterdir()) == []


def test_ingest_admin_valid_pdf_and_docx_magic_bytes(ingest_client, monkeypatch):
    client, uploads_dir = ingest_client
    private_key = configure_auth(monkeypatch, application_role="Admin")

    monkeypatch.setattr(store, "get_document_by_hash", lambda h: None)
    monkeypatch.setattr(
        ingest_route,
        "classify_document_sections",
        lambda path, logical_filename=None: _sample_classified(logical_filename),
    )
    monkeypatch.setattr(
        store,
        "register_document",
        lambda **kwargs: {
            "id": str(uuid4()),
            "filename": kwargs["filename"],
            "file_hash_sha256": kwargs["file_hash_sha256"],
            "approval_status": "pending",
            "indexing_status": "not_indexed",
            "approval_version": 1,
        },
    )

    pdf_resp = client.post(
        "/api/v1/ingest",
        headers=bearer(make_token(private_key)),
        files={"file": ("manual.pdf", b"%PDF-1.4 synthetic", "application/pdf")},
    )
    assert pdf_resp.status_code == 201

    docx_buf = io.BytesIO()
    with zipfile.ZipFile(docx_buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
    docx_resp = client.post(
        "/api/v1/ingest",
        headers=bearer(make_token(private_key)),
        files={
            "file": (
                "manual.docx",
                docx_buf.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert docx_resp.status_code == 201
    assert list(uploads_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("empty.txt", b"", 422),
        ("bad.exe", b"MZ...", 422),
        ("legacy.doc", b"doc", 422),
        ("fake.pdf", b"NOT A PDF", 422),
        ("fake.docx", b"NOT A ZIP", 422),
        ("binary.txt", b"hello\x00world", 422),
    ],
)
def test_ingest_rejects_invalid_files(
    ingest_client,
    monkeypatch,
    filename,
    content,
    expected_status,
):
    client, uploads_dir = ingest_client
    private_key = configure_auth(monkeypatch, application_role="Admin")

    response = client.post(
        "/api/v1/ingest",
        headers=bearer(make_token(private_key)),
        files={"file": (filename, content, "application/octet-stream")},
    )
    assert response.status_code == expected_status
    assert list(uploads_dir.iterdir()) == []


def test_ingest_rejects_oversized_file(ingest_client, monkeypatch):
    client, uploads_dir = ingest_client
    private_key = configure_auth(monkeypatch, application_role="Admin")
    monkeypatch.setattr(ingest_route, "MAX_UPLOAD_BYTES", 32)

    response = client.post(
        "/api/v1/ingest",
        headers=bearer(make_token(private_key)),
        files={"file": ("big.txt", b"x" * 64, "text/plain")},
    )
    assert response.status_code == 413
    assert list(uploads_dir.iterdir()) == []


def test_ingest_duplicate_returns_idempotent_200(ingest_client, monkeypatch):
    client, _ = ingest_client
    private_key = configure_auth(monkeypatch, application_role="Admin")
    doc_uuid = str(uuid4())

    import hashlib

    payload = b"Duplicate policy text."
    expected_source_id = hashlib.sha256(b"upload:policy.txt").hexdigest()

    monkeypatch.setattr(
        store,
        "get_document_by_hash",
        lambda h: {
            "id": doc_uuid,
            "filename": "policy.txt",
            "source_id": expected_source_id,
            "approval_status": "pending",
            "indexing_status": "not_indexed",
            "approval_version": 1,
        },
    )
    monkeypatch.setattr(
        store,
        "get_document_sections",
        lambda doc_id: [{"classifier_quarantined": True}, {"classifier_quarantined": False}],
    )

    response = client.post(
        "/api/v1/ingest",
        headers=bearer(make_token(private_key)),
        files={"file": ("policy.txt", payload, "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["duplicate"] is True
    assert body["document_id"] == doc_uuid
    assert body["section_count"] == 2
    assert body["quarantined_section_count"] == 1


def test_ingest_extraction_and_supabase_failures(ingest_client, monkeypatch):
    client, uploads_dir = ingest_client
    private_key = configure_auth(monkeypatch, application_role="Admin")

    monkeypatch.setattr(store, "get_document_by_hash", lambda h: None)
    monkeypatch.setattr(
        ingest_route,
        "classify_document_sections",
        lambda path, logical_filename=None: (_ for _ in ()).throw(ValueError("corrupt")),
    )
    resp_extract = client.post(
        "/api/v1/ingest",
        headers=bearer(make_token(private_key)),
        files={"file": ("policy.txt", b"content", "text/plain")},
    )
    assert resp_extract.status_code == 422

    monkeypatch.setattr(
        ingest_route,
        "classify_document_sections",
        lambda path, logical_filename=None: _sample_classified(logical_filename),
    )
    monkeypatch.setattr(
        store,
        "register_document",
        lambda **kwargs: (_ for _ in ()).throw(
            store.LocalRagStoreUnavailable("supabase down")
        ),
    )
    resp_supabase = client.post(
        "/api/v1/ingest",
        headers=bearer(make_token(private_key)),
        files={"file": ("policy.txt", b"content", "text/plain")},
    )
    assert resp_supabase.status_code == 503
    assert list(uploads_dir.iterdir()) == []


def test_local_rag_store_rpc_and_error_handling(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test_key")

    doc_uuid = str(uuid4())
    actor_uuid = str(uuid4())

    def fake_post(url, *, json, headers, timeout, follow_redirects):
        assert url.endswith("/rest/v1/rpc/local_rag_register_document")
        assert headers["apikey"] == "sb_secret_test_key"
        return SimpleNamespace(status_code=200, json=lambda: {"id": doc_uuid})

    monkeypatch.setattr(store.httpx, "post", fake_post)
    res = store.register_document(
        file_hash_sha256="A" * 64,
        source_id="b" * 64,
        filename="policy.txt",
        sections=[],
        actor_id=actor_uuid,
    )
    assert res == {"id": doc_uuid}

    def raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timeout")

    monkeypatch.setattr(store.httpx, "get", raise_timeout)
    with pytest.raises(store.LocalRagStoreUnavailable):
        store.get_document_by_hash("a" * 64)
