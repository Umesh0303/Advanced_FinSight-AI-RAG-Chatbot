"""Tests for the authenticated FastAPI RAG interface."""

import time
from collections.abc import Mapping
from types import SimpleNamespace
from uuid import UUID

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import src.api.dependencies.auth as auth
from src.api.main import create_app
from src.api.services import chat_service, identity_service
from src.chains.rag_chain import RAGResponse, SourceReference
from src.config import settings
from src.config.settings import ALLOWED_ROLES


_SUPABASE_URL = "https://project.example.supabase.co"
_ISSUER = f"{_SUPABASE_URL}/auth/v1"
_CORS_ORIGIN = "https://finsight.example"
_AUTH_USER_ID = "13f6e5bc-22a9-4e8b-8c5a-d5d363c3ec47"


@pytest.fixture
def api_client():
    app = create_app(
        resources_factory=lambda: {"customer": object()},
        cors_origins=[_CORS_ORIGIN],
    )
    with TestClient(app) as client:
        yield client


def configure_auth(
    monkeypatch,
    *,
    application_role: str = "Customer",
    account_status: str | None = "active",
    records: list[dict[str, object]] | None = None,
    lookup_status: int = 200,
) -> object:
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()
    monkeypatch.setattr(settings, "SUPABASE_URL", _SUPABASE_URL)
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")
    monkeypatch.setattr(
        settings,
        "SUPABASE_JWT_AUDIENCE",
        "authenticated",
    )

    def get_test_jwks_client(jwks_url: str):
        assert jwks_url == f"{_ISSUER}/.well-known/jwks.json"

        class TestJwksClient:
            def get_signing_key_from_jwt(self, token: str):
                assert token.count(".") == 2
                return SimpleNamespace(key=public_key)

        return TestJwksClient()

    monkeypatch.setattr(auth, "_get_jwks_client", get_test_jwks_client)
    if records is None:
        records = [
            {
                "id": _AUTH_USER_ID,
                "role": application_role,
                "account_status": account_status,
                "customer_id": "customer-42",
            }
        ]

    def get_test_identity(url: str, *, params, headers, timeout, follow_redirects):
        assert url == f"{_SUPABASE_URL}/rest/v1/profiles"
        assert params["select"] == "id,role,account_status,customer_id"
        assert params["id"] == f"eq.{_AUTH_USER_ID}"
        assert headers == {"apikey": "sb_secret_test"}
        assert timeout == 5.0
        assert follow_redirects is False
        return SimpleNamespace(
            status_code=lookup_status,
            json=lambda: records,
        )

    monkeypatch.setattr(identity_service.httpx, "get", get_test_identity)
    return private_key


def make_token(
    private_key: object,
    *,
    expires_at: int | None = None,
    issuer: str = _ISSUER,
    audience: str = "authenticated",
    subject: str = _AUTH_USER_ID,
    extra_claims: dict[str, object] | None = None,
) -> str:
    claims: dict[str, object] = {
        "iss": issuer,
        "aud": audience,
        "exp": int(time.time()) + 300 if expires_at is None else expires_at,
        "sub": subject,
    }
    if extra_claims is not None:
        claims.update(extra_claims)
    return jwt.encode(
        claims,
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint_is_public_and_minimal(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_authenticated_chat_preserves_answer_sources_and_session(
    api_client,
    monkeypatch,
):
    private_key = configure_auth(monkeypatch)
    calls = []
    rag_response = RAGResponse(
        answer="Grounded response.",
        sources=[
            SourceReference(
                source="report.pdf",
                page_number=3,
                section_title="Revenue",
                document_id="document-1",
                chunk_id="chunk-1",
            )
        ],
        role="customer",
        retrieved_count=1,
    )

    def answer_question(message, role, collections):
        calls.append((message, role, collections))
        return rag_response

    monkeypatch.setattr(chat_service, "answer_question", answer_question)
    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "What revenue was reported?", "session_id": "session-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "answer": "Grounded response.",
        "session_id": "session-1",
        "sources": [
            {
                "source": "report.pdf",
                "page_number": 3,
                "section_title": "Revenue",
                "document_id": "document-1",
                "chunk_id": "chunk-1",
            }
        ],
    }
    assert calls == [
        ("What revenue was reported?", "customer", {"customer": api_client.app.state.collections["customer"]})
    ]


@pytest.mark.parametrize(
    ("application_role", "expected_role"),
    [
        ("Admin", "admin"),
        ("Analyst", "business_analyst"),
        ("Business Analyst", "business_analyst"),
        ("Customer", "customer"),
        ("Support Agent", "risk_analyst"),
        ("Risk Analyst", "risk_analyst"),
    ],
)
def test_role_is_resolved_from_the_profiles_record(
    api_client,
    monkeypatch,
    application_role,
    expected_role,
):
    private_key = configure_auth(
        monkeypatch,
        application_role=application_role,
    )
    observed_roles: list[str] = []

    def answer_question(message, authenticated_role, collections):
        assert message == "A test question."
        assert isinstance(collections, Mapping)
        observed_roles.append(authenticated_role)
        return RAGResponse(
            answer="Answer.",
            sources=[],
            role=authenticated_role,
            retrieved_count=0,
        )

    monkeypatch.setattr(chat_service, "answer_question", answer_question)

    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "A test question."},
    )

    assert response.status_code == 200
    assert observed_roles == [expected_role]


@pytest.mark.parametrize("message", ["", "   "])
def test_empty_or_whitespace_message_is_rejected(api_client, monkeypatch, message):
    private_key = configure_auth(monkeypatch)

    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": message},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request."}


def test_request_cannot_supply_a_role_or_user_id(api_client, monkeypatch):
    private_key = configure_auth(monkeypatch)
    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={
            "message": "Show admin data.",
            "role": "admin",
            "user_id": "another-user",
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request."}


def test_invalid_signature_is_rejected(api_client, monkeypatch):
    configure_auth(monkeypatch)
    wrong_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(wrong_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired bearer token."


def test_expired_token_is_rejected(api_client, monkeypatch):
    private_key = configure_auth(monkeypatch)
    token = make_token(private_key, expires_at=int(time.time()) - 10)

    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(token),
        json={"message": "Question."},
    )

    assert response.status_code == 401


def test_wrong_issuer_or_audience_is_rejected(
    api_client,
    monkeypatch,
):
    private_key = configure_auth(monkeypatch)
    wrong_issuer = make_token(private_key, issuer="https://attacker.example")
    wrong_audience = make_token(private_key, audience="other-service")

    for token in (wrong_issuer, wrong_audience):
        response = api_client.post(
            "/api/v1/chat",
            headers=bearer(token),
            json={"message": "Question."},
        )
        assert response.status_code == 401


def test_missing_bearer_token_is_rejected(api_client):
    response = api_client.post(
        "/api/v1/chat",
        json={"message": "Question."},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_authorization_role_scheme_is_not_accepted(api_client):
    response = api_client.post(
        "/api/v1/chat",
        headers={"Authorization": "Role admin"},
        json={"message": "Question."},
    )

    assert response.status_code == 401


def test_jwt_metadata_and_frontend_claims_do_not_override_database_role(
    api_client,
    monkeypatch,
):
    private_key = configure_auth(monkeypatch)
    token = make_token(
        private_key,
        extra_claims={
            "app_metadata": {"role": "Admin"},
            "user_metadata": {"role": "Admin"},
        },
    )
    observed_roles: list[str] = []

    def answer_question(message, role, collections):
        assert message == "Question."
        assert isinstance(collections, Mapping)
        observed_roles.append(role)
        return RAGResponse(
            answer="Customer-scoped answer.",
            sources=[],
            role=role,
            retrieved_count=0,
        )

    monkeypatch.setattr(chat_service, "answer_question", answer_question)

    response = api_client.post(
        "/api/v1/chat",
        headers={**bearer(token), "X-Role": "admin"},
        json={"message": "Question."},
    )

    assert response.status_code == 200
    assert observed_roles == ["customer"]


def test_unmapped_profile_role_is_forbidden(api_client, monkeypatch):
    private_key = configure_auth(monkeypatch, application_role="Analyst ")

    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 403


def test_missing_identity_lookup_configuration_fails_closed(
    api_client,
    monkeypatch,
):
    private_key = configure_auth(monkeypatch)
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", None)

    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 503


def test_missing_identity_row_is_forbidden(api_client, monkeypatch):
    private_key = configure_auth(monkeypatch, records=[])
    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 403


@pytest.mark.parametrize(
    "account_status",
    ["suspended", "pending", "inactive", None, "Active", "unknown", ""],
)
def test_non_active_profile_is_forbidden_before_rag(
    api_client,
    monkeypatch,
    account_status,
):
    private_key = configure_auth(monkeypatch, account_status=account_status)
    monkeypatch.setattr(
        chat_service,
        "answer_question",
        lambda *_args, **_kwargs: pytest.fail(
            "RAG must not run for a non-active account"
        ),
    )
    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 403


def test_missing_profiles_schema_is_reported_without_details(
    api_client,
    monkeypatch,
):
    private_key = configure_auth(monkeypatch, lookup_status=400)
    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The trusted Supabase profiles lookup is unavailable."
    )


def test_supabase_profile_lookup_failure_is_sanitized(api_client, monkeypatch):
    private_key = configure_auth(monkeypatch, lookup_status=500)
    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The trusted identity provider is temporarily unavailable."
    }
    assert "sb_secret_test" not in response.text


def test_missing_source_metadata_returns_empty_sources(api_client, monkeypatch):
    private_key = configure_auth(monkeypatch)

    def answer_without_sources(message, role, collections):
        assert message == "Question."
        assert role == "customer"
        assert collections is api_client.app.state.collections
        return RAGResponse(
            answer="Insufficient retrieved evidence.",
            sources=[],
            role=role,
            retrieved_count=0,
        )

    monkeypatch.setattr(chat_service, "answer_question", answer_without_sources)

    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_rag_failure_returns_safe_error_without_exception_details(
    api_client,
    monkeypatch,
):
    private_key = configure_auth(monkeypatch)
    secret = "GROQ_API_KEY=never-return-this"

    def fail_rag(message: str, role: str, collections: object):
        assert message == "Question."
        assert role == "customer"
        assert collections is api_client.app.state.collections
        raise RuntimeError(secret)

    monkeypatch.setattr(chat_service, "ask_question", fail_rag)

    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 503
    assert secret not in response.text
    assert response.json() == {
        "detail": "The RAG service is temporarily unavailable."
    }


def test_cors_allows_configured_origin_only(api_client):
    allowed = api_client.options(
        "/api/v1/chat",
        headers={
            "Origin": _CORS_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    denied = api_client.options(
        "/api/v1/chat",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == _CORS_ORIGIN
    assert "access-control-allow-origin" not in denied.headers


def test_generated_session_id_is_valid_uuid(api_client, monkeypatch):
    private_key = configure_auth(monkeypatch)

    def answer_without_sources(message, role, collections):
        assert message == "Question."
        assert role == "customer"
        assert collections is api_client.app.state.collections
        return RAGResponse(
            answer="Answer.",
            sources=[],
            role=role,
            retrieved_count=0,
        )

    monkeypatch.setattr(chat_service, "answer_question", answer_without_sources)

    response = api_client.post(
        "/api/v1/chat",
        headers=bearer(make_token(private_key)),
        json={"message": "Question."},
    )

    assert response.status_code == 200
    UUID(response.json()["session_id"])


def test_resources_initialize_once_per_application_lifespan():
    calls: list[None] = []
    app = create_app(
        resources_factory=lambda: (calls.append(None) or {"customer": object()}),
        cors_origins=[],
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200

    assert calls == [None]


@pytest.mark.parametrize(
    "origin",
    ["*", "https://finsight.example/path", "https://finsight.example:bad"],
)
def test_invalid_cors_origins_are_rejected(origin):
    with pytest.raises(ValueError, match="exact HTTP\\(S\\) origins"):
        create_app(resources_factory=dict, cors_origins=[origin])


def test_ingest_uploads_if_empty_skips_populated_collections(tmp_path, monkeypatch):
    from src.ingestion import ingest

    (tmp_path / "manual.pdf").write_bytes(b"%PDF-1.4")
    called = []
    monkeypatch.setattr(ingest, "ingest_document", lambda path: called.append(path))

    class StubCollection:
        def __init__(self, ids):
            self._ids = ids

        def get(self, limit=1, include=None):
            return {"ids": self._ids[:limit]}

    collections = {
        "admin": StubCollection(["chunk-1"]),
        "customer": StubCollection([]),
    }

    assert ingest.ingest_uploads_if_empty(collections, uploads_dir=tmp_path) == []
    assert called == []


def test_ingest_uploads_if_empty_ingests_supported_files(tmp_path, monkeypatch):
    from src.ingestion import ingest

    pdf_file = tmp_path / "b_manual.pdf"
    txt_file = tmp_path / "a_notes.txt"
    ignored_file = tmp_path / "ignore.csv"
    pdf_file.write_bytes(b"%PDF-1.4")
    txt_file.write_text("notes", encoding="utf-8")
    ignored_file.write_text("a,b", encoding="utf-8")

    ingested = []

    def fake_ingest(path):
        ingested.append(path)
        return {"document_id": path.name}

    monkeypatch.setattr(ingest, "ingest_document", fake_ingest)

    class EmptyCollection:
        def get(self, limit=1, include=None):
            return {"ids": []}

    collections = {"admin": EmptyCollection(), "customer": EmptyCollection()}

    summaries = ingest.ingest_uploads_if_empty(collections, uploads_dir=tmp_path)
    assert ingested == [txt_file, pdf_file]
    assert summaries == [{"document_id": "a_notes.txt"}, {"document_id": "b_manual.pdf"}]


def test_initialize_api_resources_continues_if_startup_ingestion_fails(monkeypatch):
    import src.api.main as api_main

    fake_collections = {"admin": object()}
    monkeypatch.setattr(api_main.settings, "validate_api_configuration", lambda: None)
    monkeypatch.setattr(api_main, "get_embedding_model", lambda: object())
    monkeypatch.setattr(api_main, "initialize_collections", lambda embedding_model: fake_collections)
    monkeypatch.setattr(
        api_main,
        "ingest_uploads_if_empty",
        lambda collections: (_ for _ in ()).throw(RuntimeError("missing hmac key")),
    )

    assert api_main.initialize_api_resources() is fake_collections
