"""Tests for server-side Supabase profiles identity resolution."""

from types import SimpleNamespace

import httpx
import jwt
import pytest

from src.api.services import identity_service
from src.config import settings


_PROJECT_URL = "https://project.example.supabase.co"
_AUTH_USER_ID = "13f6e5bc-22a9-4e8b-8c5a-d5d363c3ec47"
_SECRET_KEY = "test-only-signing-key-with-32-bytes"


@pytest.fixture
def configured_supabase(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", _PROJECT_URL)
    monkeypatch.setattr(
        settings,
        "SUPABASE_SERVICE_ROLE_KEY",
        "sb_secret_test",
    )


def make_profile(
    *,
    profile_id: str = _AUTH_USER_ID,
    role: str = "Support Agent",
    account_status: object = "active",
    customer_id: str | None = "customer-42",
):
    return {
        "id": profile_id,
        "role": role,
        "account_status": account_status,
        "customer_id": customer_id,
    }


def test_lookup_uses_verified_uuid_service_credential_and_required_fields(
    configured_supabase,
    monkeypatch,
):
    calls = []

    def fake_get(url, *, params, headers, timeout, follow_redirects):
        calls.append((url, params, headers, timeout, follow_redirects))
        return SimpleNamespace(
            status_code=200,
            json=lambda: [make_profile()],
        )

    monkeypatch.setattr(identity_service.httpx, "get", fake_get)

    role = identity_service.resolve_rag_role(_AUTH_USER_ID)

    assert role == "risk_analyst"
    assert len(calls) == 1
    url, params, headers, timeout, follow_redirects = calls[0]
    assert url == f"{_PROJECT_URL}/rest/v1/profiles"
    assert params == {
        "select": "id,role,account_status,customer_id",
        "id": f"eq.{_AUTH_USER_ID}",
    }
    assert "auth_user_id" not in params
    assert headers == {"apikey": "sb_secret_test"}
    assert timeout == 5.0
    assert follow_redirects is False


@pytest.mark.parametrize(
    ("application_role", "expected_rag_role"),
    [
        ("Admin", "admin"),
        ("Analyst", "business_analyst"),
        ("Business Analyst", "business_analyst"),
        ("Customer", "customer"),
        ("Support Agent", "risk_analyst"),
        ("Risk Analyst", "risk_analyst"),
        ("admin", "admin"),
        ("business_analyst", "business_analyst"),
        ("risk_analyst", "risk_analyst"),
        ("customer", "customer"),
    ],
)
def test_profile_role_is_mapped(
    monkeypatch,
    configured_supabase,
    application_role,
    expected_rag_role,
):
    profile = make_profile(role=application_role)

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return [profile]

    monkeypatch.setattr(
        identity_service.httpx,
        "get",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    assert identity_service.resolve_rag_role(_AUTH_USER_ID) == expected_rag_role


@pytest.mark.parametrize(
    ("account_status", "include_status", "allowed"),
    [
        ("active", True, True),
        ("suspended", True, False),
        ("pending", True, False),
        ("inactive", True, False),
        (None, True, False),
        ("unknown", True, False),
        ("Active", True, False),
        (None, False, False),
    ],
)
def test_only_exact_active_profile_status_is_authorized(
    configured_supabase,
    monkeypatch,
    account_status,
    include_status,
    allowed,
):
    profile = make_profile()
    if include_status:
        profile["account_status"] = account_status
    else:
        del profile["account_status"]

    monkeypatch.setattr(
        identity_service.httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: [profile],
        ),
    )

    if allowed:
        assert identity_service.resolve_rag_role(_AUTH_USER_ID) == "risk_analyst"
    else:
        with pytest.raises(identity_service.IdentityMappingDenied):
            identity_service.resolve_rag_role(_AUTH_USER_ID)


@pytest.mark.parametrize(
    "profile",
    [
        None,
        make_profile(role=None),
        make_profile(role="Customer Support"),
        make_profile(role="business analyst"),
    ],
)
def test_missing_profile_or_unmapped_role_is_denied(
    configured_supabase,
    monkeypatch,
    profile,
):
    profiles = [] if profile is None else [profile]
    monkeypatch.setattr(
        identity_service.httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: profiles,
        ),
    )

    with pytest.raises(identity_service.IdentityMappingDenied):
        identity_service.resolve_rag_role(_AUTH_USER_ID)


def test_malformed_profile_id_fails_closed(
    configured_supabase,
    monkeypatch,
):
    monkeypatch.setattr(
        identity_service.httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: [make_profile(profile_id="not-a-uuid")],
        ),
    )

    with pytest.raises(identity_service.IdentityMappingUnavailable):
        identity_service.resolve_rag_role(_AUTH_USER_ID)


def test_duplicate_profiles_fail_closed(
    configured_supabase,
    monkeypatch,
):
    monkeypatch.setattr(
        identity_service.httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            status_code=200,
            json=lambda: [make_profile(), make_profile()],
        ),
    )

    with pytest.raises(identity_service.IdentityMappingUnavailable):
        identity_service.resolve_rag_role(_AUTH_USER_ID)


@pytest.mark.parametrize("status_code", [400, 404])
def test_missing_profiles_schema_is_reported(
    configured_supabase,
    monkeypatch,
    status_code,
):
    monkeypatch.setattr(
        identity_service.httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=status_code),
    )

    with pytest.raises(identity_service.IdentitySchemaUnavailable):
        identity_service.resolve_rag_role(_AUTH_USER_ID)


def test_publishable_key_is_rejected_for_profile_lookup(
    configured_supabase,
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "SUPABASE_SERVICE_ROLE_KEY",
        "sb_publishable_test",
    )
    monkeypatch.setattr(
        identity_service.httpx,
        "get",
        lambda *_args, **_kwargs: pytest.fail("No network call is allowed"),
    )

    with pytest.raises(identity_service.IdentityLookupNotConfigured):
        identity_service.resolve_rag_role(_AUTH_USER_ID)


def test_legacy_service_role_jwt_is_sent_as_backend_bearer(
    configured_supabase,
    monkeypatch,
):
    service_key = jwt.encode(
        {"role": "service_role"},
        _SECRET_KEY,
        algorithm="HS256",
    )
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", service_key)
    calls = []

    def fake_get(_url, *, headers, **_kwargs):
        calls.append(headers)
        return SimpleNamespace(
            status_code=200,
            json=lambda: [make_profile(role="Admin")],
        )

    monkeypatch.setattr(identity_service.httpx, "get", fake_get)

    assert identity_service.resolve_rag_role(_AUTH_USER_ID) == "admin"
    assert calls == [
        {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
        }
    ]


def test_missing_service_role_key_fails_closed(
    configured_supabase,
    monkeypatch,
):
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", None)
    monkeypatch.setattr(
        identity_service.httpx,
        "get",
        lambda *_args, **_kwargs: pytest.fail("No network call is allowed"),
    )

    with pytest.raises(identity_service.IdentityLookupNotConfigured):
        identity_service.resolve_rag_role(_AUTH_USER_ID)


def test_supabase_unauthorized_lookup_fails_closed(
    configured_supabase,
    monkeypatch,
):
    monkeypatch.setattr(
        identity_service.httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=401),
    )

    with pytest.raises(identity_service.IdentityMappingUnavailable):
        identity_service.resolve_rag_role(_AUTH_USER_ID)


def test_network_failure_is_safe_and_fails_closed(
    configured_supabase,
    monkeypatch,
):
    def fail_request(*_args, **_kwargs):
        raise httpx.ConnectError("contains internal provider details")

    monkeypatch.setattr(identity_service.httpx, "get", fail_request)

    with pytest.raises(identity_service.IdentityMappingUnavailable):
        identity_service.resolve_rag_role(_AUTH_USER_ID)
