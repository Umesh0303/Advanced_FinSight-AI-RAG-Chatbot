"""Unit tests for the read-only Supabase connectivity diagnostic."""

from types import SimpleNamespace

import httpx

import scripts.check_supabase_readonly as diag


_VALID_URL = "https://example.supabase.co"
_SECRET_KEY = "sb_secret_super_sensitive_value_do_not_leak"


def test_successful_readonly_request(monkeypatch):
    def fake_get(url, *, params, headers, timeout, follow_redirects):
        assert url == f"{_VALID_URL}/rest/v1/profiles"
        assert params == {"select": "id", "limit": "0"}
        assert headers == {"apikey": _SECRET_KEY}
        assert timeout == diag.REQUEST_TIMEOUT_SECONDS
        assert follow_redirects is False
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(diag.httpx, "get", fake_get)

    result = diag.check_supabase_readonly(
        supabase_url=_VALID_URL,
        service_role_key=_SECRET_KEY,
    )
    output = diag.format_diagnostic(result)

    assert result.reachable is True
    assert result.auth_status == "accepted"
    assert result.profiles_table == "accessible"
    assert result.http_status == 200
    assert result.error_category == "none"
    assert _SECRET_KEY not in output


def test_missing_configuration():
    result = diag.check_supabase_readonly(supabase_url="", service_role_key="")
    assert result.reachable is False
    assert result.http_status is None
    assert result.error_category == "missing_configuration"


def test_invalid_url_rejected():
    result = diag.check_supabase_readonly(
        supabase_url="http://remote-insecure.example.com",
        service_role_key=_SECRET_KEY,
    )
    output = diag.format_diagnostic(result)
    assert result.reachable is False
    assert result.error_category == "invalid_url"
    assert _SECRET_KEY not in output


def test_http_401_rejected(monkeypatch):
    monkeypatch.setattr(
        diag.httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=401),
    )
    result = diag.check_supabase_readonly(
        supabase_url=_VALID_URL,
        service_role_key=_SECRET_KEY,
    )
    output = diag.format_diagnostic(result)
    assert result.reachable is True
    assert result.auth_status == "rejected"
    assert result.profiles_table == "inaccessible"
    assert result.http_status == 401
    assert result.error_category == "authentication_failure"
    assert _SECRET_KEY not in output


def test_http_403_permission_failure(monkeypatch):
    monkeypatch.setattr(
        diag.httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=403),
    )
    result = diag.check_supabase_readonly(
        supabase_url=_VALID_URL,
        service_role_key=_SECRET_KEY,
    )
    assert result.http_status == 403
    assert result.profiles_table == "inaccessible"
    assert result.error_category == "permission_failure"


def test_http_404_missing_table(monkeypatch):
    monkeypatch.setattr(
        diag.httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=404),
    )
    result = diag.check_supabase_readonly(
        supabase_url=_VALID_URL,
        service_role_key=_SECRET_KEY,
    )
    assert result.http_status == 404
    assert result.profiles_table == "inaccessible"
    assert result.error_category == "missing_table"


def test_network_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(diag.httpx, "get", raise_timeout)
    result = diag.check_supabase_readonly(
        supabase_url=_VALID_URL,
        service_role_key=_SECRET_KEY,
    )
    output = diag.format_diagnostic(result)
    assert result.reachable is False
    assert result.http_status is None
    assert result.error_category == "network_timeout"
    assert _SECRET_KEY not in output


def test_supabase_server_error(monkeypatch):
    monkeypatch.setattr(
        diag.httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=503),
    )
    result = diag.check_supabase_readonly(
        supabase_url=_VALID_URL,
        service_role_key=_SECRET_KEY,
    )
    assert result.reachable is True
    assert result.http_status == 503
    assert result.profiles_table == "inaccessible"
    assert result.error_category == "supabase_server_error"
