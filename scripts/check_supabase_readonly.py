"""Read-only Supabase PostgREST connectivity diagnostic for FinSight AI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.identity_service import _service_key_kind
from src.config import settings


REQUEST_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class DiagnosticResult:
    """Safe, secret-free Supabase read-only connectivity diagnostic."""

    reachable: bool
    auth_status: str
    profiles_table: str
    http_status: int | None
    error_category: str


def _validate_supabase_url(raw_url: object) -> str | None:
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None
    candidate = raw_url.strip().rstrip("/")
    parsed = urlsplit(candidate)
    is_local_http = (
        parsed.scheme == "http"
        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    )
    try:
        parsed.port
    except ValueError:
        return None
    if (
        (parsed.scheme != "https" and not is_local_http)
        or not parsed.netloc
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    return candidate


def check_supabase_readonly(
    *,
    supabase_url: str | None = None,
    service_role_key: str | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> DiagnosticResult:
    """Execute a zero-row read-only query against public.profiles."""

    raw_url = settings.SUPABASE_URL if supabase_url is None else supabase_url
    raw_key = (
        settings.SUPABASE_SERVICE_ROLE_KEY
        if service_role_key is None
        else service_role_key
    )

    if (
        not isinstance(raw_url, str)
        or not raw_url.strip()
        or not isinstance(raw_key, str)
        or not raw_key.strip()
    ):
        return DiagnosticResult(
            reachable=False,
            auth_status="unknown",
            profiles_table="inaccessible",
            http_status=None,
            error_category="missing_configuration",
        )

    base_url = _validate_supabase_url(raw_url)
    if base_url is None:
        return DiagnosticResult(
            reachable=False,
            auth_status="unknown",
            profiles_table="inaccessible",
            http_status=None,
            error_category="invalid_url",
        )

    key_kind = _service_key_kind(raw_key.strip())
    if key_kind is None:
        return DiagnosticResult(
            reachable=False,
            auth_status="rejected",
            profiles_table="inaccessible",
            http_status=None,
            error_category="invalid_service_key",
        )

    endpoint = f"{base_url}/rest/v1/profiles"
    headers = {"apikey": raw_key.strip()}
    if key_kind == "legacy":
        headers["Authorization"] = f"Bearer {raw_key.strip()}"

    try:
        response = httpx.get(
            endpoint,
            params={"select": "id", "limit": "0"},
            headers=headers,
            timeout=timeout,
            follow_redirects=False,
        )
    except httpx.TimeoutException:
        return DiagnosticResult(
            reachable=False,
            auth_status="unknown",
            profiles_table="inaccessible",
            http_status=None,
            error_category="network_timeout",
        )
    except httpx.RequestError:
        return DiagnosticResult(
            reachable=False,
            auth_status="unknown",
            profiles_table="inaccessible",
            http_status=None,
            error_category="network_error",
        )

    status_code = response.status_code
    if status_code == 200:
        return DiagnosticResult(
            reachable=True,
            auth_status="accepted",
            profiles_table="accessible",
            http_status=status_code,
            error_category="none",
        )
    if status_code == 401:
        return DiagnosticResult(
            reachable=True,
            auth_status="rejected",
            profiles_table="inaccessible",
            http_status=status_code,
            error_category="authentication_failure",
        )
    if status_code == 403:
        return DiagnosticResult(
            reachable=True,
            auth_status="accepted",
            profiles_table="inaccessible",
            http_status=status_code,
            error_category="permission_failure",
        )
    if status_code == 404:
        return DiagnosticResult(
            reachable=True,
            auth_status="accepted",
            profiles_table="inaccessible",
            http_status=status_code,
            error_category="missing_table",
        )
    if 500 <= status_code <= 599:
        return DiagnosticResult(
            reachable=True,
            auth_status="unknown",
            profiles_table="inaccessible",
            http_status=status_code,
            error_category="supabase_server_error",
        )

    return DiagnosticResult(
        reachable=True,
        auth_status="unknown",
        profiles_table="inaccessible",
        http_status=status_code,
        error_category="unexpected_http_status",
    )


def format_diagnostic(result: DiagnosticResult) -> str:
    """Format diagnostic output without exposing any credentials or row data."""

    lines = [
        f"Supabase connection reachable: {result.reachable}",
        f"PostgREST authentication: {result.auth_status}",
        f"Profiles table: {result.profiles_table}",
        f"HTTP status code: {result.http_status}",
        f"Error category: {result.error_category}",
    ]
    return "\n".join(lines)


def main() -> int:
    result = check_supabase_readonly()
    print(format_diagnostic(result))
    return 0 if result.reachable and result.http_status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
