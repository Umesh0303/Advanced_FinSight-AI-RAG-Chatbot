"""Resolve verified Supabase identities through trusted public.profiles rows."""

from collections.abc import Mapping
from uuid import UUID

import httpx
import jwt

from src.config import settings
from src.security.role_mapping import map_application_role


class IdentityMappingDenied(PermissionError):
    """The authenticated user has no active, recognized application mapping."""


class IdentityMappingUnavailable(RuntimeError):
    """The trusted Supabase identity lookup could not be completed safely."""


class IdentityLookupNotConfigured(IdentityMappingUnavailable):
    """Required server-side Supabase lookup configuration is missing."""


class IdentitySchemaUnavailable(IdentityMappingUnavailable):
    """The required public.profiles fields are not available."""


def _service_key_kind(key: str) -> str | None:
    """Recognize modern secret keys and legacy service_role JWTs only."""

    if key.startswith("sb_secret_"):
        return "secret" if len(key) > len("sb_secret_") else None
    if key.startswith("sb_publishable_"):
        return None

    try:
        claims = jwt.decode(
            key,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
            },
            algorithms=["HS256"],
        )
    except jwt.InvalidTokenError:
        return None
    return "legacy" if claims.get("role") == "service_role" else None


def resolve_rag_role(auth_user_id: str) -> str:
    """Resolve a verified Auth UUID to an active profile and its RAG role.

    The UUID must come from the already-verified JWT subject. A backend-only
    service credential is used solely for this exact profile lookup.
    """

    try:
        normalized_user_id = str(UUID(auth_user_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise IdentityMappingDenied from exc

    project_url = settings.SUPABASE_URL
    service_role_key = settings.SUPABASE_SERVICE_ROLE_KEY
    if (
        not isinstance(project_url, str)
        or not project_url.strip()
        or not isinstance(service_role_key, str)
        or not service_role_key.strip()
    ):
        raise IdentityLookupNotConfigured(
            "Supabase identity lookup is not configured."
        )
    key_kind = _service_key_kind(service_role_key)
    if key_kind is None:
        raise IdentityLookupNotConfigured(
            "A Supabase service-role credential is required for profile lookup."
        )

    endpoint = f"{project_url.rstrip('/')}/rest/v1/profiles"
    headers = {"apikey": service_role_key}
    if key_kind == "legacy":
        headers["Authorization"] = f"Bearer {service_role_key}"
    try:
        response = httpx.get(
            endpoint,
            params={
                "select": "id,role,account_status,customer_id",
                "id": f"eq.{normalized_user_id}",
            },
            headers=headers,
            timeout=5.0,
            follow_redirects=False,
        )
    except httpx.RequestError as exc:
        raise IdentityMappingUnavailable from exc

    if response.status_code in {400, 404}:
        raise IdentitySchemaUnavailable(
            "The required public.profiles identity fields are unavailable."
        )
    if response.status_code != 200:
        raise IdentityMappingUnavailable from None

    try:
        records = response.json()
    except ValueError as exc:
        raise IdentityMappingUnavailable from exc

    if not isinstance(records, list):
        raise IdentityMappingUnavailable
    if not records:
        raise IdentityMappingDenied
    if len(records) != 1 or not isinstance(records[0], Mapping):
        raise IdentityMappingUnavailable

    record = records[0]
    try:
        record_user_id = str(UUID(record.get("id")))
    except (TypeError, ValueError, AttributeError) as exc:
        raise IdentityMappingUnavailable from exc
    if record_user_id != normalized_user_id:
        raise IdentityMappingDenied

    if record.get("account_status") != "active":
        raise IdentityMappingDenied

    try:
        return map_application_role(record.get("role"))
    except ValueError as exc:
        raise IdentityMappingDenied from exc
