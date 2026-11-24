"""Verify Supabase access tokens and resolve trusted application roles."""

from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import (
    InvalidTokenError,
    PyJWKClientConnectionError,
    PyJWKClientError,
)

from src.api.services.identity_service import (
    IdentityMappingDenied,
    IdentityLookupNotConfigured,
    IdentityMappingUnavailable,
    IdentitySchemaUnavailable,
    resolve_rag_role,
)
from src.config import settings


_ALLOWED_ALGORITHMS = ("ES256", "RS256")
_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Identity and application role derived only from a verified token."""

    user_id: str
    role: str


def _supabase_issuer() -> str:
    project_url = settings.SUPABASE_URL
    if not isinstance(project_url, str) or not project_url.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )

    project_url = project_url.rstrip("/")
    parsed_url = urlsplit(project_url)
    is_local_http = (
        parsed_url.scheme == "http"
        and parsed_url.hostname in {"localhost", "127.0.0.1", "::1"}
    )
    try:
        parsed_url.port
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        ) from exc
    if (
        parsed_url.scheme != "https" and not is_local_http
    ) or not parsed_url.netloc or parsed_url.path not in {"", "/"} or (
        parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )

    return f"{project_url}/auth/v1"


@lru_cache(maxsize=4)
def _get_jwks_client(jwks_url: str) -> PyJWKClient:
    """Reuse a JWKS client with a bounded key-set cache."""

    return PyJWKClient(
        jwks_url,
        timeout=5,
        cache_jwk_set=True,
        lifespan=300,
    )


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_authenticated_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedPrincipal:
    """Verify a Supabase JWT and resolve its trusted profiles role."""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    issuer = _supabase_issuer()
    token = credentials.credentials

    try:
        header = jwt.get_unverified_header(token)
    except InvalidTokenError as exc:
        raise _unauthorized() from exc

    if (
        header.get("alg") not in _ALLOWED_ALGORITHMS
        or not isinstance(header.get("kid"), str)
        or not header["kid"]
    ):
        raise _unauthorized()

    jwks_url = f"{issuer}/.well-known/jwks.json"
    try:
        signing_key = _get_jwks_client(jwks_url).get_signing_key_from_jwt(token)
    except PyJWKClientConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication provider is unavailable.",
        ) from exc
    except PyJWKClientError as exc:
        raise _unauthorized() from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication provider is unavailable.",
        ) from exc

    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=list(_ALLOWED_ALGORITHMS),
            audience=settings.SUPABASE_JWT_AUDIENCE,
            issuer=issuer,
            options={
                "require": ["exp", "iss", "aud", "sub"],
            },
        )
    except InvalidTokenError as exc:
        raise _unauthorized() from exc

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise _unauthorized()

    try:
        user_id = str(UUID(subject))
    except (TypeError, ValueError, AttributeError) as exc:
        raise _unauthorized() from exc

    try:
        role = resolve_rag_role(user_id)
    except IdentityMappingDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The authenticated user has no active authorized account.",
        ) from exc
    except IdentityLookupNotConfigured as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The trusted Supabase identity lookup is not configured.",
        ) from exc
    except IdentitySchemaUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The trusted Supabase profiles lookup is unavailable.",
        ) from exc
    except IdentityMappingUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The trusted identity provider is temporarily unavailable.",
        ) from exc

    return AuthenticatedPrincipal(user_id=user_id, role=role)
