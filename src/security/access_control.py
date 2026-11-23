"""Role validation and collection-level permission checks.

These helpers authorize role values only; they do not authenticate users.
"""

from typing import Final

from src.config.settings import ALLOWED_ROLES, COLLECTION_NAMES, ROLE_COLLECTIONS


_ADMIN_ROLE: Final[str] = "admin"


def validate_role(role: object) -> str:
    """Return an exact allowed role or raise for invalid input."""

    if not isinstance(role, str):
        raise TypeError("role must be a string.")
    if not role or not role.strip():
        raise ValueError("role must not be empty or whitespace.")
    if role not in ALLOWED_ROLES:
        allowed = ", ".join(ALLOWED_ROLES)
        raise ValueError(
            f"Unsupported role {role!r}. Allowed roles: {allowed}."
        )
    return role


def get_allowed_collections(role: object) -> tuple[str, ...]:
    """Return the immutable set of collections authorized for a valid role."""

    validated_role = validate_role(role)
    if validated_role == _ADMIN_ROLE:
        return tuple(COLLECTION_NAMES)

    collection_name = ROLE_COLLECTIONS.get(validated_role)
    if collection_name not in COLLECTION_NAMES:
        raise RuntimeError(
            f"No configured collection exists for role {validated_role!r}."
        )
    return (collection_name,)


def has_collection_access(role: object, collection_name: object) -> bool:
    """Check collection permission, denying invalid roles and unknown names."""

    try:
        validated_role = validate_role(role)
    except (TypeError, ValueError):
        return False
    return collection_name in get_allowed_collections(validated_role)