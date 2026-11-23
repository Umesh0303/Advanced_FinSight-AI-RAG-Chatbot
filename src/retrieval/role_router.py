"""Route roles to only their configured Chroma collections."""

from collections.abc import Mapping
from typing import Any

from src.security.access_control import (
    get_allowed_collections,
    validate_role,
)


def get_role_collections(
    role: object,
    collections: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a new mapping containing only collections allowed for ``role``."""

    validated_role = validate_role(role)
    allowed_names = get_allowed_collections(validated_role)

    if not isinstance(collections, Mapping):
        raise TypeError("collections must be a mapping of names to collections.")
    if any(not isinstance(name, str) for name in collections):
        raise TypeError("collection mapping keys must be strings.")
    for collection_name in allowed_names:
        if (
            collection_name not in collections
            or collections[collection_name] is None
        ):
            raise RuntimeError(
                f"Authorized collection {collection_name!r} is unavailable."
            )
    if any(
        value is None or isinstance(value, (str, bytes, int, float, bool))
        for value in collections.values()
    ):
        raise TypeError("collection mapping values must be collection objects.")

    routed_collections: dict[str, Any] = {}
    for collection_name in allowed_names:
        routed_collections[collection_name] = collections[collection_name]

    return routed_collections