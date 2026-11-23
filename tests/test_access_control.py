"""Tests for role validation and collection permissions."""

import pytest

from src.config.settings import ALLOWED_ROLES, COLLECTION_NAMES
from src.security.access_control import (
    get_allowed_collections,
    has_collection_access,
    validate_role,
)


@pytest.mark.parametrize("role", ALLOWED_ROLES)
def test_validate_role_accepts_each_exact_allowed_role(role):
    assert validate_role(role) == role


def test_admin_receives_all_configured_collections():
    allowed = get_allowed_collections("admin")

    assert isinstance(allowed, tuple)
    assert allowed == COLLECTION_NAMES


@pytest.mark.parametrize(
    ("role", "collection_name"),
    [
        ("business_analyst", "business_analyst"),
        ("risk_analyst", "risk_analyst"),
        ("customer", "customer"),
    ],
)
def test_non_admin_roles_receive_only_their_collection(role, collection_name):
    assert get_allowed_collections(role) == (collection_name,)


@pytest.mark.parametrize("role", ALLOWED_ROLES)
def test_has_collection_access_matches_configured_permissions(role):
    allowed = get_allowed_collections(role)

    for collection_name in COLLECTION_NAMES:
        assert has_collection_access(role, collection_name) is (
            collection_name in allowed
        )


@pytest.mark.parametrize(
    "role",
    [
        "executive",
        "",
        " ",
        " admin",
        "admin ",
        None,
        123,
        ["admin"],
    ],
)
def test_validate_role_rejects_invalid_values(role):
    expected_exception = TypeError if not isinstance(role, str) else ValueError
    with pytest.raises(expected_exception):
        validate_role(role)


def test_unknown_collection_is_denied_for_a_valid_role():
    assert has_collection_access("admin", "unknown_collection") is False
    assert has_collection_access("customer", "unknown_collection") is False


@pytest.mark.parametrize(
    "role",
    ["executive", "", " admin", None, 123, ["admin"]],
)
def test_invalid_role_never_grants_collection_access(role):
    for collection_name in COLLECTION_NAMES:
        assert has_collection_access(role, collection_name) is False
