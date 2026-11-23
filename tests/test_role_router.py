"""Tests for role-based routing to Chroma collections."""

import pytest

from src.config.settings import ALLOWED_ROLES, COLLECTION_NAMES
from src.retrieval.role_router import get_role_collections
from src.security.access_control import validate_role


class MockCollection:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def collections():
    return {
        name: MockCollection(name)
        for name in (*COLLECTION_NAMES, "unauthorized_extra")
    }


@pytest.mark.parametrize("role", ALLOWED_ROLES)
def test_admin_and_roles_receive_only_their_expected_collection_objects(
    collections,
    role,
):
    routed = get_role_collections(role, collections)
    expected_names = (
        COLLECTION_NAMES
        if role == "admin"
        else (role,)
    )

    assert set(routed) == set(expected_names)
    for name in expected_names:
        assert routed[name] is collections[name]
    assert "unauthorized_extra" not in routed


def test_admin_cannot_receive_a_partial_collection_set(collections):
    del collections["risk_analyst"]

    with pytest.raises(RuntimeError, match="risk_analyst.*unavailable"):
        get_role_collections("admin", collections)


@pytest.mark.parametrize(
    ("role", "missing_collection"),
    [
        ("business_analyst", "business_analyst"),
        ("risk_analyst", "risk_analyst"),
        ("customer", "customer"),
    ],
)
def test_missing_authorized_collection_raises(
    collections,
    role,
    missing_collection,
):
    del collections[missing_collection]

    with pytest.raises(RuntimeError, match="unavailable"):
        get_role_collections(role, collections)


@pytest.mark.parametrize(
    ("role", "exception_type"),
    [
        ("unsupported", ValueError),
        ("", ValueError),
        (None, TypeError),
        (" admin ", ValueError),
    ],
)
def test_invalid_role_raises_the_access_control_validation_exception(
    collections,
    role,
    exception_type,
):
    with pytest.raises(exception_type):
        get_role_collections(role, collections)
    with pytest.raises(exception_type):
        validate_role(role)


@pytest.mark.parametrize(
    "invalid_collections",
    [
        None,
        [],
        {1: MockCollection("invalid_key")},
        {
            "customer": MockCollection("customer"),
            "unauthorized_extra": None,
        },
        {"customer": "not a collection"},
    ],
)
def test_malformed_collection_mapping_is_rejected(invalid_collections):
    with pytest.raises(TypeError):
        get_role_collections("customer", invalid_collections)


def test_none_authorized_collection_is_unavailable(collections):
    collections["customer"] = None

    with pytest.raises(RuntimeError, match="customer.*unavailable"):
        get_role_collections("customer", collections)


def test_router_returns_a_new_mapping_without_modifying_input(collections):
    original_keys = tuple(collections)
    original_objects = dict(collections)

    routed = get_role_collections("business_analyst", collections)

    assert routed is not collections
    assert tuple(collections) == original_keys
    assert collections == original_objects
    assert set(routed) == {"business_analyst"}
    assert routed["business_analyst"] is collections["business_analyst"]
