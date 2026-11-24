"""Tests for trusted main-application to RAG role mapping."""

import pytest

from src.config.settings import COLLECTION_NAMES
from src.security.access_control import get_allowed_collections
from src.security.role_mapping import ROLE_MAPPING, map_application_role


@pytest.mark.parametrize(
    ("application_role", "rag_role"),
    [
        ("Admin", "admin"),
        ("Analyst", "business_analyst"),
        ("Business Analyst", "business_analyst"),
        ("Customer", "customer"),
        ("Support Agent", "risk_analyst"),
        ("Risk Analyst", "risk_analyst"),
    ],
)
def test_application_roles_map_to_exact_rag_roles(
    application_role,
    rag_role,
):
    assert map_application_role(application_role) == rag_role


@pytest.mark.parametrize(
    "application_role",
    [
        None,
        "",
        " ",
        "ADMIN",
        "Admin ",
        "analyst",
        "business analyst",
        "Support agent",
        "Risk Support",
        "Other",
    ],
)
def test_missing_unknown_or_mis_cased_roles_are_denied(application_role):
    with pytest.raises(ValueError):
        map_application_role(application_role)


@pytest.mark.parametrize(
    ("application_role", "expected_collections"),
    [
        ("Admin", COLLECTION_NAMES),
        ("Analyst", ("business_analyst",)),
        ("Business Analyst", ("business_analyst",)),
        ("Customer", ("customer",)),
        ("Support Agent", ("risk_analyst",)),
        ("Risk Analyst", ("risk_analyst",)),
    ],
)
def test_mapped_roles_preserve_existing_collection_isolation(
    application_role,
    expected_collections,
):
    rag_role = map_application_role(application_role)

    assert get_allowed_collections(rag_role) == expected_collections


def test_role_mapping_is_explicit_and_immutable():
    assert dict(ROLE_MAPPING) == {
        "Admin": "admin",
        "Analyst": "business_analyst",
        "Business Analyst": "business_analyst",
        "Customer": "customer",
        "Support Agent": "risk_analyst",
        "Risk Analyst": "risk_analyst",
        "admin": "admin",
        "business_analyst": "business_analyst",
        "risk_analyst": "risk_analyst",
        "customer": "customer",
    }
    original_role = ROLE_MAPPING["Analyst"]
    try:
        with pytest.raises(TypeError):
            ROLE_MAPPING["Analyst"] = "risk_analyst"
    finally:
        if ROLE_MAPPING.get("Analyst") != original_role:
            ROLE_MAPPING["Analyst"] = original_role
