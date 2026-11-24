"""Map trusted FinSight application roles to internal RAG roles."""

from types import MappingProxyType
from typing import Final, Mapping

from src.security.access_control import validate_role


ROLE_MAPPING: Final[Mapping[str, str]] = MappingProxyType({
    # Exact values from public.profiles
    "admin": "admin",
    "business_analyst": "business_analyst",
    "risk_analyst": "risk_analyst",
    "customer": "customer",

    # Existing application-facing role labels
    "Admin": "admin",
    "Analyst": "business_analyst",
    "Business Analyst": "business_analyst",
    "Customer": "customer",
    "Risk Analyst": "risk_analyst",
    "Support Agent": "risk_analyst",
})


def map_application_role(application_role: object) -> str:
    """Return an exact mapped RAG role or reject an unknown application role."""

    if not isinstance(application_role, str):
        raise ValueError("The application role is missing or invalid.")

    rag_role = ROLE_MAPPING.get(application_role)
    if rag_role is None:
        raise ValueError("The application role is not mapped.")
    return validate_role(rag_role)
