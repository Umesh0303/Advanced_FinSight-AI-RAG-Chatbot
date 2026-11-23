"""Tests for secure role-scoped similarity retrieval."""

import json

import pytest
from langchain_core.documents import Document

from src.config.settings import COLLECTION_NAMES
from src.retrieval.retriever import retrieve_documents


class MockCollection:
    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.search_calls = []

    def similarity_search_with_relevance_scores(self, query, k):
        self.search_calls.append((query, k))
        if self.error is not None:
            raise self.error
        return self.results[:k]


@pytest.fixture
def collections():
    return {
        name: MockCollection()
        for name in (*COLLECTION_NAMES, "unauthorized_extra")
    }


def make_document(
    chunk_id="chunk-1",
    document_id="document-1",
    role="customer",
    allowed_roles=None,
    content="retrieved content",
    **metadata_overrides,
):
    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "role": role,
        "allowed_roles": json.dumps(
            allowed_roles if allowed_roles is not None else [role]
        ),
        "source": "finance.pdf",
        "section_title": "Forecast",
        "page_number": 1,
    }
    metadata.update(metadata_overrides)
    return Document(page_content=content, metadata=metadata)


def test_results_are_sorted_limited_and_use_highest_scoring_duplicate(
    collections,
):
    duplicate_for_customer = make_document(
        chunk_id="shared",
        role="customer",
        allowed_roles=["customer", "admin"],
        content="customer copy",
    )
    duplicate_for_admin = make_document(
        chunk_id="shared",
        role="admin",
        allowed_roles=["customer", "admin"],
        content="admin copy",
    )
    collections["customer"].results = [
        (make_document("low", content="low relevance"), 0.4),
        (duplicate_for_customer, 0.7),
    ]
    collections["admin"].results = [
        (make_document("high", role="admin", allowed_roles=["admin"]), 0.9),
        (duplicate_for_admin, 0.8),
    ]
    collections["business_analyst"].results = [
        (make_document("middle", role="business_analyst"), 0.6),
    ]

    results = retrieve_documents("forecast", "admin", collections, top_k=3)

    assert [document.metadata["chunk_id"] for document in results] == [
        "high",
        "shared",
        "middle",
    ]
    assert [document.metadata["relevance_score"] for document in results] == [
        0.9,
        0.8,
        0.6,
    ]
    assert results[1].page_content == "admin copy"
    assert results[1].metadata["role"] == "admin"


def test_search_uses_authorized_collections_and_overfetches_for_admin(
    collections,
):
    results = retrieve_documents("forecast", "admin", collections, top_k=2)

    assert results == []
    assert all(
        collections[name].search_calls == [("forecast", 8)]
        for name in COLLECTION_NAMES
    )
    assert collections["unauthorized_extra"].search_calls == []


@pytest.mark.parametrize(
    ("role", "expected_collection"),
    [
        ("business_analyst", "business_analyst"),
        ("risk_analyst", "risk_analyst"),
        ("customer", "customer"),
    ],
)
def test_non_admin_searches_only_its_collection(
    collections,
    role,
    expected_collection,
):
    retrieve_documents("forecast", role, collections, top_k=3)

    assert collections[expected_collection].search_calls == [("forecast", 3)]
    for name in COLLECTION_NAMES:
        if name != expected_collection:
            assert collections[name].search_calls == []
    assert collections["unauthorized_extra"].search_calls == []


def test_relevant_result_preserves_content_metadata_and_original_document(
    collections,
):
    original = make_document(
        role="customer",
        allowed_roles=["customer"],
        content="precise forecast",
        custom="preserved",
    )
    original_metadata = dict(original.metadata)
    collections["customer"].results = [(original, 0.93)]

    results = retrieve_documents("forecast", "customer", collections)

    assert len(results) == 1
    assert results[0] is not original
    assert results[0].page_content == original.page_content
    assert {
        key: value
        for key, value in results[0].metadata.items()
        if key != "relevance_score"
    } == original_metadata
    assert results[0].metadata["relevance_score"] == 0.93
    assert original.metadata == original_metadata
    assert "relevance_score" not in original.metadata


def test_empty_search_results_return_empty_list(collections):
    assert retrieve_documents("forecast", "customer", collections) == []


def test_search_error_is_wrapped_with_collection_context(collections):
    original_error = OSError("database unavailable")
    collections["customer"].error = original_error

    with pytest.raises(RuntimeError, match="customer") as exc_info:
        retrieve_documents("forecast", "customer", collections)

    assert exc_info.value.__cause__ is original_error


@pytest.mark.parametrize(
    "malformed_results",
    [None, "not results", [("not a document", 0.5)], [(make_document(),)]],
)
def test_malformed_search_results_raise_runtime_error(
    collections,
    malformed_results,
):
    def malformed_search(query, *, k):
        assert query == "query"
        assert k > 0
        return malformed_results

    collections["customer"].similarity_search_with_relevance_scores = (
        malformed_search
    )

    with pytest.raises(RuntimeError, match="customer.*malformed"):
        retrieve_documents("query", "customer", collections)


@pytest.mark.parametrize("score", [True, float("nan"), -0.1, 1.1, 10**400])
def test_invalid_relevance_scores_raise_runtime_error(collections, score):
    collections["customer"].results = [(make_document(), score)]

    with pytest.raises(RuntimeError, match="customer.*relevance score"):
        retrieve_documents("query", "customer", collections)


@pytest.mark.parametrize(
    ("role", "collection_name", "allowed_roles"),
    [
        ("business_analyst", "business_analyst", ["risk_analyst"]),
        ("risk_analyst", "risk_analyst", ["customer"]),
        ("customer", "customer", ["admin"]),
    ],
)
def test_non_admin_never_receives_content_not_authorized_for_requester(
    collections,
    role,
    collection_name,
    allowed_roles,
):
    denied = make_document(
        role=collection_name,
        allowed_roles=allowed_roles,
        content="DENIED CONTENT",
    )
    collections[collection_name].results = [(denied, 0.99)]

    results = retrieve_documents("query", role, collections)

    assert results == []
    assert all("DENIED CONTENT" not in document.page_content for document in results)


def test_wrong_destination_role_is_rejected(collections):
    wrong_destination = make_document(
        role="customer",
        allowed_roles=["risk_analyst"],
        content="WRONG DESTINATION",
    )
    collections["risk_analyst"].results = [(wrong_destination, 0.99)]

    assert retrieve_documents("query", "risk_analyst", collections) == []


@pytest.mark.parametrize(
    "allowed_roles",
    [
        "{malformed-json",
        json.dumps([]),
        json.dumps(["customer", "customer"]),
        json.dumps(["customer", "executive"]),
        json.dumps([" customer "]),
        json.dumps("customer"),
    ],
)
def test_malformed_or_unauthorized_role_metadata_is_rejected(
    collections,
    allowed_roles,
):
    document = make_document(
        role="customer",
        allowed_roles=["customer"],
        allowed_roles_raw=allowed_roles,
    )
    document.metadata["allowed_roles"] = document.metadata.pop(
        "allowed_roles_raw"
    )
    collections["customer"].results = [(document, 0.99)]

    assert retrieve_documents("query", "customer", collections) == []


@pytest.mark.parametrize(
    "metadata_overrides",
    [
        {"chunk_id": None},
        {"document_id": ""},
        {"role": "executive"},
        {"source": " "},
        {"section_title": ""},
        {"allowed_roles": ["customer"]},
        {"page_number": 0},
        {"page_number": True},
    ],
)
def test_missing_or_invalid_security_metadata_is_rejected(
    collections,
    metadata_overrides,
):
    overrides = dict(metadata_overrides)
    role = overrides.pop("role", "customer")
    allowed_roles = overrides.pop("allowed_roles", ["customer"])
    document = make_document(
        role=role,
        allowed_roles=allowed_roles,
        **overrides,
    )
    if "allowed_roles" in metadata_overrides:
        document.metadata["allowed_roles"] = metadata_overrides["allowed_roles"]
    collections["customer"].results = [(document, 0.99)]

    assert retrieve_documents("query", "customer", collections) == []


@pytest.mark.parametrize(
    ("role", "collection_name"),
    [
        ("admin", "admin"),
        ("admin", "business_analyst"),
        ("admin", "risk_analyst"),
        ("admin", "customer"),
    ],
)
def test_admin_can_retrieve_valid_chunks_from_every_authorized_collection(
    collections,
    role,
    collection_name,
):
    document = make_document(
        chunk_id=f"{collection_name}-chunk",
        role=collection_name,
        allowed_roles=[collection_name],
        content=f"{collection_name} content",
    )
    collections[collection_name].results = [(document, 0.8)]

    results = retrieve_documents("query", role, collections)

    assert len(results) == 1
    assert results[0].page_content == f"{collection_name} content"


@pytest.mark.parametrize(
    "metadata_field",
    [
        "chunk_id",
        "document_id",
        "role",
        "allowed_roles",
        "source",
        "section_title",
    ],
)
def test_missing_required_metadata_is_rejected(collections, metadata_field):
    document = make_document(role="customer", allowed_roles=["customer"])
    del document.metadata[metadata_field]
    collections["customer"].results = [(document, 0.99)]

    assert retrieve_documents("query", "customer", collections) == []


def test_invalid_role_raises_existing_validation_exception(collections):
    with pytest.raises(ValueError, match="Unsupported role"):
        retrieve_documents("query", "executive", collections)


@pytest.mark.parametrize("query", ["", " ", "\t\n"])
def test_empty_or_whitespace_query_is_rejected(collections, query):
    with pytest.raises(ValueError, match="query"):
        retrieve_documents(query, "customer", collections)


def test_non_string_query_is_rejected(collections):
    with pytest.raises(TypeError, match="query"):
        retrieve_documents(None, "customer", collections)


@pytest.mark.parametrize("top_k", [0, -1, True, False, 1.5, "2"])
def test_invalid_top_k_is_rejected(collections, top_k):
    with pytest.raises(ValueError, match="top_k"):
        retrieve_documents("query", "customer", collections, top_k=top_k)


def test_missing_authorized_collection_propagates_router_error(collections):
    del collections["customer"]

    with pytest.raises(RuntimeError, match="customer.*unavailable"):
        retrieve_documents("query", "customer", collections)
