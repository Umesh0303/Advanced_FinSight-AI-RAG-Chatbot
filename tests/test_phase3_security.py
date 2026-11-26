"""Security, approval-indexing, failure-recovery, and Admin API tests (Phases 3C-3E)."""

import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import src.api.services.local_rag_store as store
import src.ingestion.ingest as ingest_module
from src.api.main import create_app
from src.config import settings
from src.retrieval.retriever import retrieve_documents
from tests.test_api import bearer, configure_auth, make_token


ROLES = ("admin", "business_analyst", "risk_analyst", "customer")


class MockChromaCollection:
    def __init__(self, docs_with_scores=None):
        self.docs_with_scores = list(docs_with_scores or [])
        self.stored: dict[str, Document] = {}
        self.fail_add = False

    def similarity_search_with_relevance_scores(self, query, k):
        return self.docs_with_scores[:k]

    def get(self, ids=None, where=None, include=None, limit=None):
        matching = list(self.stored)
        if ids is not None:
            matching = [cid for cid in matching if cid in ids]
        if where is not None:
            matching = [
                cid
                for cid in matching
                if all(
                    self.stored[cid].metadata.get(k) == v
                    for k, v in where.items()
                )
            ]
        if limit is not None:
            matching = matching[:limit]
        res = {"ids": matching}
        if include and "metadatas" in include:
            res["metadatas"] = [self.stored[cid].metadata for cid in matching]
        return res

    def add_documents(self, documents, ids):
        if self.fail_add:
            self.fail_add = False
            raise RuntimeError("synthetic Chroma write failure")
        for cid, doc in zip(ids, documents):
            self.stored[cid] = doc

    def delete(self, ids):
        for cid in ids:
            self.stored.pop(cid, None)


def _chunk_doc(doc_id: str, sec_id: str, role: str, allowed_roles: list[str], chunk_id: str = None) -> Document:
    import json

    return Document(
        page_content=f"Evidence for {role}",
        metadata={
            "document_id": doc_id,
            "section_id": sec_id,
            "chunk_id": chunk_id or str(uuid4()),
            "role": role,
            "allowed_roles": json.dumps(allowed_roles),
            "source": "policy.pdf",
            "section_title": "1. Policy",
            "page_number": 1,
        },
    )


def test_verify_chunks_enforces_approval_indexing_ownership_and_roles(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")

    approved_doc = str(uuid4())
    pending_doc = str(uuid4())
    revoked_doc = str(uuid4())
    other_doc = str(uuid4())

    sec_cust = str(uuid4())
    sec_risk = str(uuid4())
    sec_excluded = str(uuid4())
    sec_pending_doc = str(uuid4())
    sec_revoked_doc = str(uuid4())

    def fake_get(url, *, params, headers, timeout, follow_redirects):
        if url.endswith("/local_rag_documents"):
            return SimpleNamespace(
                status_code=200,
                json=lambda: [
                    {"id": approved_doc, "approval_status": "approved", "indexing_status": "indexed"},
                    {"id": pending_doc, "approval_status": "pending", "indexing_status": "not_indexed"},
                    {"id": revoked_doc, "approval_status": "revoked", "indexing_status": "stale"},
                ],
            )
        if url.endswith("/local_rag_document_sections"):
            return SimpleNamespace(
                status_code=200,
                json=lambda: [
                    {"id": sec_cust, "document_id": approved_doc, "review_status": "approved", "approved_roles": ["customer"]},
                    {"id": sec_risk, "document_id": approved_doc, "review_status": "approved", "approved_roles": ["risk_analyst"]},
                    {"id": sec_excluded, "document_id": approved_doc, "review_status": "excluded", "approved_roles": []},
                    {"id": sec_pending_doc, "document_id": pending_doc, "review_status": "approved", "approved_roles": ["admin", "customer"]},
                    {"id": sec_revoked_doc, "document_id": revoked_doc, "review_status": "approved", "approved_roles": ["admin", "customer"]},
                ],
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(store.httpx, "get", fake_get)

    refs = [
        (approved_doc, sec_cust),
        (approved_doc, sec_risk),
        (approved_doc, sec_excluded),
        (pending_doc, sec_pending_doc),
        (revoked_doc, sec_revoked_doc),
        (other_doc, sec_cust),  # section ownership mismatch
        ("not-a-uuid", sec_cust),  # legacy non-UUID blocked
    ]

    assert store.verify_chunks_for_retrieval(refs, "customer") == {(approved_doc, sec_cust)}
    assert store.verify_chunks_for_retrieval(refs, "risk_analyst") == {(approved_doc, sec_risk)}
    assert store.verify_chunks_for_retrieval(refs, "business_analyst") == set()
    # Admin bypasses role membership only, NEVER pending/revoked/excluded/ownership
    assert store.verify_chunks_for_retrieval(refs, "admin") == {
        (approved_doc, sec_cust),
        (approved_doc, sec_risk),
    }


def test_retrieve_documents_fails_closed_on_supabase_outage(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test")

    doc_id = str(uuid4())
    sec_id = str(uuid4())
    collections = {
        role: MockChromaCollection(
            [(_chunk_doc(doc_id, sec_id, role, ["customer"]), 0.9)]
            if role == "customer"
            else []
        )
        for role in ROLES
    }

    monkeypatch.setattr(
        store,
        "verify_chunks_for_retrieval",
        lambda refs, role: (_ for _ in ()).throw(
            store.LocalRagStoreUnavailable("outage")
        ),
    )

    with pytest.raises(RuntimeError, match="Supabase retrieval authorization"):
        retrieve_documents("What is the policy?", "customer", collections)


def test_index_approved_document_skips_excluded_and_verifies_chromadb(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    doc_file = uploads / "policy.txt"
    sec0_text = "1. Public Section\nCustomer may read this."
    sec1_text = "2. Excluded Section\nSecret unapproved text."
    full_text = f"{sec0_text}\n\n{sec1_text}"
    doc_file.write_text(full_text, encoding="utf-8")

    monkeypatch.setattr("src.ingestion.document_loader.UPLOADS_DIR", uploads)

    file_hash = hashlib.sha256(doc_file.read_bytes()).hexdigest().lower()
    sec0_hash = hashlib.sha256(sec0_text.encode("utf-8")).hexdigest().lower()
    sec1_hash = hashlib.sha256(sec1_text.encode("utf-8")).hexdigest().lower()

    doc_id = str(uuid4())
    sec0_id = str(uuid4())
    sec1_id = str(uuid4())
    actor_id = str(uuid4())

    monkeypatch.setattr(
        ingest_module,
        "load_document",
        lambda path: [Document(page_content=full_text, metadata={"page": 0})],
    )
    monkeypatch.setattr(
        store,
        "get_document_by_id",
        lambda did: {
            "id": doc_id,
            "file_hash_sha256": file_hash,
            "source_id": "s" * 64,
            "filename": "policy.txt",
            "approval_status": "approved",
            "approval_version": 2,
        },
    )
    monkeypatch.setattr(
        store,
        "get_document_sections",
        lambda did: [
            {
                "id": sec0_id,
                "section_index": 0,
                "section_title": "1. Public Section",
                "section_sha256": sec0_hash,
                "review_status": "approved",
                "approved_roles": ["customer"],
            },
            {
                "id": sec1_id,
                "section_index": 1,
                "section_title": "2. Excluded Section",
                "section_sha256": sec1_hash,
                "review_status": "excluded",
                "approved_roles": [],
            },
        ],
    )

    events = []
    monkeypatch.setattr(
        store,
        "start_indexing",
        lambda **kw: events.append(("start", kw)) or {"id": doc_id},
    )
    monkeypatch.setattr(
        store,
        "complete_indexing",
        lambda **kw: events.append(("complete", kw))
        or {"id": doc_id, "approval_status": "approved", "indexing_status": "indexed", "approval_version": 2},
    )
    monkeypatch.setattr(
        store,
        "fail_indexing",
        lambda **kw: events.append(("fail", kw)) or {"id": doc_id},
    )

    collections = {role: MockChromaCollection() for role in ROLES}
    summary = ingest_module.index_approved_document(
        doc_file,
        doc_id,
        collections=collections,
        actor_id=actor_id,
    )

    assert summary["indexing_status"] == "indexed"
    assert summary["verified_collections"] == ["customer"]
    assert len(collections["customer"].stored) == 1
    assert len(collections["admin"].stored) == 0
    assert [e[0] for e in events] == ["start", "complete"]

    # Partial failure recovery: Chroma write failure triggers fail_indexing
    events.clear()
    collections_fail = {role: MockChromaCollection() for role in ROLES}
    collections_fail["customer"].fail_add = True
    with pytest.raises(RuntimeError, match="synthetic Chroma write failure"):
        ingest_module.index_approved_document(
            doc_file,
            doc_id,
            collections=collections_fail,
            actor_id=actor_id,
        )
    assert [e[0] for e in events] == ["start", "fail"]


def test_admin_document_endpoints_enforce_admin_and_workflow(monkeypatch):
    doc_id = str(uuid4())
    sec_id = str(uuid4())
    collections = {role: MockChromaCollection() for role in ROLES}
    collections["customer"].stored["c1"] = _chunk_doc(doc_id, sec_id, "customer", ["customer"], "c1")

    app = create_app(
        resources_factory=lambda: collections,
        cors_origins=["https://finsight.example"],
    )

    with TestClient(app) as client:
        # Non-admin denied
        cust_key = configure_auth(monkeypatch, application_role="Customer")
        assert client.get("/api/v1/documents", headers=bearer(make_token(cust_key))).status_code == 403
        assert client.post(
            f"/api/v1/documents/{doc_id}/reject",
            headers=bearer(make_token(cust_key)),
            json={"decision_reason": "no"},
        ).status_code == 403
        assert client.delete(
            f"/api/v1/documents/{doc_id}",
            headers=bearer(make_token(cust_key)),
        ).status_code == 403

        # Admin allowed
        admin_key = configure_auth(monkeypatch, application_role="Admin")
        monkeypatch.setattr(store, "list_documents", lambda: [{"id": doc_id, "approval_status": "pending"}])
        assert client.get("/api/v1/documents", headers=bearer(make_token(admin_key))).status_code == 200

        monkeypatch.setattr(
            store,
            "approve_document",
            lambda **kw: {"id": doc_id, "approval_status": "approved", "indexing_status": "not_indexed"},
        )
        approve_resp = client.post(
            f"/api/v1/documents/{doc_id}/approve",
            headers=bearer(make_token(admin_key)),
            json={
                "section_decisions": [
                    {"section_id": sec_id, "review_status": "approved", "approved_roles": ["customer"]}
                ],
                "decision_reason": "Reviewed",
            },
        )
        assert approve_resp.status_code == 200

        monkeypatch.setattr(
            store,
            "reject_or_revoke_document",
            lambda **kw: {"id": doc_id, "approval_status": "revoked", "indexing_status": "stale"},
        )
        monkeypatch.setattr(store, "delete_pgvector_chunks", lambda did: 2)
        revoke_resp = client.post(
            f"/api/v1/documents/{doc_id}/revoke",
            headers=bearer(make_token(admin_key)),
            json={"decision_reason": "Outdated policy"},
        )
        assert revoke_resp.status_code == 200
        assert revoke_resp.json()["removed_chunks"] == 3
        assert collections["customer"].stored == {}

        # Admin delete removes DB record, pgvector chunks, and storage object
        deleted_storage_calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            store,
            "delete_document",
            lambda **kw: {
                "document_id": doc_id,
                "filename": "policy.pdf",
                "file_hash_sha256": "a" * 64,
                "deleted_chunks": 4,
            },
        )
        monkeypatch.setattr(
            store,
            "delete_source_file",
            lambda h, fn: deleted_storage_calls.append((h, fn)) or True,
        )
        del_resp = client.delete(
            f"/api/v1/documents/{doc_id}",
            headers=bearer(make_token(admin_key)),
        )
        assert del_resp.status_code == 200
        del_body = del_resp.json()
        assert del_body["success"] is True
        assert del_body["status"] == "deleted"
        assert del_body["document_id"] == doc_id
        assert del_body["deleted_chunks"] == 4
        assert del_body["storage_deleted"] is True
        assert deleted_storage_calls == [("a" * 64, "policy.pdf")]


def test_sync_approved_documents_rehydrates_already_indexed_doc(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    doc_file = uploads / "policy.txt"
    sec0_text = "1. Public Section\nCustomer may read this."
    doc_file.write_text(sec0_text, encoding="utf-8")

    monkeypatch.setattr("src.ingestion.document_loader.UPLOADS_DIR", uploads)

    file_hash = hashlib.sha256(doc_file.read_bytes()).hexdigest().lower()
    sec0_hash = hashlib.sha256(sec0_text.encode("utf-8")).hexdigest().lower()
    doc_id = str(uuid4())
    sec0_id = str(uuid4())

    monkeypatch.setattr(
        ingest_module,
        "load_document",
        lambda path: [Document(page_content=sec0_text, metadata={"page": 0})],
    )
    monkeypatch.setattr(
        store,
        "list_documents",
        lambda limit=500: [
            {
                "id": doc_id,
                "file_hash_sha256": file_hash,
                "source_id": "s" * 64,
                "filename": "policy.txt",
                "approval_status": "approved",
                "indexing_status": "indexed",
            }
        ],
    )
    monkeypatch.setattr(
        store,
        "get_document_sections",
        lambda did: [
            {
                "id": sec0_id,
                "section_index": 0,
                "section_title": "1. Public Section",
                "section_sha256": sec0_hash,
                "review_status": "approved",
                "approved_roles": ["customer"],
            }
        ],
    )
    monkeypatch.setattr(
        store,
        "start_indexing",
        lambda **kw: (_ for _ in ()).throw(AssertionError("should not call start_indexing")),
    )

    collections = {role: MockChromaCollection() for role in ROLES}
    summaries = ingest_module.sync_approved_documents_if_empty(
        collections,
        uploads_dir=uploads,
    )
    assert len(summaries) == 1
    assert summaries[0]["indexing_status"] == "indexed"
    assert len(collections["customer"].stored) == 1


def test_sync_approved_documents_downloads_from_supabase_storage(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    sec0_text = "1. Public Section\nCustomer may read this."
    raw_bytes = sec0_text.encode("utf-8")

    monkeypatch.setattr("src.ingestion.document_loader.UPLOADS_DIR", uploads)

    file_hash = hashlib.sha256(raw_bytes).hexdigest().lower()
    sec0_hash = hashlib.sha256(raw_bytes).hexdigest().lower()
    doc_id = str(uuid4())
    sec0_id = str(uuid4())

    monkeypatch.setattr(
        ingest_module,
        "load_document",
        lambda path: [Document(page_content=sec0_text, metadata={"page": 0})],
    )
    monkeypatch.setattr(
        store,
        "list_documents",
        lambda limit=500: [
            {
                "id": doc_id,
                "file_hash_sha256": file_hash,
                "source_id": "s" * 64,
                "filename": "remote_only.txt",
                "approval_status": "approved",
                "indexing_status": "indexed",
            }
        ],
    )
    monkeypatch.setattr(
        store,
        "get_document_sections",
        lambda did: [
            {
                "id": sec0_id,
                "section_index": 0,
                "section_title": "1. Public Section",
                "section_sha256": sec0_hash,
                "review_status": "approved",
                "approved_roles": ["customer"],
            }
        ],
    )

    def fake_download(fh, fname, dest):
        assert fh == file_hash
        assert fname == "remote_only.txt"
        dest.write_bytes(raw_bytes)
        return True

    monkeypatch.setattr(store, "download_source_file", fake_download)

    collections = {role: MockChromaCollection() for role in ROLES}
    summaries = ingest_module.sync_approved_documents_if_empty(
        collections,
        uploads_dir=uploads,
    )
    assert len(summaries) == 1
    assert summaries[0]["indexing_status"] == "indexed"
    assert len(collections["customer"].stored) == 1
    assert list(uploads.iterdir()) == []


def test_index_approved_document_pgvector_and_retrieve(tmp_path, monkeypatch):
    from src.chains.rag_chain import ask_question
    from src.retrieval.retriever import retrieve_documents_pgvector

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    doc_file = uploads / "policy.txt"
    sec0_text = "1. Public Section\nCustomer may read this."
    doc_file.write_text(sec0_text, encoding="utf-8")

    monkeypatch.setattr("src.ingestion.document_loader.UPLOADS_DIR", uploads)

    file_hash = hashlib.sha256(doc_file.read_bytes()).hexdigest().lower()
    sec0_hash = hashlib.sha256(sec0_text.encode("utf-8")).hexdigest().lower()
    doc_id = str(uuid4())
    sec0_id = str(uuid4())
    actor_id = str(uuid4())

    monkeypatch.setattr(
        ingest_module,
        "load_document",
        lambda path: [Document(page_content=sec0_text, metadata={"page": 0})],
    )
    monkeypatch.setattr(
        store,
        "get_document_by_id",
        lambda did: {
            "id": doc_id,
            "file_hash_sha256": file_hash,
            "source_id": "s" * 64,
            "filename": "policy.txt",
            "approval_status": "approved",
            "approval_version": 2,
        },
    )
    monkeypatch.setattr(
        store,
        "get_document_sections",
        lambda did: [
            {
                "id": sec0_id,
                "section_index": 0,
                "section_title": "1. Public Section",
                "section_sha256": sec0_hash,
                "review_status": "approved",
                "approved_roles": ["customer"],
            }
        ],
    )

    events = []
    monkeypatch.setattr(
        store,
        "start_indexing",
        lambda **kw: events.append("start") or {"id": doc_id},
    )
    monkeypatch.setattr(
        store,
        "upsert_pgvector_chunks",
        lambda **kw: events.append(("upsert", len(kw["chunks"])))
        or {"document_id": doc_id, "inserted_count": len(kw["chunks"])},
    )
    monkeypatch.setattr(store, "count_pgvector_chunks", lambda did: 1)
    monkeypatch.setattr(
        store,
        "complete_indexing",
        lambda **kw: events.append("complete")
        or {"id": doc_id, "approval_status": "approved", "indexing_status": "indexed", "approval_version": 2},
    )

    class FakeEmbedder:
        def embed_documents(self, texts):
            return [[0.1] * 768 for _ in texts]

        def embed_query(self, query):
            return [0.1] * 768

    summary = ingest_module.index_approved_document_pgvector(
        doc_file,
        doc_id,
        embedding_model=FakeEmbedder(),
        actor_id=actor_id,
    )
    assert summary["indexing_status"] == "indexed"
    assert summary["verified_chunk_count"] == 1
    assert events == ["start", ("upsert", 1), "complete"]

    monkeypatch.setattr(
        store,
        "match_pgvector_chunks",
        lambda **kw: [
            {
                "id": str(uuid4()),
                "document_id": doc_id,
                "section_id": sec0_id,
                "chunk_id": "chunk-0",
                "chunk_index": 0,
                "page_number": 1,
                "content": sec0_text,
                "metadata": {"source": "policy.txt", "section_title": "1. Public Section"},
                "similarity": 0.92,
            }
        ],
    )
    docs = retrieve_documents_pgvector("policy", "customer", embedding_model=FakeEmbedder())
    assert len(docs) == 1
    assert docs[0].metadata["document_id"] == doc_id

    class FakeLLM:
        def invoke(self, prompt_value):
            return "Grounded pgvector answer."

    rag_resp = ask_question(
        "What is the policy?",
        "customer",
        {"backend": "pgvector", "embedding_model": FakeEmbedder()},
        llm=FakeLLM(),
    )
    assert rag_resp.answer == "Grounded pgvector answer."
    assert rag_resp.retrieved_count == 1
