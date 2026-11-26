"""Static contract tests for Phase 3A Supabase migration SQL."""

import re
from pathlib import Path


MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "supabase"
    / "migrations"
    / "202611250001_local_rag_approval_and_audit.sql"
)


def _read_sql() -> str:
    return MIGRATION_PATH.read_text(encoding="utf-8")


def _read_sql_no_comments() -> str:
    return re.sub(r"--[^\n]*", "", _read_sql())


def test_preserves_existing_google_ai_studio_objects():
    sql = _read_sql_no_comments()
    assert "public.rag_documents" not in sql
    assert "public.rag_chunks" not in sql
    assert "public.match_rag_chunks" not in sql


def test_all_security_definer_functions_use_pg_catalog_first_search_path():
    sql = _read_sql_no_comments()
    sec_definer_count = len(re.findall(r"SECURITY DEFINER", sql))
    search_path_count = len(
        re.findall(r"SET search_path = pg_catalog, public, pg_temp", sql)
    )
    assert sec_definer_count == 15
    assert sec_definer_count == search_path_count


def test_assert_active_admin_is_null_safe():
    sql = _read_sql()
    match = re.search(
        r"CREATE OR REPLACE FUNCTION public\.local_rag_assert_active_admin.*?\$\$;",
        sql,
        re.DOTALL,
    )
    assert match is not None
    body = match.group(0)
    assert "v_status IS NULL" in body
    assert "v_status IS DISTINCT FROM 'active'" in body
    assert "v_profile_role IS NULL" in body
    assert "v_profile_role NOT IN ('admin', 'Admin')" in body


def test_supersession_and_concurrency_in_register_document():
    sql = _read_sql()
    match = re.search(
        r"CREATE OR REPLACE FUNCTION public\.local_rag_register_document.*?\$\$;",
        sql,
        re.DOTALL,
    )
    assert match is not None
    body = match.group(0)
    assert "pg_advisory_xact_lock" in body
    assert "approval_status = 'superseded'" in body
    assert "'document_superseded'" in body
    assert "superseded_by_document_id" in body


def test_section_parent_change_prohibited_and_lifecycle_handled():
    sql = _read_sql()
    match = re.search(
        r"CREATE OR REPLACE FUNCTION public\.local_rag_invalidate_on_section_lifecycle.*?\$\$;",
        sql,
        re.DOTALL,
    )
    assert match is not None
    body = match.group(0)
    assert "NEW.document_id IS DISTINCT FROM OLD.document_id" in body
    assert "AFTER INSERT OR UPDATE OR DELETE ON public.local_rag_document_sections" in sql


def test_start_indexing_rejects_duplicate_indexing_and_fail_indexing_hardened():
    sql = _read_sql()
    start_match = re.search(
        r"CREATE OR REPLACE FUNCTION public\.local_rag_start_indexing.*?\$\$;",
        sql,
        re.DOTALL,
    )
    fail_match = re.search(
        r"CREATE OR REPLACE FUNCTION public\.local_rag_fail_indexing.*?\$\$;",
        sql,
        re.DOTALL,
    )
    assert start_match is not None and fail_match is not None
    start_body = start_match.group(0)
    fail_body = fail_match.group(0)

    assert "v_doc.indexing_status NOT IN ('not_indexed', 'failed', 'stale')" in start_body
    assert "v_doc.approval_status <> 'approved'" in fail_body
    assert "v_doc.indexing_status <> 'indexing'" in fail_body
    assert "pg_catalog.jsonb_typeof(p_error_details) <> 'object'" in fail_body


def test_reject_and_revoke_clear_conflicting_decision_fields():
    sql = _read_sql()
    assert "local_rag_documents_approved_fields_chk" in sql
    assert "local_rag_documents_rejected_fields_chk" in sql
    assert "local_rag_documents_revoked_fields_chk" in sql
    assert "local_rag_documents_superseded_fields_chk" in sql


def test_unnest_alias_and_verified_at_timestamp_validation():
    sql = _read_sql()
    assert "pg_catalog.unnest(v_section.approved_roles) AS ar(role_name)" in sql
    assert "v_verified_at := (p_details->>'verified_at')::timestamptz;" in sql


def test_sha256_hex_validation_on_documents_and_sections():
    sql = _read_sql_no_comments()
    assert "file_hash_sha256 ~ '^[0-9a-f]{64}$'" in sql
    assert "section_sha256 ~ '^[0-9a-f]{64}$'" in sql
    assert "public.local_rag_is_sha256_hex(p_file_hash_sha256)" in sql
    assert "public.local_rag_is_sha256_hex(v_sec_hash)" in sql
    assert "v_file_hash := pg_catalog.lower(pg_catalog.btrim(p_file_hash_sha256));" in sql
    assert "v_sec_hash := pg_catalog.lower(v_sec_hash);" in sql


def test_quarantine_resolution_required_and_audited():
    sql = _read_sql_no_comments()
    assert "local_rag_sections_quarantine_resolution_chk" in sql
    assert "quarantine_resolution_reason" in sql
    assert "v_section.classifier_quarantined AND v_quarantine_reason IS NULL" in sql


def test_supersession_fk_uses_on_delete_restrict():
    sql = _read_sql_no_comments()
    assert "REFERENCES public.local_rag_documents(id) ON DELETE RESTRICT" in sql
    assert "ON DELETE SET NULL" not in sql


def test_complete_indexing_requires_verified_collections_evidence():
    sql = _read_sql_no_comments()
    assert "p_details->'verified_collections'" in sql
    assert "pg_catalog.jsonb_typeof(vc.val) <> 'string'" in sql
    assert "v_vc_count <> v_vc_distinct_count" in sql
    assert "v_verified_collections IS DISTINCT FROM v_expected_collections" in sql


def test_null_reject_revoke_action_is_explicitly_rejected():
    sql = _read_sql_no_comments()
    match = re.search(
        r"CREATE OR REPLACE FUNCTION public\.local_rag_reject_or_revoke_document.*?\$\$;",
        sql,
        re.DOTALL,
    )
    assert match is not None
    body = match.group(0)
    assert "p_action IS NULL OR p_action NOT IN ('rejected', 'revoked')" in body


def test_indexing_rpcs_reject_null_version_and_invalid_expected_hash():
    sql = _read_sql_no_comments()
    for fn_name in (
        "local_rag_start_indexing",
        "local_rag_complete_indexing",
        "local_rag_fail_indexing",
    ):
        match = re.search(
            rf"CREATE OR REPLACE FUNCTION public\.{fn_name}.*?\$\$;",
            sql,
            re.DOTALL,
        )
        assert match is not None
        body = match.group(0)
        assert "p_expected_approval_version IS NULL OR p_expected_approval_version < 1" in body
        assert "NOT public.local_rag_is_sha256_hex(p_expected_file_hash)" in body
        assert "v_expected_file_hash := pg_catalog.lower(pg_catalog.btrim(p_expected_file_hash));" in body


def test_state_machine_transitions_enforced():
    sql = _read_sql_no_comments()
    approve_match = re.search(
        r"CREATE OR REPLACE FUNCTION public\.local_rag_approve_document.*?\$\$;",
        sql,
        re.DOTALL,
    )
    reject_match = re.search(
        r"CREATE OR REPLACE FUNCTION public\.local_rag_reject_or_revoke_document.*?\$\$;",
        sql,
        re.DOTALL,
    )
    assert approve_match is not None and reject_match is not None
    assert "v_doc.approval_status IS DISTINCT FROM 'pending'" in approve_match.group(0)
    reject_body = reject_match.group(0)
    assert "p_action = 'rejected' AND v_doc.approval_status IS DISTINCT FROM 'pending'" in reject_body
    assert "p_action = 'revoked' AND v_doc.approval_status IS DISTINCT FROM 'approved'" in reject_body


def test_classifier_quarantined_requires_json_boolean():
    sql = _read_sql_no_comments()
    match = re.search(
        r"CREATE OR REPLACE FUNCTION public\.local_rag_register_document.*?\$\$;",
        sql,
        re.DOTALL,
    )
    assert match is not None
    body = match.group(0)
    assert "pg_catalog.jsonb_typeof(v_sec->'classifier_quarantined') <> 'boolean'" in body
