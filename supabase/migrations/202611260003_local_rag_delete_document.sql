-- Additive migration: Controlled hard-delete for local RAG documents, sections, and pgvector chunks.
-- Preserves immutable audit history by recording a 'document_deleted' entry in local_rag_document_audit_log.
-- Preserves existing public.rag_documents, public.rag_chunks, and public.match_rag_chunks untouched.

-- 1. Allow 'document_deleted' in the immutable audit log action constraint
ALTER TABLE public.local_rag_document_audit_log
    DROP CONSTRAINT IF EXISTS local_rag_document_audit_log_action_check;

ALTER TABLE public.local_rag_document_audit_log
    ADD CONSTRAINT local_rag_document_audit_log_action_check CHECK (action IN (
        'registered',
        'classified',
        'section_approved',
        'section_excluded',
        'document_approved',
        'document_rejected',
        'document_revoked',
        'document_superseded',
        'document_deleted',
        'approval_invalidated',
        'indexing_started',
        'indexing_verified',
        'indexing_failed',
        'reconciled'
    ));

-- 2. Extend direct-write guard to allow controlled delete_document RPC context
CREATE OR REPLACE FUNCTION public.local_rag_guard_direct_writes()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_rpc_ctx text := pg_catalog.current_setting('local_rag.rpc_context', true);
BEGIN
    IF COALESCE(v_rpc_ctx, '') NOT IN (
        'register',
        'approve',
        'reject_revoke',
        'start_indexing',
        'complete_indexing',
        'fail_indexing',
        'pgvector_index',
        'pgvector_cleanup',
        'delete_document',
        'invalidation_trigger'
    ) THEN
        RAISE EXCEPTION 'Direct writes to % are prohibited; use controlled local_rag_* RPCs.', TG_TABLE_NAME;
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$;

-- 3. Controlled RPC to delete a document, its sections, and its pgvector chunks
CREATE OR REPLACE FUNCTION public.local_rag_delete_document(
    p_document_id uuid,
    p_actor_id uuid
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_actor_role text;
    v_doc public.local_rag_documents;
    v_deleted_chunks integer := 0;
BEGIN
    v_actor_role := public.local_rag_assert_active_admin(p_actor_id);

    SELECT * INTO v_doc
    FROM public.local_rag_documents AS d
    WHERE d.id = p_document_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Document % not found.', p_document_id;
    END IF;

    IF v_doc.indexing_status = 'indexing' THEN
        RAISE EXCEPTION 'Cannot delete document % while indexing is in progress.', p_document_id;
    END IF;

    PERFORM pg_catalog.set_config('local_rag.rpc_context', 'delete_document', true);

    INSERT INTO public.local_rag_document_audit_log (
        document_id, action, actor_id, actor_role, details
    ) VALUES (
        p_document_id,
        'document_deleted',
        p_actor_id,
        v_actor_role,
        pg_catalog.jsonb_build_object(
            'filename', v_doc.filename,
            'file_hash_sha256', v_doc.file_hash_sha256,
            'approval_status', v_doc.approval_status,
            'indexing_status', v_doc.indexing_status,
            'approval_version', v_doc.approval_version
        )
    );

    DELETE FROM public.local_rag_chunks
    WHERE document_id = p_document_id;
    GET DIAGNOSTICS v_deleted_chunks = ROW_COUNT;

    DELETE FROM public.local_rag_document_sections
    WHERE document_id = p_document_id;

    DELETE FROM public.local_rag_documents
    WHERE id = p_document_id;

    RETURN pg_catalog.jsonb_build_object(
        'document_id', v_doc.id,
        'filename', v_doc.filename,
        'file_hash_sha256', v_doc.file_hash_sha256,
        'deleted_chunks', v_deleted_chunks
    );
END;
$$;

REVOKE ALL ON FUNCTION public.local_rag_delete_document(uuid, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.local_rag_delete_document(uuid, uuid) TO service_role;
