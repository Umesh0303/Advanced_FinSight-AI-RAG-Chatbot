-- Additive migration: Supabase pgvector storage and retrieval for local RAG.
-- Preserves existing public.rag_documents, public.rag_chunks, and public.match_rag_chunks untouched.

-- 1. pgvector chunks table linked to local_rag_documents and local_rag_document_sections
CREATE TABLE IF NOT EXISTS public.local_rag_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES public.local_rag_documents(id) ON DELETE CASCADE,
    section_id uuid NOT NULL REFERENCES public.local_rag_document_sections(id) ON DELETE CASCADE,
    chunk_id text NOT NULL,
    chunk_index integer NOT NULL CHECK (chunk_index >= 0),
    page_number integer CHECK (page_number IS NULL OR page_number > 0),
    content text NOT NULL CHECK (length(trim(content)) > 0),
    embedding extensions.vector(768) NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, section_id, chunk_index),
    UNIQUE (document_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_local_rag_chunks_doc_sec
    ON public.local_rag_chunks (document_id, section_id);

CREATE INDEX IF NOT EXISTS idx_local_rag_chunks_embedding_hnsw
    ON public.local_rag_chunks
    USING hnsw (embedding extensions.vector_cosine_ops);

-- 2. Extend direct-write guard to allow pgvector RPC contexts
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

DROP TRIGGER IF EXISTS trg_local_rag_chunks_write_guard ON public.local_rag_chunks;
CREATE TRIGGER trg_local_rag_chunks_write_guard
    BEFORE INSERT OR UPDATE OR DELETE ON public.local_rag_chunks
    FOR EACH ROW
    EXECUTE FUNCTION public.local_rag_guard_direct_writes();

-- 3. Controlled RPC to atomically replace and store pgvector chunks for an approved, indexing document
CREATE OR REPLACE FUNCTION public.local_rag_upsert_pgvector_chunks(
    p_document_id uuid,
    p_expected_approval_version integer,
    p_expected_file_hash text,
    p_chunks jsonb,
    p_actor_id uuid DEFAULT NULL
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, pg_temp
AS $$
DECLARE
    v_expected_file_hash text;
    v_doc public.local_rag_documents;
    v_chunk jsonb;
    v_sec_id uuid;
    v_chunk_id text;
    v_chunk_index integer;
    v_page_number integer;
    v_content text;
    v_embedding_text text;
    v_meta jsonb;
    v_inserted_count integer := 0;
BEGIN
    IF p_expected_approval_version IS NULL OR p_expected_approval_version < 1 THEN
        RAISE EXCEPTION 'p_expected_approval_version must be >= 1.';
    END IF;
    IF NOT public.local_rag_is_sha256_hex(p_expected_file_hash) THEN
        RAISE EXCEPTION 'p_expected_file_hash must be a valid 64-character hex SHA-256 digest.';
    END IF;
    v_expected_file_hash := pg_catalog.lower(pg_catalog.btrim(p_expected_file_hash));

    IF p_actor_id IS NOT NULL THEN
        PERFORM public.local_rag_assert_active_admin(p_actor_id);
    END IF;

    IF p_chunks IS NULL
       OR pg_catalog.jsonb_typeof(p_chunks) <> 'array'
       OR pg_catalog.jsonb_array_length(p_chunks) = 0 THEN
        RAISE EXCEPTION 'p_chunks must be a non-empty JSON array.';
    END IF;

    SELECT * INTO v_doc
    FROM public.local_rag_documents AS d
    WHERE d.id = p_document_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Document % not found.', p_document_id;
    END IF;

    IF v_doc.approval_status IS DISTINCT FROM 'approved' THEN
        RAISE EXCEPTION 'Cannot store pgvector chunks while approval_status is %.', v_doc.approval_status;
    END IF;

    IF v_doc.indexing_status IS DISTINCT FROM 'indexing' THEN
        RAISE EXCEPTION 'Document % must be in indexing status (current: %).', p_document_id, v_doc.indexing_status;
    END IF;

    IF v_doc.approval_version IS DISTINCT FROM p_expected_approval_version
       OR v_doc.file_hash_sha256 IS DISTINCT FROM v_expected_file_hash THEN
        RAISE EXCEPTION 'Stale pgvector indexing rejected: expected version %/hash %, current version %/hash %.',
            p_expected_approval_version, v_expected_file_hash, v_doc.approval_version, v_doc.file_hash_sha256;
    END IF;

    PERFORM pg_catalog.set_config('local_rag.rpc_context', 'pgvector_index', true);

    DELETE FROM public.local_rag_chunks
    WHERE document_id = p_document_id;

    FOR v_chunk IN
        SELECT c.item FROM pg_catalog.jsonb_array_elements(p_chunks) AS c(item)
    LOOP
        IF pg_catalog.jsonb_typeof(v_chunk) <> 'object' THEN
            RAISE EXCEPTION 'Each chunk entry must be a JSON object.';
        END IF;

        IF NOT public.local_rag_is_uuid_text(v_chunk->>'section_id') THEN
            RAISE EXCEPTION 'Malformed section_id in p_chunks.';
        END IF;
        v_sec_id := (v_chunk->>'section_id')::uuid;

        IF NOT EXISTS (
            SELECT 1
            FROM public.local_rag_document_sections AS s
            WHERE s.id = v_sec_id
              AND s.document_id = p_document_id
              AND s.review_status = 'approved'
        ) THEN
            RAISE EXCEPTION 'Section % is not an approved section of document %.', v_sec_id, p_document_id;
        END IF;

        v_chunk_id := NULLIF(pg_catalog.btrim(COALESCE(v_chunk->>'chunk_id', '')), '');
        IF v_chunk_id IS NULL THEN
            RAISE EXCEPTION 'chunk_id is required.';
        END IF;

        IF pg_catalog.jsonb_typeof(v_chunk->'chunk_index') <> 'number'
           OR COALESCE(v_chunk->>'chunk_index', '') !~ '^[0-9]+$' THEN
            RAISE EXCEPTION 'chunk_index must be a non-negative integer.';
        END IF;
        v_chunk_index := (v_chunk->>'chunk_index')::integer;

        v_page_number := NULL;
        IF v_chunk ? 'page_number' AND v_chunk->'page_number' IS NOT NULL AND pg_catalog.jsonb_typeof(v_chunk->'page_number') <> 'null' THEN
            IF pg_catalog.jsonb_typeof(v_chunk->'page_number') <> 'number'
               OR COALESCE(v_chunk->>'page_number', '') !~ '^[1-9][0-9]*$' THEN
                RAISE EXCEPTION 'page_number must be a positive integer.';
            END IF;
            v_page_number := (v_chunk->>'page_number')::integer;
        END IF;

        v_content := NULLIF(pg_catalog.btrim(COALESCE(v_chunk->>'content', '')), '');
        IF v_content IS NULL THEN
            RAISE EXCEPTION 'Chunk content must not be empty.';
        END IF;

        v_embedding_text := NULLIF(pg_catalog.btrim(COALESCE(v_chunk->>'embedding', '')), '');
        IF v_embedding_text IS NULL THEN
            RAISE EXCEPTION 'Chunk embedding is required.';
        END IF;

        v_meta := COALESCE(v_chunk->'metadata', '{}'::jsonb);
        IF pg_catalog.jsonb_typeof(v_meta) <> 'object' THEN
            RAISE EXCEPTION 'Chunk metadata must be a JSON object.';
        END IF;

        INSERT INTO public.local_rag_chunks (
            document_id,
            section_id,
            chunk_id,
            chunk_index,
            page_number,
            content,
            embedding,
            metadata
        ) VALUES (
            p_document_id,
            v_sec_id,
            v_chunk_id,
            v_chunk_index,
            v_page_number,
            v_chunk->>'content',
            v_embedding_text::extensions.vector(768),
            v_meta
        );

        v_inserted_count := v_inserted_count + 1;
    END LOOP;

    RETURN pg_catalog.jsonb_build_object(
        'document_id', p_document_id,
        'inserted_count', v_inserted_count
    );
END;
$$;

-- 4. Controlled RPC to delete pgvector chunks on revocation/rejection cleanup
CREATE OR REPLACE FUNCTION public.local_rag_delete_pgvector_chunks(
    p_document_id uuid
)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_deleted_count integer := 0;
BEGIN
    PERFORM pg_catalog.set_config('local_rag.rpc_context', 'pgvector_cleanup', true);

    DELETE FROM public.local_rag_chunks
    WHERE document_id = p_document_id;

    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    RETURN v_deleted_count;
END;
$$;

-- 5. Authoritative role-scoped pgvector similarity search
CREATE OR REPLACE FUNCTION public.match_local_rag_chunks(
    query_embedding extensions.vector(768),
    requesting_role text,
    match_count integer DEFAULT 5,
    match_threshold double precision DEFAULT 0.0
)
RETURNS TABLE (
    id uuid,
    document_id uuid,
    section_id uuid,
    chunk_id text,
    chunk_index integer,
    page_number integer,
    content text,
    metadata jsonb,
    similarity double precision
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions, pg_temp
AS $$
DECLARE
    v_limit integer := LEAST(GREATEST(COALESCE(match_count, 5), 1), 50);
    v_threshold double precision := COALESCE(match_threshold, 0.0);
BEGIN
    IF requesting_role IS NULL OR requesting_role NOT IN ('admin', 'business_analyst', 'risk_analyst', 'customer') THEN
        RAISE EXCEPTION 'Invalid requesting_role: %', COALESCE(requesting_role, 'NULL');
    END IF;

    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.section_id,
        c.chunk_id,
        c.chunk_index,
        c.page_number,
        c.content,
        c.metadata,
        (1.0 - (c.embedding <=> query_embedding))::double precision AS similarity
    FROM public.local_rag_chunks AS c
    INNER JOIN public.local_rag_documents AS d
        ON d.id = c.document_id
    INNER JOIN public.local_rag_document_sections AS s
        ON s.id = c.section_id
       AND s.document_id = d.id
    WHERE d.approval_status = 'approved'
      AND d.indexing_status = 'indexed'
      AND s.review_status = 'approved'
      AND (
          requesting_role = 'admin'
          OR requesting_role = ANY(s.approved_roles)
      )
      AND (1.0 - (c.embedding <=> query_embedding)) >= v_threshold
    ORDER BY c.embedding <=> query_embedding ASC
    LIMIT v_limit;
END;
$$;

-- 6. RLS & Least-Privilege Grants
ALTER TABLE public.local_rag_chunks ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON public.local_rag_chunks FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON public.local_rag_chunks TO service_role;

REVOKE ALL ON FUNCTION public.local_rag_upsert_pgvector_chunks(uuid, integer, text, jsonb, uuid) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_delete_pgvector_chunks(uuid) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.match_local_rag_chunks(extensions.vector, text, integer, double precision) FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.local_rag_upsert_pgvector_chunks(uuid, integer, text, jsonb, uuid) TO service_role;
GRANT EXECUTE ON FUNCTION public.local_rag_delete_pgvector_chunks(uuid) TO service_role;
GRANT EXECUTE ON FUNCTION public.match_local_rag_chunks(extensions.vector, text, integer, double precision) TO service_role;
