-- Phase 3A (Final Security Corrections): Additive migration for local ChromaDB RAG approval,
-- section permissions, document supersession, stale-job protection, and immutable audit trail.
-- Preserves existing public.rag_documents, public.rag_chunks, and public.match_rag_chunks untouched.
-- Do NOT execute directly without explicit approval.

-- 1. Canonical local RAG document registry
CREATE TABLE IF NOT EXISTS public.local_rag_documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash_sha256 text NOT NULL UNIQUE
        CHECK (file_hash_sha256 ~ '^[0-9a-f]{64}$'),
    source_id text NOT NULL,
    filename text NOT NULL,
    legacy_chroma_document_id text,
    approval_status text NOT NULL DEFAULT 'pending'
        CHECK (approval_status IN ('pending', 'approved', 'rejected', 'revoked', 'superseded')),
    indexing_status text NOT NULL DEFAULT 'not_indexed'
        CHECK (indexing_status IN ('not_indexed', 'indexing', 'indexed', 'failed', 'stale')),
    approval_version integer NOT NULL DEFAULT 1 CHECK (approval_version >= 1),
    uploaded_by uuid REFERENCES public.profiles(id),
    approved_by uuid REFERENCES public.profiles(id),
    approved_at timestamptz,
    rejected_by uuid REFERENCES public.profiles(id),
    rejected_at timestamptz,
    revoked_by uuid REFERENCES public.profiles(id),
    revoked_at timestamptz,
    superseded_by_document_id uuid REFERENCES public.local_rag_documents(id) ON DELETE RESTRICT,
    superseded_at timestamptz,
    decision_reason text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT local_rag_documents_approved_fields_chk CHECK (
        approval_status <> 'approved' OR (
            approved_by IS NOT NULL
            AND approved_at IS NOT NULL
            AND rejected_by IS NULL
            AND rejected_at IS NULL
            AND revoked_by IS NULL
            AND revoked_at IS NULL
        )
    ),
    CONSTRAINT local_rag_documents_rejected_fields_chk CHECK (
        approval_status <> 'rejected' OR (
            rejected_by IS NOT NULL
            AND rejected_at IS NOT NULL
            AND approved_by IS NULL
            AND approved_at IS NULL
            AND revoked_by IS NULL
            AND revoked_at IS NULL
        )
    ),
    CONSTRAINT local_rag_documents_revoked_fields_chk CHECK (
        approval_status <> 'revoked' OR (
            revoked_by IS NOT NULL
            AND revoked_at IS NOT NULL
            AND approved_by IS NULL
            AND approved_at IS NULL
            AND rejected_by IS NULL
            AND rejected_at IS NULL
        )
    ),
    CONSTRAINT local_rag_documents_superseded_fields_chk CHECK (
        approval_status <> 'superseded' OR (
            superseded_by_document_id IS NOT NULL
            AND superseded_at IS NOT NULL
            AND approved_by IS NULL
            AND approved_at IS NULL
        )
    ),
    CONSTRAINT local_rag_documents_indexed_requires_approved_chk CHECK (
        indexing_status NOT IN ('indexing', 'indexed') OR approval_status = 'approved'
    )
);

CREATE INDEX IF NOT EXISTS idx_local_rag_documents_status
    ON public.local_rag_documents (approval_status, indexing_status);

CREATE INDEX IF NOT EXISTS idx_local_rag_documents_source_id
    ON public.local_rag_documents (source_id);

-- 2. Section-level classification, review, role permissions, and exclusion tracking
CREATE TABLE IF NOT EXISTS public.local_rag_document_sections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES public.local_rag_documents(id) ON DELETE CASCADE,
    section_index integer NOT NULL CHECK (section_index >= 0),
    section_title text NOT NULL,
    page_number integer CHECK (page_number IS NULL OR page_number > 0),
    section_sha256 text NOT NULL
        CHECK (section_sha256 ~ '^[0-9a-f]{64}$'),
    proposed_roles text[] NOT NULL DEFAULT '{}'
        CHECK (proposed_roles <@ ARRAY['admin', 'business_analyst', 'risk_analyst', 'customer']::text[]),
    approved_roles text[] NOT NULL DEFAULT '{}'
        CHECK (approved_roles <@ ARRAY['admin', 'business_analyst', 'risk_analyst', 'customer']::text[]),
    classification_method text,
    classification_confidence double precision
        CHECK (classification_confidence IS NULL OR (classification_confidence >= 0.0 AND classification_confidence <= 1.0)),
    classifier_quarantined boolean NOT NULL DEFAULT false,
    review_status text NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'excluded')),
    reviewed_by uuid REFERENCES public.profiles(id),
    reviewed_at timestamptz,
    quarantine_resolution_reason text,
    exclusion_reason text,
    chunk_ids_by_role jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, section_index),
    UNIQUE (id, document_id),
    CONSTRAINT local_rag_sections_approved_chk CHECK (
        review_status <> 'approved' OR (
            reviewed_by IS NOT NULL
            AND reviewed_at IS NOT NULL
            AND cardinality(approved_roles) > 0
        )
    ),
    CONSTRAINT local_rag_sections_quarantine_resolution_chk CHECK (
        NOT (classifier_quarantined AND review_status = 'approved') OR (
            quarantine_resolution_reason IS NOT NULL
            AND length(trim(quarantine_resolution_reason)) > 0
        )
    ),
    CONSTRAINT local_rag_sections_excluded_chk CHECK (
        review_status <> 'excluded' OR (
            reviewed_by IS NOT NULL
            AND reviewed_at IS NOT NULL
            AND exclusion_reason IS NOT NULL
            AND length(trim(exclusion_reason)) > 0
            AND cardinality(approved_roles) = 0
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_local_rag_sections_doc_review
    ON public.local_rag_document_sections (document_id, review_status);

-- 3. Immutable, append-only audit log (no cascading or mutating FKs)
CREATE TABLE IF NOT EXISTS public.local_rag_document_audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL,
    section_id uuid,
    action text NOT NULL CHECK (action IN (
        'registered',
        'classified',
        'section_approved',
        'section_excluded',
        'document_approved',
        'document_rejected',
        'document_revoked',
        'document_superseded',
        'approval_invalidated',
        'indexing_started',
        'indexing_verified',
        'indexing_failed',
        'reconciled'
    )),
    actor_id uuid,
    actor_role text,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_local_rag_audit_doc_created
    ON public.local_rag_document_audit_log (document_id, created_at DESC);

-- 4. Helper functions (pg_catalog first in search_path)
-- Trusted identity boundary: p_actor_id MUST be derived by the FastAPI backend from a
-- verified Supabase JWT (never from frontend-supplied actor IDs or roles) and is
-- independently re-verified against public.profiles here.
CREATE OR REPLACE FUNCTION public.local_rag_is_uuid_text(p_value text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
    SELECT p_value IS NOT NULL
       AND p_value ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';
$$;

CREATE OR REPLACE FUNCTION public.local_rag_is_sha256_hex(p_value text)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
    SELECT p_value IS NOT NULL
       AND p_value ~* '^[0-9a-f]{64}$';
$$;

CREATE OR REPLACE FUNCTION public.local_rag_assert_active_admin(p_actor_id uuid)
RETURNS text
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_profile_role text;
    v_status text;
BEGIN
    IF p_actor_id IS NULL THEN
        RAISE EXCEPTION 'Actor UUID is required.';
    END IF;

    SELECT p.role, p.account_status
    INTO v_profile_role, v_status
    FROM public.profiles AS p
    WHERE p.id = p_actor_id;

    IF NOT FOUND
       OR v_status IS NULL
       OR v_status IS DISTINCT FROM 'active'
       OR v_profile_role IS NULL
       OR v_profile_role NOT IN ('admin', 'Admin') THEN
        RAISE EXCEPTION 'Actor % is not an active Admin.', p_actor_id;
    END IF;

    RETURN 'admin';
END;
$$;

-- 5. Immutability, direct-write protection, timestamp, and lifecycle invalidation triggers
CREATE OR REPLACE FUNCTION public.local_rag_prevent_audit_mutation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
BEGIN
    RAISE EXCEPTION 'local_rag_document_audit_log is append-only; updates and deletes are prohibited.';
END;
$$;

DROP TRIGGER IF EXISTS trg_local_rag_audit_immutable ON public.local_rag_document_audit_log;
CREATE TRIGGER trg_local_rag_audit_immutable
    BEFORE UPDATE OR DELETE ON public.local_rag_document_audit_log
    FOR EACH ROW
    EXECUTE FUNCTION public.local_rag_prevent_audit_mutation();

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

DROP TRIGGER IF EXISTS trg_local_rag_documents_write_guard ON public.local_rag_documents;
CREATE TRIGGER trg_local_rag_documents_write_guard
    BEFORE INSERT OR UPDATE OR DELETE ON public.local_rag_documents
    FOR EACH ROW
    EXECUTE FUNCTION public.local_rag_guard_direct_writes();

DROP TRIGGER IF EXISTS trg_local_rag_sections_write_guard ON public.local_rag_document_sections;
CREATE TRIGGER trg_local_rag_sections_write_guard
    BEFORE INSERT OR UPDATE OR DELETE ON public.local_rag_document_sections
    FOR EACH ROW
    EXECUTE FUNCTION public.local_rag_guard_direct_writes();

CREATE OR REPLACE FUNCTION public.local_rag_touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
BEGIN
    NEW.updated_at := pg_catalog.now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_local_rag_documents_updated_at ON public.local_rag_documents;
CREATE TRIGGER trg_local_rag_documents_updated_at
    BEFORE UPDATE ON public.local_rag_documents
    FOR EACH ROW
    EXECUTE FUNCTION public.local_rag_touch_updated_at();

DROP TRIGGER IF EXISTS trg_local_rag_sections_updated_at ON public.local_rag_document_sections;
CREATE TRIGGER trg_local_rag_sections_updated_at
    BEFORE UPDATE ON public.local_rag_document_sections
    FOR EACH ROW
    EXECUTE FUNCTION public.local_rag_touch_updated_at();

CREATE OR REPLACE FUNCTION public.local_rag_invalidate_on_doc_change()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
BEGIN
    IF NEW.file_hash_sha256 IS DISTINCT FROM OLD.file_hash_sha256 THEN
        NEW.approval_status := 'pending';
        NEW.indexing_status := CASE
            WHEN OLD.indexing_status IN ('indexed', 'indexing') THEN 'stale'
            ELSE 'not_indexed'
        END;
        NEW.approved_by := NULL;
        NEW.approved_at := NULL;
        NEW.rejected_by := NULL;
        NEW.rejected_at := NULL;
        NEW.revoked_by := NULL;
        NEW.revoked_at := NULL;
        NEW.superseded_by_document_id := NULL;
        NEW.superseded_at := NULL;
        NEW.decision_reason := 'Content hash changed; re-review required.';
        NEW.approval_version := OLD.approval_version + 1;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_local_rag_doc_hash_invalidation ON public.local_rag_documents;
CREATE TRIGGER trg_local_rag_doc_hash_invalidation
    BEFORE UPDATE ON public.local_rag_documents
    FOR EACH ROW
    EXECUTE FUNCTION public.local_rag_invalidate_on_doc_change();

CREATE OR REPLACE FUNCTION public.local_rag_audit_doc_hash_invalidation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
BEGIN
    IF NEW.file_hash_sha256 IS DISTINCT FROM OLD.file_hash_sha256 THEN
        INSERT INTO public.local_rag_document_audit_log (
            document_id, action, actor_id, actor_role, details
        ) VALUES (
            NEW.id,
            'approval_invalidated',
            NULL,
            'system',
            pg_catalog.jsonb_build_object(
                'reason', 'file_hash_sha256_changed',
                'previous_approval_status', OLD.approval_status,
                'previous_indexing_status', OLD.indexing_status,
                'old_hash', OLD.file_hash_sha256,
                'new_hash', NEW.file_hash_sha256,
                'new_approval_version', NEW.approval_version
            )
        );
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_local_rag_doc_hash_invalidation_audit ON public.local_rag_documents;
CREATE TRIGGER trg_local_rag_doc_hash_invalidation_audit
    AFTER UPDATE ON public.local_rag_documents
    FOR EACH ROW
    EXECUTE FUNCTION public.local_rag_audit_doc_hash_invalidation();

-- Section lifecycle protection & invalidation for INSERT, UPDATE, and DELETE
CREATE OR REPLACE FUNCTION public.local_rag_invalidate_on_section_lifecycle()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_rpc_ctx text := pg_catalog.current_setting('local_rag.rpc_context', true);
    v_doc_id uuid;
    v_section_id uuid;
    v_section_index integer;
    v_reason text := NULL;
    v_new_version integer;
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF COALESCE(v_rpc_ctx, '') <> 'register' THEN
            v_reason := 'section_inserted_outside_register';
            v_doc_id := NEW.document_id;
            v_section_id := NEW.id;
            v_section_index := NEW.section_index;
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        IF COALESCE(v_rpc_ctx, '') <> 'register' THEN
            v_reason := 'section_deleted';
            v_doc_id := OLD.document_id;
            v_section_id := OLD.id;
            v_section_index := OLD.section_index;
        END IF;
    ELSIF TG_OP = 'UPDATE' THEN
        IF NEW.document_id IS DISTINCT FROM OLD.document_id THEN
            RAISE EXCEPTION 'Changing section document_id after creation is prohibited.';
        END IF;
        IF NEW.section_sha256 IS DISTINCT FROM OLD.section_sha256
           OR NEW.section_index IS DISTINCT FROM OLD.section_index
           OR NEW.section_title IS DISTINCT FROM OLD.section_title
           OR NEW.page_number IS DISTINCT FROM OLD.page_number THEN
            v_reason := 'section_structure_or_hash_changed';
        ELSIF (NEW.approved_roles IS DISTINCT FROM OLD.approved_roles
               OR NEW.review_status IS DISTINCT FROM OLD.review_status)
              AND COALESCE(v_rpc_ctx, '') <> 'approve' THEN
            v_reason := 'section_approval_modified_outside_approve_rpc';
        END IF;
        v_doc_id := NEW.document_id;
        v_section_id := NEW.id;
        v_section_index := NEW.section_index;
    END IF;

    IF v_reason IS NOT NULL AND v_doc_id IS NOT NULL THEN
        IF EXISTS (SELECT 1 FROM public.local_rag_documents AS d WHERE d.id = v_doc_id) THEN
            PERFORM pg_catalog.set_config('local_rag.rpc_context', 'invalidation_trigger', true);

            UPDATE public.local_rag_documents
            SET approval_status = 'pending',
                indexing_status = CASE
                    WHEN indexing_status IN ('indexed', 'indexing') THEN 'stale'
                    ELSE 'not_indexed'
                END,
                approved_by = NULL,
                approved_at = NULL,
                rejected_by = NULL,
                rejected_at = NULL,
                revoked_by = NULL,
                revoked_at = NULL,
                superseded_by_document_id = NULL,
                superseded_at = NULL,
                decision_reason = 'Section lifecycle change invalidated prior state; re-review required.',
                approval_version = approval_version + 1
            WHERE id = v_doc_id
            RETURNING approval_version INTO v_new_version;

            IF v_rpc_ctx IS NOT NULL AND v_rpc_ctx <> '' THEN
                PERFORM pg_catalog.set_config('local_rag.rpc_context', v_rpc_ctx, true);
            END IF;

            INSERT INTO public.local_rag_document_audit_log (
                document_id, section_id, action, actor_id, actor_role, details
            ) VALUES (
                v_doc_id,
                v_section_id,
                'approval_invalidated',
                NULL,
                'system',
                pg_catalog.jsonb_build_object(
                    'reason', v_reason,
                    'trigger_op', TG_OP,
                    'section_index', v_section_index,
                    'new_approval_version', v_new_version
                )
            );
        END IF;
    END IF;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_local_rag_section_lifecycle ON public.local_rag_document_sections;
CREATE TRIGGER trg_local_rag_section_lifecycle
    AFTER INSERT OR UPDATE OR DELETE ON public.local_rag_document_sections
    FOR EACH ROW
    EXECUTE FUNCTION public.local_rag_invalidate_on_section_lifecycle();

-- 6. Controlled RPCs
CREATE OR REPLACE FUNCTION public.local_rag_register_document(
    p_file_hash_sha256 text,
    p_source_id text,
    p_filename text,
    p_sections jsonb,
    p_actor_id uuid DEFAULT NULL,
    p_legacy_chroma_document_id text DEFAULT NULL
)
RETURNS public.local_rag_documents
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_actor_role text := 'system';
    v_file_hash text;
    v_doc public.local_rag_documents;
    v_prev_doc public.local_rag_documents;
    v_sec jsonb;
    v_sec_index integer;
    v_sec_title text;
    v_sec_page integer;
    v_sec_hash text;
    v_proposed text[];
    v_method text;
    v_conf double precision;
    v_quarantined boolean;
    v_is_update boolean := false;
    v_superseded_ids uuid[] := '{}';
    v_sup_id uuid;
BEGIN
    IF NOT public.local_rag_is_sha256_hex(p_file_hash_sha256) THEN
        RAISE EXCEPTION 'file_hash_sha256 must be a valid 64-character hexadecimal SHA-256 digest.';
    END IF;
    v_file_hash := pg_catalog.lower(pg_catalog.btrim(p_file_hash_sha256));

    IF NULLIF(pg_catalog.btrim(COALESCE(p_source_id, '')), '') IS NULL
       OR NULLIF(pg_catalog.btrim(COALESCE(p_filename, '')), '') IS NULL THEN
        RAISE EXCEPTION 'file_hash_sha256, source_id, and filename are required.';
    END IF;

    IF p_sections IS NULL
       OR pg_catalog.jsonb_typeof(p_sections) <> 'array'
       OR pg_catalog.jsonb_array_length(p_sections) = 0 THEN
        RAISE EXCEPTION 'p_sections must be a non-empty JSON array.';
    END IF;

    IF p_actor_id IS NOT NULL THEN
        v_actor_role := public.local_rag_assert_active_admin(p_actor_id);
    END IF;

    -- Serialize concurrent registrations for the same logical source_id
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext(p_source_id));
    PERFORM pg_catalog.set_config('local_rag.rpc_context', 'register', true);

    -- Lock any existing versions of this source_id deterministically
    FOR v_prev_doc IN
        SELECT * FROM public.local_rag_documents AS d
        WHERE d.source_id = p_source_id
          AND d.file_hash_sha256 <> v_file_hash
          AND d.approval_status IN ('pending', 'approved')
        ORDER BY d.created_at, d.id
        FOR UPDATE
    LOOP
        v_superseded_ids := pg_catalog.array_append(v_superseded_ids, v_prev_doc.id);
    END LOOP;

    SELECT * INTO v_doc
    FROM public.local_rag_documents AS d
    WHERE d.file_hash_sha256 = v_file_hash
    FOR UPDATE;

    IF FOUND THEN
        IF v_doc.source_id IS DISTINCT FROM p_source_id THEN
            RAISE EXCEPTION 'file_hash_sha256 % is already registered under a different source_id.', v_file_hash;
        END IF;

        v_is_update := true;
        UPDATE public.local_rag_documents
        SET filename = p_filename,
            legacy_chroma_document_id = COALESCE(p_legacy_chroma_document_id, legacy_chroma_document_id),
            approval_status = 'pending',
            indexing_status = CASE
                WHEN indexing_status IN ('indexed', 'indexing') THEN 'stale'
                ELSE 'not_indexed'
            END,
            approved_by = NULL,
            approved_at = NULL,
            rejected_by = NULL,
            rejected_at = NULL,
            revoked_by = NULL,
            revoked_at = NULL,
            superseded_by_document_id = NULL,
            superseded_at = NULL,
            decision_reason = 'Re-registered/re-classified; pending Admin review.',
            approval_version = approval_version + 1
        WHERE id = v_doc.id
        RETURNING * INTO v_doc;

        DELETE FROM public.local_rag_document_sections
        WHERE document_id = v_doc.id;
    ELSE
        INSERT INTO public.local_rag_documents (
            file_hash_sha256,
            source_id,
            filename,
            legacy_chroma_document_id,
            uploaded_by
        ) VALUES (
            v_file_hash,
            p_source_id,
            p_filename,
            p_legacy_chroma_document_id,
            p_actor_id
        )
        RETURNING * INTO v_doc;
    END IF;

    -- Atomically supersede prior active/pending versions for the same source_id
    FOREACH v_sup_id IN ARRAY v_superseded_ids
    LOOP
        UPDATE public.local_rag_documents
        SET approval_status = 'superseded',
            indexing_status = CASE
                WHEN indexing_status IN ('indexed', 'indexing') THEN 'stale'
                ELSE indexing_status
            END,
            approved_by = NULL,
            approved_at = NULL,
            superseded_by_document_id = v_doc.id,
            superseded_at = pg_catalog.now(),
            decision_reason = 'Superseded by newer document version.',
            approval_version = approval_version + 1
        WHERE id = v_sup_id;

        INSERT INTO public.local_rag_document_audit_log (
            document_id, action, actor_id, actor_role, details
        ) VALUES (
            v_sup_id,
            'document_superseded',
            p_actor_id,
            v_actor_role,
            pg_catalog.jsonb_build_object(
                'source_id', p_source_id,
                'superseded_by_document_id', v_doc.id,
                'new_file_hash_sha256', v_file_hash
            )
        );
    END LOOP;

    FOR v_sec IN
        SELECT s.item FROM pg_catalog.jsonb_array_elements(p_sections) AS s(item)
    LOOP
        IF pg_catalog.jsonb_typeof(v_sec) <> 'object' THEN
            RAISE EXCEPTION 'Each section entry must be a JSON object.';
        END IF;
        IF pg_catalog.jsonb_typeof(v_sec->'section_index') <> 'number'
           OR COALESCE(v_sec->>'section_index', '') !~ '^[0-9]+$' THEN
            RAISE EXCEPTION 'Invalid section_index.';
        END IF;
        v_sec_index := (v_sec->>'section_index')::integer;
        v_sec_title := NULLIF(pg_catalog.btrim(COALESCE(v_sec->>'section_title', '')), '');
        v_sec_hash := NULLIF(pg_catalog.btrim(COALESCE(v_sec->>'section_sha256', '')), '');
        IF v_sec_title IS NULL OR NOT public.local_rag_is_sha256_hex(v_sec_hash) THEN
            RAISE EXCEPTION 'section_title and valid 64-char hex section_sha256 are required for section %.', v_sec_index;
        END IF;
        v_sec_hash := pg_catalog.lower(v_sec_hash);

        v_sec_page := NULL;
        IF v_sec ? 'page_number' AND v_sec->'page_number' IS NOT NULL AND pg_catalog.jsonb_typeof(v_sec->'page_number') <> 'null' THEN
            IF pg_catalog.jsonb_typeof(v_sec->'page_number') <> 'number'
               OR COALESCE(v_sec->>'page_number', '') !~ '^[1-9][0-9]*$' THEN
                RAISE EXCEPTION 'Invalid page_number for section %.', v_sec_index;
            END IF;
            v_sec_page := (v_sec->>'page_number')::integer;
        END IF;

        IF pg_catalog.jsonb_typeof(COALESCE(v_sec->'proposed_roles', '[]'::jsonb)) <> 'array' THEN
            RAISE EXCEPTION 'proposed_roles must be a JSON array for section %.', v_sec_index;
        END IF;
        IF EXISTS (
            SELECT 1 FROM pg_catalog.jsonb_array_elements(COALESCE(v_sec->'proposed_roles', '[]'::jsonb)) AS pr(val)
            WHERE pg_catalog.jsonb_typeof(pr.val) <> 'string'
        ) THEN
            RAISE EXCEPTION 'proposed_roles must contain only strings for section %.', v_sec_index;
        END IF;

        SELECT COALESCE(pg_catalog.array_agg(DISTINCT pr.role_text), '{}')
        INTO v_proposed
        FROM pg_catalog.jsonb_array_elements_text(COALESCE(v_sec->'proposed_roles', '[]'::jsonb)) AS pr(role_text);

        IF NOT (v_proposed <@ ARRAY['admin', 'business_analyst', 'risk_analyst', 'customer']::text[]) THEN
            RAISE EXCEPTION 'proposed_roles contains invalid role(s) for section %.', v_sec_index;
        END IF;

        v_method := v_sec->>'classification_method';
        v_conf := NULL;
        IF v_sec ? 'classification_confidence' AND v_sec->'classification_confidence' IS NOT NULL AND pg_catalog.jsonb_typeof(v_sec->'classification_confidence') <> 'null' THEN
            IF pg_catalog.jsonb_typeof(v_sec->'classification_confidence') <> 'number' THEN
                RAISE EXCEPTION 'classification_confidence must be numeric for section %.', v_sec_index;
            END IF;
            v_conf := (v_sec->>'classification_confidence')::double precision;
        END IF;

        v_quarantined := false;
        IF v_sec ? 'classifier_quarantined' THEN
            IF pg_catalog.jsonb_typeof(v_sec->'classifier_quarantined') <> 'boolean' THEN
                RAISE EXCEPTION 'classifier_quarantined must be a JSON boolean for section %.', v_sec_index;
            END IF;
            v_quarantined := (v_sec->>'classifier_quarantined')::boolean;
        END IF;

        INSERT INTO public.local_rag_document_sections (
            document_id,
            section_index,
            section_title,
            page_number,
            section_sha256,
            proposed_roles,
            classification_method,
            classification_confidence,
            classifier_quarantined
        ) VALUES (
            v_doc.id,
            v_sec_index,
            v_sec_title,
            v_sec_page,
            v_sec_hash,
            v_proposed,
            v_method,
            v_conf,
            v_quarantined
        );
    END LOOP;

    INSERT INTO public.local_rag_document_audit_log (
        document_id, action, actor_id, actor_role, details
    ) VALUES (
        v_doc.id,
        CASE WHEN v_is_update THEN 'approval_invalidated' ELSE 'registered' END,
        p_actor_id,
        v_actor_role,
        pg_catalog.jsonb_build_object(
            'source_id', p_source_id,
            'filename', p_filename,
            'section_count', pg_catalog.jsonb_array_length(p_sections),
            'superseded_document_ids', v_superseded_ids,
            'approval_version', v_doc.approval_version
        )
    );

    INSERT INTO public.local_rag_document_audit_log (
        document_id, action, actor_id, actor_role, details
    ) VALUES (
        v_doc.id,
        'classified',
        p_actor_id,
        v_actor_role,
        pg_catalog.jsonb_build_object('section_count', pg_catalog.jsonb_array_length(p_sections))
    );

    RETURN v_doc;
END;
$$;

CREATE OR REPLACE FUNCTION public.local_rag_approve_document(
    p_document_id uuid,
    p_actor_id uuid,
    p_section_decisions jsonb,
    p_decision_reason text DEFAULT NULL
)
RETURNS public.local_rag_documents
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_actor_role text;
    v_doc public.local_rag_documents;
    v_section public.local_rag_document_sections;
    v_decision_count integer;
    v_distinct_count integer;
    v_unexpected_count integer;
    v_section_id_text text;
    v_decision jsonb;
    v_status text;
    v_roles text[];
    v_reason text;
    v_quarantine_reason text;
    v_approved_count integer := 0;
BEGIN
    v_actor_role := public.local_rag_assert_active_admin(p_actor_id);

    SELECT * INTO v_doc
    FROM public.local_rag_documents AS d
    WHERE d.id = p_document_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Document % not found.', p_document_id;
    END IF;

    -- Document state machine:
    --   pending -> approved | rejected (or superseded via re-registration)
    --   approved -> revoked (or superseded via re-registration)
    --   rejected, revoked, superseded are terminal; use local_rag_register_document for a new pending review.
    IF v_doc.approval_status IS DISTINCT FROM 'pending' THEN
        RAISE EXCEPTION 'Only pending documents can be approved (current status: %).', v_doc.approval_status;
    END IF;

    IF p_section_decisions IS NULL
       OR pg_catalog.jsonb_typeof(p_section_decisions) <> 'array'
       OR pg_catalog.jsonb_array_length(p_section_decisions) = 0 THEN
        RAISE EXCEPTION 'p_section_decisions must be a non-empty JSON array.';
    END IF;

    FOR v_decision IN
        SELECT d.item
        FROM pg_catalog.jsonb_array_elements(p_section_decisions) AS d(item)
    LOOP
        IF pg_catalog.jsonb_typeof(v_decision) <> 'object' THEN
            RAISE EXCEPTION 'Each section decision must be a JSON object.';
        END IF;
        v_section_id_text := v_decision->>'section_id';
        IF NOT public.local_rag_is_uuid_text(v_section_id_text) THEN
            RAISE EXCEPTION 'Malformed section_id in p_section_decisions: %', COALESCE(v_section_id_text, 'NULL');
        END IF;
    END LOOP;

    SELECT COUNT(*), COUNT(DISTINCT (d.item->>'section_id')::uuid)
    INTO v_decision_count, v_distinct_count
    FROM pg_catalog.jsonb_array_elements(p_section_decisions) AS d(item);

    IF v_decision_count <> v_distinct_count THEN
        RAISE EXCEPTION 'Duplicate section_id entries in p_section_decisions are not allowed.';
    END IF;

    SELECT COUNT(*)
    INTO v_unexpected_count
    FROM pg_catalog.jsonb_array_elements(p_section_decisions) AS d(item)
    WHERE NOT EXISTS (
        SELECT 1
        FROM public.local_rag_document_sections AS s
        WHERE s.id = (d.item->>'section_id')::uuid
          AND s.document_id = p_document_id
    );

    IF v_unexpected_count > 0 THEN
        RAISE EXCEPTION 'p_section_decisions contains % section_id(s) not belonging to document %.', v_unexpected_count, p_document_id;
    END IF;

    PERFORM pg_catalog.set_config('local_rag.rpc_context', 'approve', true);

    FOR v_section IN
        SELECT * FROM public.local_rag_document_sections AS s
        WHERE s.document_id = p_document_id
        ORDER BY s.section_index
        FOR UPDATE
    LOOP
        SELECT d.item INTO v_decision
        FROM pg_catalog.jsonb_array_elements(p_section_decisions) AS d(item)
        WHERE (d.item->>'section_id')::uuid = v_section.id;

        IF v_decision IS NULL THEN
            RAISE EXCEPTION 'Missing explicit Admin decision for section % (%).', v_section.id, v_section.section_index;
        END IF;

        v_status := v_decision->>'review_status';
        IF v_status = 'approved' THEN
            IF pg_catalog.jsonb_typeof(v_decision->'approved_roles') <> 'array' THEN
                RAISE EXCEPTION 'approved_roles must be a JSON array for section %.', v_section.id;
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_catalog.jsonb_array_elements(v_decision->'approved_roles') AS r(role_val)
                WHERE pg_catalog.jsonb_typeof(r.role_val) <> 'string'
            ) THEN
                RAISE EXCEPTION 'approved_roles must contain only strings for section %.', v_section.id;
            END IF;

            SELECT COALESCE(pg_catalog.array_agg(DISTINCT r.role_text), '{}')
            INTO v_roles
            FROM pg_catalog.jsonb_array_elements_text(v_decision->'approved_roles') AS r(role_text);

            IF pg_catalog.cardinality(v_roles) = 0
               OR pg_catalog.cardinality(v_roles) <> pg_catalog.jsonb_array_length(v_decision->'approved_roles')
               OR NOT (v_roles <@ ARRAY['admin', 'business_analyst', 'risk_analyst', 'customer']::text[]) THEN
                RAISE EXCEPTION 'Invalid or duplicate approved_roles for section %.', v_section.id;
            END IF;

            v_quarantine_reason := NULLIF(pg_catalog.btrim(COALESCE(v_decision->>'quarantine_resolution_reason', '')), '');
            IF v_section.classifier_quarantined AND v_quarantine_reason IS NULL THEN
                RAISE EXCEPTION 'Quarantined section % requires a non-empty quarantine_resolution_reason before approval.', v_section.id;
            END IF;

            UPDATE public.local_rag_document_sections
            SET review_status = 'approved',
                approved_roles = v_roles,
                reviewed_by = p_actor_id,
                reviewed_at = pg_catalog.now(),
                quarantine_resolution_reason = CASE WHEN v_section.classifier_quarantined THEN v_quarantine_reason ELSE NULL END,
                exclusion_reason = NULL,
                chunk_ids_by_role = '{}'::jsonb
            WHERE id = v_section.id;

            v_approved_count := v_approved_count + 1;

            INSERT INTO public.local_rag_document_audit_log (
                document_id, section_id, action, actor_id, actor_role, details
            ) VALUES (
                p_document_id, v_section.id, 'section_approved', p_actor_id, v_actor_role,
                pg_catalog.jsonb_build_object(
                    'approved_roles', v_roles,
                    'classifier_quarantined', v_section.classifier_quarantined,
                    'quarantine_resolution_reason', v_quarantine_reason
                )
            );
        ELSIF v_status = 'excluded' THEN
            v_reason := NULLIF(pg_catalog.btrim(COALESCE(v_decision->>'exclusion_reason', '')), '');
            IF v_reason IS NULL THEN
                RAISE EXCEPTION 'Exclusion reason is required for excluded section %.', v_section.id;
            END IF;

            UPDATE public.local_rag_document_sections
            SET review_status = 'excluded',
                approved_roles = '{}',
                reviewed_by = p_actor_id,
                reviewed_at = pg_catalog.now(),
                quarantine_resolution_reason = NULL,
                exclusion_reason = v_reason,
                chunk_ids_by_role = '{}'::jsonb
            WHERE id = v_section.id;

            INSERT INTO public.local_rag_document_audit_log (
                document_id, section_id, action, actor_id, actor_role, details
            ) VALUES (
                p_document_id, v_section.id, 'section_excluded', p_actor_id, v_actor_role,
                pg_catalog.jsonb_build_object(
                    'exclusion_reason', v_reason,
                    'classifier_quarantined', v_section.classifier_quarantined
                )
            );
        ELSE
            RAISE EXCEPTION 'Invalid review_status % for section %.', COALESCE(v_status, 'NULL'), v_section.id;
        END IF;
    END LOOP;

    IF v_approved_count = 0 THEN
        RAISE EXCEPTION 'At least one section must be approved to approve document %.', p_document_id;
    END IF;

    UPDATE public.local_rag_documents
    SET approval_status = 'approved',
        indexing_status = 'not_indexed',
        approved_by = p_actor_id,
        approved_at = pg_catalog.now(),
        rejected_by = NULL,
        rejected_at = NULL,
        revoked_by = NULL,
        revoked_at = NULL,
        superseded_by_document_id = NULL,
        superseded_at = NULL,
        decision_reason = p_decision_reason,
        approval_version = v_doc.approval_version + 1
    WHERE id = p_document_id
    RETURNING * INTO v_doc;

    INSERT INTO public.local_rag_document_audit_log (
        document_id, action, actor_id, actor_role, details
    ) VALUES (
        p_document_id, 'document_approved', p_actor_id, v_actor_role,
        pg_catalog.jsonb_build_object(
            'approved_sections', v_approved_count,
            'reason', p_decision_reason,
            'approval_version', v_doc.approval_version
        )
    );

    RETURN v_doc;
END;
$$;

CREATE OR REPLACE FUNCTION public.local_rag_reject_or_revoke_document(
    p_document_id uuid,
    p_action text,
    p_actor_id uuid,
    p_decision_reason text
)
RETURNS public.local_rag_documents
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_actor_role text;
    v_doc public.local_rag_documents;
    v_previous_status text;
    v_reason text := NULLIF(pg_catalog.btrim(COALESCE(p_decision_reason, '')), '');
BEGIN
    v_actor_role := public.local_rag_assert_active_admin(p_actor_id);

    IF p_action IS NULL OR p_action NOT IN ('rejected', 'revoked') THEN
        RAISE EXCEPTION 'Action must be rejected or revoked.';
    END IF;
    IF v_reason IS NULL THEN
        RAISE EXCEPTION 'Decision reason is required for %.', p_action;
    END IF;

    SELECT * INTO v_doc
    FROM public.local_rag_documents AS d
    WHERE d.id = p_document_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Document % not found.', p_document_id;
    END IF;

    IF p_action = 'rejected' AND v_doc.approval_status IS DISTINCT FROM 'pending' THEN
        RAISE EXCEPTION 'Only pending documents can be rejected (current status: %).', v_doc.approval_status;
    END IF;
    IF p_action = 'revoked' AND v_doc.approval_status IS DISTINCT FROM 'approved' THEN
        RAISE EXCEPTION 'Only approved documents can be revoked (current status: %).', v_doc.approval_status;
    END IF;

    v_previous_status := v_doc.approval_status;
    PERFORM pg_catalog.set_config('local_rag.rpc_context', 'reject_revoke', true);

    IF p_action = 'rejected' THEN
        UPDATE public.local_rag_documents
        SET approval_status = 'rejected',
            indexing_status = CASE WHEN v_doc.indexing_status IN ('indexed', 'indexing') THEN 'stale' ELSE v_doc.indexing_status END,
            rejected_by = p_actor_id,
            rejected_at = pg_catalog.now(),
            approved_by = NULL,
            approved_at = NULL,
            revoked_by = NULL,
            revoked_at = NULL,
            decision_reason = v_reason,
            approval_version = v_doc.approval_version + 1
        WHERE id = p_document_id
        RETURNING * INTO v_doc;
    ELSE
        UPDATE public.local_rag_documents
        SET approval_status = 'revoked',
            indexing_status = CASE WHEN v_doc.indexing_status IN ('indexed', 'indexing') THEN 'stale' ELSE v_doc.indexing_status END,
            revoked_by = p_actor_id,
            revoked_at = pg_catalog.now(),
            approved_by = NULL,
            approved_at = NULL,
            rejected_by = NULL,
            rejected_at = NULL,
            decision_reason = v_reason,
            approval_version = v_doc.approval_version + 1
        WHERE id = p_document_id
        RETURNING * INTO v_doc;
    END IF;

    INSERT INTO public.local_rag_document_audit_log (
        document_id, action, actor_id, actor_role, details
    ) VALUES (
        p_document_id,
        CASE WHEN p_action = 'rejected' THEN 'document_rejected' ELSE 'document_revoked' END,
        p_actor_id,
        v_actor_role,
        pg_catalog.jsonb_build_object(
            'reason', v_reason,
            'previous_approval_status', v_previous_status,
            'new_approval_version', v_doc.approval_version
        )
    );

    RETURN v_doc;
END;
$$;

CREATE OR REPLACE FUNCTION public.local_rag_start_indexing(
    p_document_id uuid,
    p_expected_approval_version integer,
    p_expected_file_hash text,
    p_actor_id uuid DEFAULT NULL
)
RETURNS public.local_rag_documents
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_actor_role text := 'system';
    v_expected_file_hash text;
    v_doc public.local_rag_documents;
BEGIN
    IF p_expected_approval_version IS NULL OR p_expected_approval_version < 1 THEN
        RAISE EXCEPTION 'p_expected_approval_version must be a non-null integer >= 1.';
    END IF;
    IF NOT public.local_rag_is_sha256_hex(p_expected_file_hash) THEN
        RAISE EXCEPTION 'p_expected_file_hash must be a valid 64-character hexadecimal SHA-256 digest.';
    END IF;
    v_expected_file_hash := pg_catalog.lower(pg_catalog.btrim(p_expected_file_hash));

    IF p_actor_id IS NOT NULL THEN
        v_actor_role := public.local_rag_assert_active_admin(p_actor_id);
    END IF;

    SELECT * INTO v_doc
    FROM public.local_rag_documents AS d
    WHERE d.id = p_document_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Document % not found.', p_document_id;
    END IF;

    IF v_doc.approval_status <> 'approved' THEN
        RAISE EXCEPTION 'Cannot start indexing document % with approval_status %.', p_document_id, v_doc.approval_status;
    END IF;

    IF v_doc.approval_version IS DISTINCT FROM p_expected_approval_version
       OR v_doc.file_hash_sha256 IS DISTINCT FROM v_expected_file_hash THEN
        RAISE EXCEPTION 'Stale indexing job rejected: expected version %/hash %, current version %/hash %.',
            p_expected_approval_version, v_expected_file_hash, v_doc.approval_version, v_doc.file_hash_sha256;
    END IF;

    IF v_doc.indexing_status NOT IN ('not_indexed', 'failed', 'stale') THEN
        RAISE EXCEPTION 'Cannot start indexing: document % already has indexing_status %.', p_document_id, v_doc.indexing_status;
    END IF;

    PERFORM pg_catalog.set_config('local_rag.rpc_context', 'start_indexing', true);

    UPDATE public.local_rag_documents
    SET indexing_status = 'indexing'
    WHERE id = p_document_id
    RETURNING * INTO v_doc;

    INSERT INTO public.local_rag_document_audit_log (
        document_id, action, actor_id, actor_role, details
    ) VALUES (
        p_document_id,
        'indexing_started',
        p_actor_id,
        v_actor_role,
        pg_catalog.jsonb_build_object(
            'approval_version', v_doc.approval_version,
            'file_hash_sha256', v_doc.file_hash_sha256
        )
    );

    RETURN v_doc;
END;
$$;

-- FastAPI Responsibility: Before calling local_rag_complete_indexing, the FastAPI backend MUST
-- query ChromaDB directly to verify actual chunk existence, canonical document_id UUID ownership,
-- section_id ownership, and role-collection parity, then supply attested p_section_chunks and p_details.
-- SQL validates structural completeness, role parity, chunk ID uniqueness, and stale-job binding.
CREATE OR REPLACE FUNCTION public.local_rag_complete_indexing(
    p_document_id uuid,
    p_expected_approval_version integer,
    p_expected_file_hash text,
    p_section_chunks jsonb,
    p_details jsonb,
    p_actor_id uuid DEFAULT NULL
)
RETURNS public.local_rag_documents
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_actor_role text := 'system';
    v_expected_file_hash text;
    v_doc public.local_rag_documents;
    v_section public.local_rag_document_sections;
    v_item jsonb;
    v_section_id_text text;
    v_chunks_obj jsonb;
    v_role_keys text[];
    v_expected_roles text[];
    v_verified_collections text[];
    v_expected_collections text[];
    v_vc_count integer;
    v_vc_distinct_count integer;
    v_role_key text;
    v_role_chunks jsonb;
    v_chunk_elem jsonb;
    v_chunk_text text;
    v_role_chunk_count integer;
    v_role_distinct_count integer;
    v_item_count integer;
    v_distinct_item_count integer;
    v_unexpected_item_count integer;
    v_approved_section_count integer;
    v_total_verified_chunks integer := 0;
    v_claimed_verified_chunks integer;
    v_verified_at timestamptz;
BEGIN
    IF p_expected_approval_version IS NULL OR p_expected_approval_version < 1 THEN
        RAISE EXCEPTION 'p_expected_approval_version must be a non-null integer >= 1.';
    END IF;
    IF NOT public.local_rag_is_sha256_hex(p_expected_file_hash) THEN
        RAISE EXCEPTION 'p_expected_file_hash must be a valid 64-character hexadecimal SHA-256 digest.';
    END IF;
    v_expected_file_hash := pg_catalog.lower(pg_catalog.btrim(p_expected_file_hash));

    IF p_actor_id IS NOT NULL THEN
        v_actor_role := public.local_rag_assert_active_admin(p_actor_id);
    END IF;

    SELECT * INTO v_doc
    FROM public.local_rag_documents AS d
    WHERE d.id = p_document_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Document % not found.', p_document_id;
    END IF;

    IF v_doc.approval_status <> 'approved' THEN
        RAISE EXCEPTION 'Cannot complete indexing while approval_status is %.', v_doc.approval_status;
    END IF;

    IF v_doc.approval_version IS DISTINCT FROM p_expected_approval_version
       OR v_doc.file_hash_sha256 IS DISTINCT FROM v_expected_file_hash THEN
        RAISE EXCEPTION 'Stale indexing completion rejected: expected version %/hash %, current version %/hash %.',
            p_expected_approval_version, v_expected_file_hash, v_doc.approval_version, v_doc.file_hash_sha256;
    END IF;

    IF v_doc.indexing_status <> 'indexing' THEN
        RAISE EXCEPTION 'Document % must be in indexing status before completion (current: %).', p_document_id, v_doc.indexing_status;
    END IF;

    IF p_section_chunks IS NULL
       OR pg_catalog.jsonb_typeof(p_section_chunks) <> 'array'
       OR pg_catalog.jsonb_array_length(p_section_chunks) = 0 THEN
        RAISE EXCEPTION 'Backend-attested verification requires non-empty p_section_chunks array.';
    END IF;

    IF p_details IS NULL OR pg_catalog.jsonb_typeof(p_details) <> 'object' THEN
        RAISE EXCEPTION 'Backend-attested verification requires p_details JSON object.';
    END IF;

    IF pg_catalog.jsonb_typeof(p_details->'verified_chunk_count') <> 'number'
       OR COALESCE(p_details->>'verified_chunk_count', '') !~ '^[0-9]+$' THEN
        RAISE EXCEPTION 'p_details must include non-negative integer verified_chunk_count.';
    END IF;
    v_claimed_verified_chunks := (p_details->>'verified_chunk_count')::integer;

    IF pg_catalog.jsonb_typeof(p_details->'verified_at') <> 'string'
       OR NULLIF(pg_catalog.btrim(p_details->>'verified_at'), '') IS NULL THEN
        RAISE EXCEPTION 'p_details must include non-empty ISO timestamp verified_at.';
    END IF;

    BEGIN
        v_verified_at := (p_details->>'verified_at')::timestamptz;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'p_details.verified_at is not a valid timestamptz: %', p_details->>'verified_at';
    END;

    IF pg_catalog.jsonb_typeof(p_details->'verification_source') <> 'string'
       OR NULLIF(pg_catalog.btrim(p_details->>'verification_source'), '') IS NULL THEN
        RAISE EXCEPTION 'p_details must include non-empty verification_source.';
    END IF;

    IF pg_catalog.jsonb_typeof(p_details->'verified_collections') <> 'array'
       OR pg_catalog.jsonb_array_length(p_details->'verified_collections') = 0 THEN
        RAISE EXCEPTION 'p_details must include non-empty verified_collections array from ChromaDB verification.';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements(p_details->'verified_collections') AS vc(val)
        WHERE pg_catalog.jsonb_typeof(vc.val) <> 'string'
           OR pg_catalog.length(pg_catalog.btrim(vc.val #>> '{}')) = 0
    ) THEN
        RAISE EXCEPTION 'p_details.verified_collections must contain only non-empty strings.';
    END IF;

    SELECT COUNT(*), COUNT(DISTINCT pg_catalog.btrim(vc.col_name))
    INTO v_vc_count, v_vc_distinct_count
    FROM pg_catalog.jsonb_array_elements_text(p_details->'verified_collections') AS vc(col_name);

    IF v_vc_count <> v_vc_distinct_count THEN
        RAISE EXCEPTION 'p_details.verified_collections must not contain duplicate collection names.';
    END IF;

    SELECT COALESCE(pg_catalog.array_agg(DISTINCT pg_catalog.btrim(vc.col_name) ORDER BY pg_catalog.btrim(vc.col_name)), '{}')
    INTO v_verified_collections
    FROM pg_catalog.jsonb_array_elements_text(p_details->'verified_collections') AS vc(col_name);

    SELECT COALESCE(pg_catalog.array_agg(DISTINCT r.role_name ORDER BY r.role_name), '{}')
    INTO v_expected_collections
    FROM public.local_rag_document_sections AS s,
         pg_catalog.unnest(s.approved_roles) AS r(role_name)
    WHERE s.document_id = p_document_id
      AND s.review_status = 'approved';

    IF v_verified_collections IS DISTINCT FROM v_expected_collections THEN
        RAISE EXCEPTION 'p_details.verified_collections (%) does not match expected approved role collections (%).',
            v_verified_collections, v_expected_collections;
    END IF;

    FOR v_item IN
        SELECT sc.item
        FROM pg_catalog.jsonb_array_elements(p_section_chunks) AS sc(item)
    LOOP
        IF pg_catalog.jsonb_typeof(v_item) <> 'object' THEN
            RAISE EXCEPTION 'Each p_section_chunks entry must be a JSON object.';
        END IF;
        v_section_id_text := v_item->>'section_id';
        IF NOT public.local_rag_is_uuid_text(v_section_id_text) THEN
            RAISE EXCEPTION 'Malformed section_id in p_section_chunks: %', COALESCE(v_section_id_text, 'NULL');
        END IF;
        IF pg_catalog.jsonb_typeof(v_item->'chunk_ids_by_role') <> 'object' THEN
            RAISE EXCEPTION 'chunk_ids_by_role must be a JSON object for section %.', v_section_id_text;
        END IF;
    END LOOP;

    SELECT COUNT(*), COUNT(DISTINCT (sc.item->>'section_id')::uuid)
    INTO v_item_count, v_distinct_item_count
    FROM pg_catalog.jsonb_array_elements(p_section_chunks) AS sc(item);

    IF v_item_count <> v_distinct_item_count THEN
        RAISE EXCEPTION 'Duplicate section_id entries in p_section_chunks are not allowed.';
    END IF;

    SELECT COUNT(*)
    INTO v_approved_section_count
    FROM public.local_rag_document_sections AS s
    WHERE s.document_id = p_document_id
      AND s.review_status = 'approved';

    IF v_approved_section_count = 0 THEN
        RAISE EXCEPTION 'Document % has no approved sections.', p_document_id;
    END IF;

    SELECT COUNT(*)
    INTO v_unexpected_item_count
    FROM pg_catalog.jsonb_array_elements(p_section_chunks) AS sc(item)
    WHERE NOT EXISTS (
        SELECT 1
        FROM public.local_rag_document_sections AS s
        WHERE s.id = (sc.item->>'section_id')::uuid
          AND s.document_id = p_document_id
          AND s.review_status = 'approved'
    );

    IF v_unexpected_item_count > 0 THEN
        RAISE EXCEPTION 'p_section_chunks references % non-approved or foreign section(s).', v_unexpected_item_count;
    END IF;

    IF v_distinct_item_count <> v_approved_section_count THEN
        RAISE EXCEPTION 'Incomplete approved-section coverage: expected %, got %.', v_approved_section_count, v_distinct_item_count;
    END IF;

    PERFORM pg_catalog.set_config('local_rag.rpc_context', 'complete_indexing', true);

    FOR v_section IN
        SELECT * FROM public.local_rag_document_sections AS s
        WHERE s.document_id = p_document_id
          AND s.review_status = 'approved'
        ORDER BY s.section_index
        FOR UPDATE
    LOOP
        SELECT sc.item->'chunk_ids_by_role'
        INTO v_chunks_obj
        FROM pg_catalog.jsonb_array_elements(p_section_chunks) AS sc(item)
        WHERE (sc.item->>'section_id')::uuid = v_section.id;

        SELECT COALESCE(pg_catalog.array_agg(k.key_name ORDER BY k.key_name), '{}')
        INTO v_role_keys
        FROM pg_catalog.jsonb_object_keys(v_chunks_obj) AS k(key_name);

        SELECT COALESCE(pg_catalog.array_agg(ar.role_name ORDER BY ar.role_name), '{}')
        INTO v_expected_roles
        FROM pg_catalog.unnest(v_section.approved_roles) AS ar(role_name);

        IF v_expected_roles IS DISTINCT FROM v_role_keys THEN
            RAISE EXCEPTION 'chunk_ids_by_role keys do not match approved_roles for section %.', v_section.id;
        END IF;

        FOREACH v_role_key IN ARRAY v_role_keys
        LOOP
            v_role_chunks := v_chunks_obj->v_role_key;
            IF pg_catalog.jsonb_typeof(v_role_chunks) <> 'array' OR pg_catalog.jsonb_array_length(v_role_chunks) = 0 THEN
                RAISE EXCEPTION 'Missing verified chunk IDs for role % in section %.', v_role_key, v_section.id;
            END IF;

            FOR v_chunk_elem IN
                SELECT c.elem FROM pg_catalog.jsonb_array_elements(v_role_chunks) AS c(elem)
            LOOP
                IF pg_catalog.jsonb_typeof(v_chunk_elem) <> 'string' THEN
                    RAISE EXCEPTION 'Non-string chunk ID for role % in section %.', v_role_key, v_section.id;
                END IF;
                v_chunk_text := pg_catalog.btrim(v_chunk_elem #>> '{}');
                IF pg_catalog.length(v_chunk_text) = 0 THEN
                    RAISE EXCEPTION 'Empty chunk ID for role % in section %.', v_role_key, v_section.id;
                END IF;
                v_total_verified_chunks := v_total_verified_chunks + 1;
            END LOOP;

            SELECT COUNT(*), COUNT(DISTINCT c.cid)
            INTO v_role_chunk_count, v_role_distinct_count
            FROM pg_catalog.jsonb_array_elements_text(v_role_chunks) AS c(cid);

            IF v_role_chunk_count <> v_role_distinct_count THEN
                RAISE EXCEPTION 'Duplicate chunk IDs for role % in section %.', v_role_key, v_section.id;
            END IF;
        END LOOP;

        UPDATE public.local_rag_document_sections
        SET chunk_ids_by_role = v_chunks_obj
        WHERE id = v_section.id;
    END LOOP;

    IF v_total_verified_chunks <= 0 OR v_claimed_verified_chunks <> v_total_verified_chunks THEN
        RAISE EXCEPTION 'Verified chunk count mismatch: attested %, counted %.', v_claimed_verified_chunks, v_total_verified_chunks;
    END IF;

    UPDATE public.local_rag_documents
    SET indexing_status = 'indexed'
    WHERE id = p_document_id
    RETURNING * INTO v_doc;

    INSERT INTO public.local_rag_document_audit_log (
        document_id, action, actor_id, actor_role, details
    ) VALUES (
        p_document_id,
        'indexing_verified',
        p_actor_id,
        v_actor_role,
        p_details || pg_catalog.jsonb_build_object(
            'approval_version', v_doc.approval_version,
            'file_hash_sha256', v_doc.file_hash_sha256,
            'verified_at_parsed', v_verified_at,
            'verified_chunk_count', v_total_verified_chunks
        )
    );

    RETURN v_doc;
END;
$$;

CREATE OR REPLACE FUNCTION public.local_rag_fail_indexing(
    p_document_id uuid,
    p_expected_approval_version integer,
    p_expected_file_hash text,
    p_error_details jsonb,
    p_actor_id uuid DEFAULT NULL
)
RETURNS public.local_rag_documents
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public, pg_temp
AS $$
DECLARE
    v_actor_role text := 'system';
    v_expected_file_hash text;
    v_doc public.local_rag_documents;
BEGIN
    IF p_expected_approval_version IS NULL OR p_expected_approval_version < 1 THEN
        RAISE EXCEPTION 'p_expected_approval_version must be a non-null integer >= 1.';
    END IF;
    IF NOT public.local_rag_is_sha256_hex(p_expected_file_hash) THEN
        RAISE EXCEPTION 'p_expected_file_hash must be a valid 64-character hexadecimal SHA-256 digest.';
    END IF;
    v_expected_file_hash := pg_catalog.lower(pg_catalog.btrim(p_expected_file_hash));

    IF p_error_details IS NULL OR pg_catalog.jsonb_typeof(p_error_details) <> 'object' THEN
        RAISE EXCEPTION 'p_error_details must be a non-null JSON object.';
    END IF;

    IF p_actor_id IS NOT NULL THEN
        v_actor_role := public.local_rag_assert_active_admin(p_actor_id);
    END IF;

    SELECT * INTO v_doc
    FROM public.local_rag_documents AS d
    WHERE d.id = p_document_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Document % not found.', p_document_id;
    END IF;

    IF v_doc.approval_status <> 'approved' THEN
        RAISE EXCEPTION 'Cannot fail indexing while approval_status is %.', v_doc.approval_status;
    END IF;

    IF v_doc.indexing_status <> 'indexing' THEN
        RAISE EXCEPTION 'Cannot fail indexing: document % is in indexing_status % (expected indexing).', p_document_id, v_doc.indexing_status;
    END IF;

    IF v_doc.approval_version IS DISTINCT FROM p_expected_approval_version
       OR v_doc.file_hash_sha256 IS DISTINCT FROM v_expected_file_hash THEN
        RAISE EXCEPTION 'Stale indexing failure report rejected for document % (expected version %, current %).',
            p_document_id, p_expected_approval_version, v_doc.approval_version;
    END IF;

    PERFORM pg_catalog.set_config('local_rag.rpc_context', 'fail_indexing', true);

    UPDATE public.local_rag_documents
    SET indexing_status = 'failed'
    WHERE id = p_document_id
    RETURNING * INTO v_doc;

    INSERT INTO public.local_rag_document_audit_log (
        document_id, action, actor_id, actor_role, details
    ) VALUES (
        p_document_id,
        'indexing_failed',
        p_actor_id,
        v_actor_role,
        p_error_details || pg_catalog.jsonb_build_object(
            'approval_version', v_doc.approval_version,
            'file_hash_sha256', v_doc.file_hash_sha256
        )
    );

    RETURN v_doc;
END;
$$;

-- 7. Row-Level Security & Least-Privilege Grants
ALTER TABLE public.local_rag_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.local_rag_document_sections ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.local_rag_document_audit_log ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON public.local_rag_documents FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON public.local_rag_document_sections FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON public.local_rag_document_audit_log FROM PUBLIC, anon, authenticated, service_role;

-- Service role has SELECT-only table privileges; all mutations MUST go through SECURITY DEFINER RPCs
GRANT SELECT ON public.local_rag_documents TO service_role;
GRANT SELECT ON public.local_rag_document_sections TO service_role;
GRANT SELECT ON public.local_rag_document_audit_log TO service_role;

REVOKE ALL ON FUNCTION public.local_rag_is_uuid_text(text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_is_sha256_hex(text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_assert_active_admin(uuid) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_prevent_audit_mutation() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_guard_direct_writes() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_touch_updated_at() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_invalidate_on_doc_change() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_audit_doc_hash_invalidation() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_invalidate_on_section_lifecycle() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_register_document(text, text, text, jsonb, uuid, text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_approve_document(uuid, uuid, jsonb, text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_reject_or_revoke_document(uuid, text, uuid, text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_start_indexing(uuid, integer, text, uuid) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_complete_indexing(uuid, integer, text, jsonb, jsonb, uuid) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.local_rag_fail_indexing(uuid, integer, text, jsonb, uuid) FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.local_rag_register_document(text, text, text, jsonb, uuid, text) TO service_role;
GRANT EXECUTE ON FUNCTION public.local_rag_approve_document(uuid, uuid, jsonb, text) TO service_role;
GRANT EXECUTE ON FUNCTION public.local_rag_reject_or_revoke_document(uuid, text, uuid, text) TO service_role;
GRANT EXECUTE ON FUNCTION public.local_rag_start_indexing(uuid, integer, text, uuid) TO service_role;
GRANT EXECUTE ON FUNCTION public.local_rag_complete_indexing(uuid, integer, text, jsonb, jsonb, uuid) TO service_role;
GRANT EXECUTE ON FUNCTION public.local_rag_fail_indexing(uuid, integer, text, jsonb, uuid) TO service_role;
