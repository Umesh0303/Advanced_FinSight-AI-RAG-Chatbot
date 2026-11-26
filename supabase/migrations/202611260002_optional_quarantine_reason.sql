-- Additive migration: make quarantine_resolution_reason optional when an Admin approves a quarantined section.
-- Preserves Admin-only verification, row locking, and audit logging.

ALTER TABLE public.local_rag_document_sections
    DROP CONSTRAINT IF EXISTS local_rag_sections_quarantine_resolution_chk;

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
