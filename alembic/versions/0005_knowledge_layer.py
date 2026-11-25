"""knowledge layer: kl_* tables, annotations extension, runs.knowledge_items

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kl_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("scope_key", sa.String(length=256), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("prompt_version", sa.String(length=32), nullable=True),
        sa.Column("prompt_hash", sa.String(length=64), nullable=True),
        sa.Column("schema_version", sa.Integer(), nullable=True),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("input_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("output_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cost_usd", sa.Float(), server_default="0", nullable=False),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'done', 'failed', 'aborted', 'promoted')",
            name="ck_kl_jobs_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_kl_jobs_stage_scope", "kl_jobs", ["stage", "scope_key"], unique=False)
    op.create_table(
        "kl_overrides",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("target_key", sa.String(length=256), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("corrected", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("reviewer", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "action IN ('approve', 'edit', 'reject', 'set_boundary', 'set_abbrev')",
            name="ck_kl_overrides_action",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_kl_overrides_target_key"), "kl_overrides", ["target_key"], unique=False
    )
    op.create_table(
        "kl_witnesses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("siglum", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("name_he", sa.String(length=256), nullable=True),
        sa.Column("date_from", sa.Integer(), nullable=True),
        sa.Column("date_to", sa.Integer(), nullable=True),
        sa.Column("tradition", sa.String(length=64), nullable=True),
        sa.Column("holding", sa.String(length=256), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("license_basis", sa.Text(), nullable=True),
        sa.Column("can_index", sa.Boolean(), nullable=False),
        sa.Column("can_display_public", sa.Boolean(), nullable=False),
        sa.CheckConstraint(
            "kind IN ('manuscript', 'print', 'genizah', 'rishon_quote', 'edition')",
            name="ck_kl_witnesses_kind",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("siglum"),
    )
    op.create_table(
        "kl_tasks",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("task_key", sa.String(length=256), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("output_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("cost_usd", sa.Float(), server_default="0", nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state IN ('pending', 'running', 'done', 'failed', 'skipped')", name="ck_kl_tasks_state"
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["kl_jobs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "task_key"),
    )
    op.create_index(op.f("ix_kl_tasks_input_hash"), "kl_tasks", ["input_hash"], unique=False)
    op.create_index(op.f("ix_kl_tasks_job_id"), "kl_tasks", ["job_id"], unique=False)
    op.create_table(
        "kl_coverage",
        sa.Column("scope_key", sa.String(length=256), nullable=False),
        sa.Column("work_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('covered', 'not_in_work', 'not_ingested', 'unknown')",
            name="ck_kl_coverage_status",
        ),
        sa.ForeignKeyConstraint(
            ["work_id"],
            ["works.id"],
        ),
        sa.PrimaryKeyConstraint("scope_key", "work_id"),
    )
    op.create_table(
        "kl_entities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("name_he", sa.String(length=256), nullable=True),
        sa.Column("name_en", sa.String(length=256), nullable=True),
        sa.Column("author_id", sa.Integer(), nullable=True),
        sa.Column("work_id", sa.Integer(), nullable=True),
        sa.Column("period", sa.String(length=32), nullable=True),
        sa.Column("aliases", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["author_id"],
            ["authors.id"],
        ),
        sa.ForeignKeyConstraint(
            ["work_id"],
            ["works.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_index(op.f("ix_kl_entities_author_id"), "kl_entities", ["author_id"], unique=False)
    op.create_index(op.f("ix_kl_entities_work_id"), "kl_entities", ["work_id"], unique=False)
    op.create_table(
        "kl_sources",
        sa.Column("work_id", sa.Integer(), nullable=False),
        sa.Column("tier", sa.String(length=24), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("origin", sa.String(length=64), nullable=True),
        sa.Column("license_tier", sa.String(length=32), nullable=False),
        sa.Column("license_basis", sa.Text(), nullable=True),
        sa.Column("weight_tier", sa.String(length=12), nullable=True),
        sa.Column("authority_areas", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("added_by", sa.String(length=64), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "license_tier IN ('PUBLIC_COMMERCIAL', 'NON_COMMERCIAL_ONLY', 'PERSONAL_ONLY_RESTRICTED')",
            name="ck_kl_sources_license_tier",
        ),
        sa.CheckConstraint(
            "role IN ('primary', 'lens', 'witness', 'reference_only')", name="ck_kl_sources_role"
        ),
        sa.CheckConstraint(
            "tier IN ('canonical', 'rishonim', 'codes', 'nosei_kelim', 'acharonim', 'lamdanut', 'later_responsa', 'sugya_compendium', 'modern_secondary', 'girsa_witness')",
            name="ck_kl_sources_tier",
        ),
        sa.CheckConstraint(
            "weight_tier IN ('core', 'major', 'extended', 'long_tail') OR weight_tier IS NULL",
            name="ck_kl_sources_weight_tier",
        ),
        sa.ForeignKeyConstraint(
            ["work_id"],
            ["works.id"],
        ),
        sa.PrimaryKeyConstraint("work_id"),
    )
    op.create_table(
        "kl_work_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("work_id", sa.Integer(), nullable=False),
        sa.Column("alias", sa.String(length=256), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=True),
        sa.ForeignKeyConstraint(
            ["work_id"],
            ["works.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("work_id", "alias"),
    )
    op.create_index(op.f("ix_kl_work_aliases_alias"), "kl_work_aliases", ["alias"], unique=False)
    op.create_index(
        op.f("ix_kl_work_aliases_work_id"), "kl_work_aliases", ["work_id"], unique=False
    )
    op.create_table(
        "kl_items",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("item_key", sa.String(length=256), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("scope_key", sa.String(length=256), nullable=False),
        sa.Column("work_id", sa.Integer(), nullable=True),
        sa.Column("start_segment_id", sa.BigInteger(), nullable=True),
        sa.Column("end_segment_id", sa.BigInteger(), nullable=True),
        sa.Column("title_he", sa.String(length=512), nullable=True),
        sa.Column("body", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("stale", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("prompt_version", sa.String(length=32), nullable=True),
        sa.Column("prompt_hash", sa.String(length=64), nullable=True),
        sa.Column("source_version_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("can_display_public", sa.Boolean(), nullable=False),
        sa.Column("can_train", sa.Boolean(), nullable=False),
        sa.Column("check_report", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("rejected_by", sa.String(length=16), nullable=True),
        sa.Column("search_text_he", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "kind IN ('sugya', 'sugya_summary', 'commentary_note', 'issue', 'position', 'machloket', 'open_question', 'chakira', 'source_claim', 'girsa_variant', 'method_principle', 'topic_card', 'mishnah_card')",
            name="ck_kl_items_kind",
        ),
        sa.CheckConstraint(
            "status IN ('generated', 'verified', 'rejected')", name="ck_kl_items_status"
        ),
        sa.ForeignKeyConstraint(
            ["end_segment_id"],
            ["segments.id"],
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["kl_jobs.id"],
        ),
        sa.ForeignKeyConstraint(
            ["start_segment_id"],
            ["segments.id"],
        ),
        sa.ForeignKeyConstraint(
            ["work_id"],
            ["works.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("item_key", "input_hash"),
        sa.UniqueConstraint("item_key", "version"),
    )
    op.create_index(op.f("ix_kl_items_job_id"), "kl_items", ["job_id"], unique=False)
    op.create_index("ix_kl_items_scope_kind", "kl_items", ["scope_key", "kind"], unique=False)
    op.create_index(
        "ix_kl_items_search",
        "kl_items",
        [sa.literal_column("to_tsvector('simple', coalesce(search_text_he, ''))")],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(op.f("ix_kl_items_work_id"), "kl_items", ["work_id"], unique=False)
    op.create_index(
        "uq_kl_items_current",
        "kl_items",
        ["item_key"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )
    op.create_table(
        "kl_evidence",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("claim_id", sa.String(length=64), nullable=True),
        sa.Column("segment_id", sa.BigInteger(), nullable=True),
        sa.Column("version_id", sa.Integer(), nullable=True),
        sa.Column("span_start", sa.Integer(), nullable=True),
        sa.Column("span_end", sa.Integer(), nullable=True),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("quote_prefix", sa.String(length=32), nullable=True),
        sa.Column("quote_suffix", sa.String(length=32), nullable=True),
        sa.Column("raw_hash", sa.String(length=64), nullable=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("locate_status", sa.String(length=16), nullable=False),
        sa.Column("entail_status", sa.String(length=16), nullable=True),
        sa.CheckConstraint(
            "locate_status IN ('located', 'not_found', 'ambiguous', 'out_of_scope')",
            name="ck_kl_evidence_locate",
        ),
        sa.CheckConstraint(
            "role IN ('states', 'supports', 'attributes', 'adopts', 'disputes', 'interprets', 'proposes', 'claims_error', 'witness')",
            name="ck_kl_evidence_role",
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["kl_items.id"],
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["segments.id"],
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["versions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_kl_evidence_item", "kl_evidence", ["item_id"], unique=False)
    op.create_index("ix_kl_evidence_segment", "kl_evidence", ["segment_id"], unique=False)
    op.create_table(
        "kl_item_segments",
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("role", sa.String(length=12), nullable=False),
        sa.Column("ord", sa.Integer(), nullable=False),
        sa.CheckConstraint("role IN ('member', 'anchor', 'commentary')", name="ck_kl_iseg_role"),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["kl_items.id"],
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["segments.id"],
        ),
        sa.PrimaryKeyConstraint("item_id", "segment_id", "role"),
    )
    op.create_index("ix_kl_item_segments_segment", "kl_item_segments", ["segment_id"], unique=False)
    op.create_table(
        "kl_links",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("from_item_id", sa.BigInteger(), nullable=False),
        sa.Column("to_item_id", sa.BigInteger(), nullable=False),
        sa.Column("rel", sa.String(length=16), nullable=False),
        sa.CheckConstraint(
            "rel IN ('part_of', 'summarizes', 'annotates', 'depends_on', 'discusses', 'continues', 'sub_issue_of', 'position_on', 'about_issue', 'raised_in')",
            name="ck_kl_links_rel",
        ),
        sa.ForeignKeyConstraint(
            ["from_item_id"],
            ["kl_items.id"],
        ),
        sa.ForeignKeyConstraint(
            ["to_item_id"],
            ["kl_items.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("from_item_id", "to_item_id", "rel"),
    )
    op.create_index("ix_kl_links_to", "kl_links", ["to_item_id"], unique=False)
    op.create_table(
        "kl_reception",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("target_item_id", sa.BigInteger(), nullable=False),
        sa.Column("target_claim", sa.String(length=64), nullable=False),
        sa.Column("context", sa.String(length=48), nullable=False),
        sa.Column("weight_label", sa.String(length=16), nullable=False),
        sa.Column("basis", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "weight_label <> 'rejected' OR context = 'lemaase'", name="ck_kl_reception_rejected"
        ),
        sa.CheckConstraint(
            "weight_label IN ('primary', 'alternative', 'snif_only', 'rejected', 'disputed')",
            name="ck_kl_reception_label",
        ),
        sa.CheckConstraint(
            "context IN ('iyun', 'lemaase', 'snif', 'shaat_hadchak', 'bediavad', 'public_instruction') OR context LIKE 'minhag:%'",
            name="ck_kl_reception_context",
        ),
        sa.ForeignKeyConstraint(
            ["target_item_id"],
            ["kl_items.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("target_item_id", "target_claim", "context", "policy_version"),
    )
    op.create_table(
        "kl_relations",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("from_item_id", sa.BigInteger(), nullable=False),
        sa.Column("from_claim", sa.String(length=64), nullable=True),
        sa.Column("to_item_id", sa.BigInteger(), nullable=False),
        sa.Column("to_claim", sa.String(length=64), nullable=True),
        sa.Column("rel", sa.String(length=24), nullable=False),
        sa.Column("asserted_by_entity", sa.Integer(), nullable=True),
        sa.Column("asserted_by_work", sa.Integer(), nullable=True),
        sa.Column("attribution", sa.String(length=24), nullable=True),
        sa.Column("stance", sa.String(length=16), nullable=True),
        sa.Column("evidence_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.CheckConstraint(
            "attribution IN ('direct', 'reported', 'explicit', 'inferred_by_source', 'inferred_by_system') OR attribution IS NULL",
            name="ck_kl_relations_attribution",
        ),
        sa.CheckConstraint(
            "rel IN ('objects_to', 'answers', 'supports', 'disputes', 'rules_like', 'qualifies', 'reads_as', 'aligns_with_side', 'rejects_reading', 'claims_error_in', 'depends_on_girsa', 'parallel_to', 'same_as', 'derived_from', 'takes_precedence_over')",
            name="ck_kl_relations_rel",
        ),
        sa.CheckConstraint(
            "stance IN ('asserts', 'suggests', 'considers', 'rejects', 'reports_others', 'as_snif') OR stance IS NULL",
            name="ck_kl_relations_stance",
        ),
        sa.ForeignKeyConstraint(
            ["asserted_by_entity"],
            ["kl_entities.id"],
        ),
        sa.ForeignKeyConstraint(
            ["asserted_by_work"],
            ["works.id"],
        ),
        sa.ForeignKeyConstraint(
            ["from_item_id"],
            ["kl_items.id"],
        ),
        sa.ForeignKeyConstraint(
            ["to_item_id"],
            ["kl_items.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_kl_relations_from", "kl_relations", ["from_item_id"], unique=False)
    op.create_index("ix_kl_relations_to", "kl_relations", ["to_item_id"], unique=False)
    op.create_table(
        "kl_reviews",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("item_id", sa.BigInteger(), nullable=False),
        sa.Column("reviewer", sa.String(length=64), nullable=True),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["kl_items.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_kl_reviews_item_id"), "kl_reviews", ["item_id"], unique=False)
    op.add_column(
        "annotations",
        sa.Column("status", sa.String(length=16), server_default="generated", nullable=False),
    )
    op.add_column(
        "annotations",
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("true"), nullable=False),
    )
    op.add_column("annotations", sa.Column("method", sa.String(length=48), nullable=True))
    op.add_column("annotations", sa.Column("method_version", sa.String(length=32), nullable=True))
    op.add_column("annotations", sa.Column("context_domain", sa.String(length=16), nullable=True))
    op.add_column(
        "annotations", sa.Column("context", postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    )
    op.add_column("annotations", sa.Column("entity_id", sa.Integer(), nullable=True))
    op.add_column("annotations", sa.Column("job_id", sa.Integer(), nullable=True))
    op.add_column(
        "annotations",
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True
        ),
    )
    op.create_index(op.f("ix_annotations_job_id"), "annotations", ["job_id"], unique=False)
    op.create_index(
        "uq_annotations_current_span",
        "annotations",
        ["segment_id", "version_id", "span_start", "span_end", "kind"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )
    op.create_foreign_key(
        "fk_annotations_entity_id", "annotations", "kl_entities", ["entity_id"], ["id"]
    )
    op.create_foreign_key("fk_annotations_job_id", "annotations", "kl_jobs", ["job_id"], ["id"])
    op.add_column(
        "runs",
        sa.Column(
            "knowledge_items",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
    )
    # Human decisions that a later model output has run into: review queue.
    op.execute(
        """
        CREATE VIEW kl_conflicts AS
        SELECT o.id AS override_id, o.target_key, o.action, i.id AS item_id,
               i.version, i.job_id, i.status, i.created_at AS item_created_at
        FROM kl_overrides o
        JOIN kl_items i ON i.item_key = o.target_key
        WHERE o.action IN ('reject', 'edit', 'approve', 'set_boundary')
          AND i.created_at > o.created_at
          AND i.prompt_version IS DISTINCT FROM 'human'
        """
    )
    for table in ("cards", "card_dependencies"):
        op.execute(f"COMMENT ON TABLE {table} IS 'DEPRECATED: superseded by kl_items'")


def downgrade() -> None:
    for table in ("cards", "card_dependencies"):
        op.execute(f"COMMENT ON TABLE {table} IS NULL")
    op.execute("DROP VIEW IF EXISTS kl_conflicts")
    op.drop_column("runs", "knowledge_items")
    op.drop_constraint("fk_annotations_job_id", "annotations", type_="foreignkey")
    op.drop_constraint("fk_annotations_entity_id", "annotations", type_="foreignkey")
    op.drop_index(
        "uq_annotations_current_span",
        table_name="annotations",
        postgresql_where=sa.text("is_current"),
    )
    op.drop_index(op.f("ix_annotations_job_id"), table_name="annotations")
    op.drop_column("annotations", "created_at")
    op.drop_column("annotations", "job_id")
    op.drop_column("annotations", "entity_id")
    op.drop_column("annotations", "context")
    op.drop_column("annotations", "context_domain")
    op.drop_column("annotations", "method_version")
    op.drop_column("annotations", "method")
    op.drop_column("annotations", "is_current")
    op.drop_column("annotations", "status")
    op.drop_index(op.f("ix_kl_reviews_item_id"), table_name="kl_reviews")
    op.drop_table("kl_reviews")
    op.drop_index("ix_kl_relations_to", table_name="kl_relations")
    op.drop_index("ix_kl_relations_from", table_name="kl_relations")
    op.drop_table("kl_relations")
    op.drop_table("kl_reception")
    op.drop_index("ix_kl_links_to", table_name="kl_links")
    op.drop_table("kl_links")
    op.drop_index("ix_kl_item_segments_segment", table_name="kl_item_segments")
    op.drop_table("kl_item_segments")
    op.drop_index("ix_kl_evidence_segment", table_name="kl_evidence")
    op.drop_index("ix_kl_evidence_item", table_name="kl_evidence")
    op.drop_table("kl_evidence")
    op.drop_index(
        "uq_kl_items_current", table_name="kl_items", postgresql_where=sa.text("is_current")
    )
    op.drop_index(op.f("ix_kl_items_work_id"), table_name="kl_items")
    op.drop_index("ix_kl_items_search", table_name="kl_items", postgresql_using="gin")
    op.drop_index("ix_kl_items_scope_kind", table_name="kl_items")
    op.drop_index(op.f("ix_kl_items_job_id"), table_name="kl_items")
    op.drop_table("kl_items")
    op.drop_index(op.f("ix_kl_work_aliases_work_id"), table_name="kl_work_aliases")
    op.drop_index(op.f("ix_kl_work_aliases_alias"), table_name="kl_work_aliases")
    op.drop_table("kl_work_aliases")
    op.drop_table("kl_sources")
    op.drop_index(op.f("ix_kl_entities_work_id"), table_name="kl_entities")
    op.drop_index(op.f("ix_kl_entities_author_id"), table_name="kl_entities")
    op.drop_table("kl_entities")
    op.drop_table("kl_coverage")
    op.drop_index(op.f("ix_kl_tasks_job_id"), table_name="kl_tasks")
    op.drop_index(op.f("ix_kl_tasks_input_hash"), table_name="kl_tasks")
    op.drop_table("kl_tasks")
    op.drop_table("kl_witnesses")
    op.drop_index(op.f("ix_kl_overrides_target_key"), table_name="kl_overrides")
    op.drop_table("kl_overrides")
    op.drop_index("ix_kl_jobs_stage_scope", table_name="kl_jobs")
    op.drop_table("kl_jobs")
