"""runs table (eval / system runs)

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("question_id", sa.String(length=128), nullable=True),
        sa.Column("golden_version", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("prompt_hash", sa.String(length=64), nullable=True),
        sa.Column("retrieved_sources", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("answer", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("verification", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("verification_passed", sa.Boolean(), nullable=True),
        sa.Column("review_label", sa.String(length=32), nullable=True),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_runs_created_at"), "runs", ["created_at"], unique=False)
    op.create_index(op.f("ix_runs_question_id"), "runs", ["question_id"], unique=False)
    op.create_index(op.f("ix_runs_model"), "runs", ["model"], unique=False)
    op.create_index(op.f("ix_runs_prompt_hash"), "runs", ["prompt_hash"], unique=False)
    op.create_index(
        op.f("ix_runs_verification_passed"), "runs", ["verification_passed"], unique=False
    )


def downgrade() -> None:
    op.drop_table("runs")
