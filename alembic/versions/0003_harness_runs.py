"""harness run columns on runs + full-text index for search_text

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "messages",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    for col in ("steps", "input_tokens", "output_tokens"):
        op.add_column(
            "runs", sa.Column(col, sa.Integer(), server_default=sa.text("0"), nullable=False)
        )
    op.add_column("runs", sa.Column("cost_usd", sa.Float(), nullable=True))
    # search_text(): basic Postgres full-text search on the search-oriented form B
    op.execute(
        "CREATE INDEX ix_segment_text_fts ON segment_text USING gin (to_tsvector('simple', text_b))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_segment_text_fts")
    for col in ("cost_usd", "output_tokens", "input_tokens", "steps", "messages"):
        op.drop_column("runs", col)
