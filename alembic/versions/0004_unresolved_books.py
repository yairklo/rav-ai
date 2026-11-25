"""unresolved_links: which books the two citations belong to (per-book resolution reports)

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("unresolved_links", sa.Column("book_a", sa.String(512), nullable=True))
    op.add_column("unresolved_links", sa.Column("book_b", sa.String(512), nullable=True))
    op.create_index("ix_unresolved_links_book_a", "unresolved_links", ["book_a"])
    op.create_index("ix_unresolved_links_book_b", "unresolved_links", ["book_b"])


def downgrade() -> None:
    op.drop_index("ix_unresolved_links_book_b", "unresolved_links")
    op.drop_index("ix_unresolved_links_book_a", "unresolved_links")
    op.drop_column("unresolved_links", "book_b")
    op.drop_column("unresolved_links", "book_a")
