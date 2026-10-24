"""Mark the schema that existed before reliable memory jobs.

Existing installations stamp this revision once. A fresh installation proceeds to
the next revision, which creates the current ORM schema when no tables exist.

Revision ID: 20260819_0001
Revises: None
"""

from typing import Optional

revision: str = "20260819_0001"
down_revision: Optional[str] = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
