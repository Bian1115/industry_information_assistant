"""会话临时证据空间、正文、chunk和独立清理任务。"""
from alembic import op
from models.session_evidence import (SessionEvidenceSpace, SessionEvidenceDocument,
                                     SessionEvidenceChunk, SessionEvidencePurge)

revision = "20261008_0013"
down_revision = "20261006_0012"
branch_labels = None
depends_on = None


def upgrade():
    for model in (SessionEvidenceSpace, SessionEvidenceDocument, SessionEvidenceChunk, SessionEvidencePurge):
        model.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    for model in (SessionEvidencePurge, SessionEvidenceChunk, SessionEvidenceDocument, SessionEvidenceSpace):
        model.__table__.drop(op.get_bind(), checkfirst=True)
