"""研究消耗独立持久化，包含失败尝试和未返回usage的请求预留。"""
from alembic import op
from models.research_budget import ResearchBudget, ResearchBudgetAttempt, ResearchBudgetCall

revision = "20261014_0014"
down_revision = "20261008_0013"
branch_labels = None
depends_on = None


def upgrade():
    for model in (ResearchBudget, ResearchBudgetAttempt, ResearchBudgetCall):
        model.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    for model in (ResearchBudgetCall, ResearchBudgetAttempt, ResearchBudget):
        model.__table__.drop(op.get_bind(), checkfirst=True)
