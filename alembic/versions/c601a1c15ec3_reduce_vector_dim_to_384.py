"""reduce vector dim to 384

Revision ID: c601a1c15ec3
Revises: 
Create Date: 2025-06-24 14:51:37.057032

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy.vector import VECTOR

# revision identifiers, used by Alembic.
revision: str = 'c601a1c15ec3'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('codex_entries', 'vector',
               existing_type=VECTOR(dim=1536),
               type_=VECTOR(dim=384),
               existing_nullable=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('codex_entries', 'vector',
               existing_type=VECTOR(dim=384),
               type_=VECTOR(dim=1536),
               existing_nullable=True)
    # ### end Alembic commands ###
