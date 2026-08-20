from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from typing import Sequence, Union

revision: str = '53c310bad2d5'
down_revision: Union[str, Sequence[str], None] = 'af067e988f42'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""


    document_status_enum = postgresql.ENUM(
        'UPLOADING', 'PROCESSING', 'READY', 'FAILED',
        name='documentstatus'
    )
    document_status_enum.create(op.get_bind(), checkfirst=True)

    op.add_column('documents', sa.Column('user_id', sa.Integer(), nullable=True))
    op.add_column('documents', sa.Column('parent_id', sa.Integer(), nullable=True))
    op.add_column(
        'documents',
        sa.Column('is_folder', sa.Boolean(), nullable=False, server_default=sa.text('false'))
    )
    op.add_column('documents', sa.Column('gcs_path', sa.String(length=1024), nullable=True))
    op.add_column('documents', sa.Column('mime_type', sa.String(length=128), nullable=True))
    op.add_column('documents', sa.Column('size_bytes', sa.BigInteger(), nullable=True))

    op.add_column(
        'documents',
        sa.Column(
            'status',
            postgresql.ENUM(
                'UPLOADING', 'PROCESSING', 'READY', 'FAILED',
                name='documentstatus',
                create_type=False
            ),
            nullable=False,
            server_default='READY'
        )
    )

    op.add_column('documents', sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True))
    op.add_column('documents', sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True))
    op.add_column('documents', sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True))

    op.alter_column('documents', 'conversation_id',
               existing_type=sa.INTEGER(),
               nullable=True)

    op.create_index(op.f('ix_documents_parent_id'), 'documents', ['parent_id'], unique=False)
    op.create_index(op.f('ix_documents_user_id'), 'documents', ['user_id'], unique=False)
    op.create_unique_constraint(None, 'documents', ['gcs_path'])
    op.create_foreign_key(None, 'documents', 'documents', ['parent_id'], ['id'])
    op.create_foreign_key(None, 'documents', 'users', ['user_id'], ['id'])

    op.execute("""
        UPDATE documents
        SET user_id = conversations.user_id
        FROM conversations
        WHERE documents.conversation_id = conversations.id
          AND documents.user_id IS NULL
    """)

    op.alter_column('documents', 'user_id', existing_type=sa.Integer(), nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(None, 'documents', type_='foreignkey')
    op.drop_constraint(None, 'documents', type_='foreignkey')
    op.drop_constraint(None, 'documents', type_='unique')
    op.drop_index(op.f('ix_documents_user_id'), table_name='documents')
    op.drop_index(op.f('ix_documents_parent_id'), table_name='documents')
    op.alter_column('documents', 'conversation_id',
               existing_type=sa.INTEGER(),
               nullable=False)
    op.drop_column('documents', 'deleted_at')
    op.drop_column('documents', 'updated_at')
    op.drop_column('documents', 'created_at')
    op.drop_column('documents', 'status')
    op.drop_column('documents', 'size_bytes')
    op.drop_column('documents', 'mime_type')
    op.drop_column('documents', 'gcs_path')
    op.drop_column('documents', 'is_folder')
    op.drop_column('documents', 'parent_id')
    op.drop_column('documents', 'user_id')

    postgresql.ENUM(name='documentstatus').drop(op.get_bind(), checkfirst=True)