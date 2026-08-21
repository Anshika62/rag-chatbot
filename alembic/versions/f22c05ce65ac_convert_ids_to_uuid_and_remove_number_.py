"""convert ids to uuid and remove number fields

Revision ID: f22c05ce65ac
Revises: a755ae3ea408
Create Date: 2026-08-21 12:04:13.455989
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f22c05ce65ac"
down_revision: Union[str, Sequence[str], None] = "a755ae3ea408"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # Required for generating UUIDs for existing records.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ---------------------------------------------------------
    # Drop existing foreign keys before changing related ID types
    # ---------------------------------------------------------
    op.execute(
        """
        DO $$
        DECLARE
            constraint_record RECORD;
        BEGIN
            FOR constraint_record IN
                SELECT
                    conrelid::regclass AS table_name,
                    conname AS constraint_name
                FROM pg_constraint
                WHERE contype = 'f'
                AND conrelid IN (
                    'conversations'::regclass,
                    'messages'::regclass,
                    'documents'::regclass,
                    'docs_chunks'::regclass
                )
            LOOP
                EXECUTE format(
                    'ALTER TABLE %s DROP CONSTRAINT %I',
                    constraint_record.table_name,
                    constraint_record.constraint_name
                );
            END LOOP;
        END $$;
        """
    )

    # ---------------------------------------------------------
    # Remove old global number logic
    # ---------------------------------------------------------
    op.drop_constraint(
        op.f("uq_user_conversation_number"),
        "conversations",
        type_="unique",
    )
    op.drop_column("conversations", "conversation_number")

    op.drop_constraint(
        op.f("uq_conversation_message_number"),
        "messages",
        type_="unique",
    )
    op.drop_column("messages", "message_number")

    # ---------------------------------------------------------
    # Convert all related ID columns from Integer to String(36)
    # ---------------------------------------------------------

    # Users
    op.alter_column(
        "users",
        "id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=False,
        postgresql_using="id::varchar",
    )

    # Conversations
    op.alter_column(
        "conversations",
        "id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=False,
        postgresql_using="id::varchar",
    )
    op.alter_column(
        "conversations",
        "user_id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=False,
        postgresql_using="user_id::varchar",
    )

    # Messages
    op.alter_column(
        "messages",
        "id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=False,
        postgresql_using="id::varchar",
    )
    op.alter_column(
        "messages",
        "conversation_id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=False,
        postgresql_using="conversation_id::varchar",
    )

    # Documents
    op.alter_column(
        "documents",
        "id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=False,
        postgresql_using="id::varchar",
    )
    op.alter_column(
        "documents",
        "user_id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=False,
        postgresql_using="user_id::varchar",
    )
    op.alter_column(
        "documents",
        "parent_id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=True,
        postgresql_using="parent_id::varchar",
    )
    op.alter_column(
        "documents",
        "conversation_id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=True,
        postgresql_using="conversation_id::varchar",
    )

    # Document chunks
    op.alter_column(
        "docs_chunks",
        "id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=False,
        postgresql_using="id::varchar",
    )
    op.alter_column(
        "docs_chunks",
        "doc_id",
        existing_type=sa.INTEGER(),
        type_=sa.String(length=36),
        existing_nullable=False,
        postgresql_using="doc_id::varchar",
    )

    # ---------------------------------------------------------
    # Convert existing User IDs to actual UUID strings
    # and update all related user_id references
    # ---------------------------------------------------------
    op.execute(
        """
        CREATE TEMP TABLE user_id_map AS
        SELECT
            id AS old_id,
            gen_random_uuid()::text AS new_id
        FROM users
        """
    )

    op.execute(
        """
        UPDATE conversations AS c
        SET user_id = m.new_id
        FROM user_id_map AS m
        WHERE c.user_id = m.old_id
        """
    )

    op.execute(
        """
        UPDATE documents AS d
        SET user_id = m.new_id
        FROM user_id_map AS m
        WHERE d.user_id = m.old_id
        """
    )

    op.execute(
        """
        UPDATE users AS u
        SET id = m.new_id
        FROM user_id_map AS m
        WHERE u.id = m.old_id
        """
    )

    # ---------------------------------------------------------
    # Convert existing Conversation IDs to actual UUID strings
    # and update all related conversation_id references
    # ---------------------------------------------------------
    op.execute(
        """
        CREATE TEMP TABLE conversation_id_map AS
        SELECT
            id AS old_id,
            gen_random_uuid()::text AS new_id
        FROM conversations
        """
    )

    op.execute(
        """
        UPDATE messages AS m
        SET conversation_id = c.new_id
        FROM conversation_id_map AS c
        WHERE m.conversation_id = c.old_id
        """
    )

    op.execute(
        """
        UPDATE documents AS d
        SET conversation_id = c.new_id
        FROM conversation_id_map AS c
        WHERE d.conversation_id = c.old_id
        """
    )

    op.execute(
        """
        UPDATE conversations AS c
        SET id = m.new_id
        FROM conversation_id_map AS m
        WHERE c.id = m.old_id
        """
    )

    # ---------------------------------------------------------
    # Convert existing Document IDs to actual UUID strings
    # and update doc_id / parent_id references
    # ---------------------------------------------------------
    op.execute(
        """
        CREATE TEMP TABLE document_id_map AS
        SELECT
            id AS old_id,
            gen_random_uuid()::text AS new_id
        FROM documents
        """
    )

    op.execute(
        """
        UPDATE docs_chunks AS dc
        SET doc_id = m.new_id
        FROM document_id_map AS m
        WHERE dc.doc_id = m.old_id
        """
    )

    op.execute(
        """
        UPDATE documents AS d
        SET parent_id = m.new_id
        FROM document_id_map AS m
        WHERE d.parent_id = m.old_id
        """
    )

    op.execute(
        """
        UPDATE documents AS d
        SET id = m.new_id
        FROM document_id_map AS m
        WHERE d.id = m.old_id
        """
    )

    # ---------------------------------------------------------
    # Convert Message and Document Chunk IDs to UUID strings
    # ---------------------------------------------------------
    op.execute(
        """
        UPDATE messages
        SET id = gen_random_uuid()::text
        """
    )

    op.execute(
        """
        UPDATE docs_chunks
        SET id = gen_random_uuid()::text
        """
    )

    # ---------------------------------------------------------
    # Recreate foreign key relationships
    # ---------------------------------------------------------
    op.create_foreign_key(
        "fk_conversations_user_id",
        "conversations",
        "users",
        ["user_id"],
        ["id"],
    )

    op.create_foreign_key(
        "fk_messages_conversation_id",
        "messages",
        "conversations",
        ["conversation_id"],
        ["id"],
    )

    op.create_foreign_key(
        "fk_documents_user_id",
        "documents",
        "users",
        ["user_id"],
        ["id"],
    )

    op.create_foreign_key(
        "fk_documents_parent_id",
        "documents",
        "documents",
        ["parent_id"],
        ["id"],
    )

    op.create_foreign_key(
        "fk_documents_conversation_id",
        "documents",
        "conversations",
        ["conversation_id"],
        ["id"],
    )

    op.create_foreign_key(
        "fk_docs_chunks_doc_id",
        "docs_chunks",
        "documents",
        ["doc_id"],
        ["id"],
    )


def downgrade() -> None:
    """Downgrade schema."""

    raise NotImplementedError(
        "This migration cannot safely be downgraded because existing "
        "integer IDs were replaced with generated UUID values."
    )