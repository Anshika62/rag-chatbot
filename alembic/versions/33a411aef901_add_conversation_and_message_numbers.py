"""add conversation and message numbers

Revision ID: 33a411aef901
Revises: 53c310bad2d5
Create Date: 2026-08-20
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "33a411aef901"
down_revision = "53c310bad2d5"
branch_labels = None
depends_on = None


def upgrade():

    # ==========================================
    # 1. Add conversation_number as nullable
    # ==========================================

    op.add_column(
        "conversations",
        sa.Column(
            "conversation_number",
            sa.Integer(),
            nullable=True
        )
    )

    # Give existing conversations a number
    # separately for every user, starting from 1
    op.execute(
        """
        WITH numbered_conversations AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY created_at ASC, id ASC
                ) AS new_number
            FROM conversations
        )
        UPDATE conversations
        SET conversation_number =
            numbered_conversations.new_number
        FROM numbered_conversations
        WHERE conversations.id =
            numbered_conversations.id
        """
    )

    # Now make the column NOT NULL
    op.alter_column(
        "conversations",
        "conversation_number",
        existing_type=sa.Integer(),
        nullable=False
    )

    # Add unique constraint
    op.create_unique_constraint(
        "uq_user_conversation_number",
        "conversations",
        [
            "user_id",
            "conversation_number"
        ]
    )

    # ==========================================
    # 2. Add message_number as nullable
    # ==========================================

    op.add_column(
        "messages",
        sa.Column(
            "message_number",
            sa.Integer(),
            nullable=True
        )
    )

    # Give existing messages a number
    # separately for every conversation,
    # starting from 1
    op.execute(
        """
        WITH numbered_messages AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY conversation_id
                    ORDER BY created_at ASC, id ASC
                ) AS new_number
            FROM messages
        )
        UPDATE messages
        SET message_number =
            numbered_messages.new_number
        FROM numbered_messages
        WHERE messages.id =
            numbered_messages.id
        """
    )

    # Now make the column NOT NULL
    op.alter_column(
        "messages",
        "message_number",
        existing_type=sa.Integer(),
        nullable=False
    )

    # Add unique constraint
    op.create_unique_constraint(
        "uq_conversation_message_number",
        "messages",
        [
            "conversation_id",
            "message_number"
        ]
    )


def downgrade():

    # Remove message constraint and column
    op.drop_constraint(
        "uq_conversation_message_number",
        "messages",
        type_="unique"
    )

    op.drop_column(
        "messages",
        "message_number"
    )

    # Remove conversation constraint and column
    op.drop_constraint(
        "uq_user_conversation_number",
        "conversations",
        type_="unique"
    )

    op.drop_column(
        "conversations",
        "conversation_number"
    )