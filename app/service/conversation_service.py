from sqlalchemy.orm import Session

from app.repository.conversation_repo import (
    create_conversation,
    create_message,
    get_conversation,
    get_last_10_messages,
)


# ============================================================
# CREATE CONVERSATION
# ============================================================


def start_conversation(
    db: Session,
    user_id: int,
    title: str | None = None,
):
    return create_conversation(
        db=db,
        user_id=user_id,
        title=title,
    )


# ============================================================
# SAVE MESSAGE
# ============================================================


def save_message(
    db: Session,
    user_id: int,
    conversation_id: int,
    role: str,
    content: str,
):
    conversation = get_conversation(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
    )

    if conversation is None:
        return None

    return create_message(
        db=db,
        conversation_id=conversation_id,
        role=role,
        content=content,
    )


# ============================================================
# GET CONVERSATION HISTORY
# ============================================================


def get_conversation_history(
    db: Session,
    conversation_id: int,
    user_id: int,
):
    conversation = get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=user_id,
    )

    if conversation is None:
        return None

    return get_last_10_messages(
        db=db,
        conversation_id=conversation_id,
    )