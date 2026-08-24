from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.message import Message


def create_conversation(
    db: Session,
    user_id: str,
    title: str | None = None,
):
    conversation = Conversation(
        user_id=user_id,
        title=title,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


# ============================================================
# CREATE MESSAGE
# ============================================================


def create_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
):
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


# ============================================================
# GET SINGLE CONVERSATION
# ============================================================


def get_conversation(
    db: Session,
    conversation_id: str,
    user_id: str,
):
    return (
        db.query(Conversation)
        .filter(
            Conversation.id == str(conversation_id),
            Conversation.user_id == str(user_id),
        )
        .first()
    )


# ============================================================
# GET ALL CONVERSATIONS FOR USER
# ============================================================


def get_conversations_by_user(
    db: Session,
    user_id: str,
):
    return (
        db.query(Conversation)
        .filter(
            Conversation.user_id == str(user_id),
        )
        .order_by(
            Conversation.updated_at.desc(),
        )
        .all()
    )


# ============================================================
# GET LAST 10 MESSAGES
# ============================================================


def get_last_10_messages(
    db: Session,
    conversation_id: str,
):
    messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == str(conversation_id),
        )
        .order_by(
            Message.created_at.desc(),
        )
        .limit(10)
        .all()
    )

    return list(reversed(messages))


# ============================================================
# GET ALL MESSAGES
# ============================================================


def get_all_messages(
    db: Session,
    conversation_id: str,
):
    return (
        db.query(Message)
        .filter(
            Message.conversation_id == str(conversation_id),
        )
        .order_by(
            Message.created_at.asc(),
        )
        .all()
    )


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================


def update_conversation_title(
    db: Session,
    conversation_id: str,
    user_id: str,
    title: str,
):
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == str(conversation_id),
            Conversation.user_id == str(user_id),
        )
        .first()
    )

    if not conversation:
        return None

    conversation.title = title

    db.commit()
    db.refresh(conversation)

    return conversation


# ============================================================
# DELETE CONVERSATION
# ============================================================


def delete_conversation(
    db: Session,
    conversation_id: str,
    user_id: str,
):
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == str(conversation_id),
            Conversation.user_id == str(user_id),
        )
        .first()
    )

    if not conversation:
        return False

    db.delete(conversation)
    db.commit()

    return True