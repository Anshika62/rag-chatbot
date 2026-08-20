from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.message import Message


# ============================================================
# CREATE CONVERSATION
# ============================================================

def create_conversation(
    db: Session,
    user_id: int,
    title: str | None
):
    # Get latest conversation number for this user
    last_conversation = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == user_id
        )
        .order_by(
            Conversation.conversation_number.desc()
        )
        .first()
    )

    next_conversation_number = (
        1
        if last_conversation is None
        else last_conversation.conversation_number + 1
    )

    conversation = Conversation(
        user_id=user_id,
        title=title,
        conversation_number=next_conversation_number
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
    conversation_id: int,
    role: str,
    content: str
):
    # Get latest message number for this conversation
    last_message = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(
            Message.message_number.desc()
        )
        .first()
    )

    next_message_number = (
        1
        if last_message is None
        else last_message.message_number + 1
    )

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        message_number=next_message_number
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
    conversation_id: int,
    user_id: int
):
    return (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        )
        .first()
    )


# ============================================================
# GET ALL CONVERSATIONS FOR USER
# ============================================================

def get_conversations_by_user(
    db: Session,
    user_id: int
):
    return (
        db.query(Conversation)
        .filter(
            Conversation.user_id == user_id
        )
        .order_by(
            Conversation.updated_at.desc()
        )
        .all()
    )


# ============================================================
# GET LAST 10 MESSAGES
# ============================================================

def get_last_10_messages(
    db: Session,
    conversation_id: int
):
    messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(
            Message.created_at.desc()
        )
        .limit(10)
        .all()
    )

    return list(
        reversed(messages)
    )


# ============================================================
# GET ALL MESSAGES
# ============================================================

def get_all_messages(
    db: Session,
    conversation_id: int
):
    return (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(
            Message.message_number.asc()
        )
        .all()
    )


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================

def update_conversation_title(
    db: Session,
    conversation_id: int,
    user_id: int,
    title: str
):
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
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
    conversation_id: int,
    user_id: int
):
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        )
        .first()
    )

    if not conversation:
        return False

    db.delete(conversation)
    db.commit()

    return True