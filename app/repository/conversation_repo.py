from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.models.conversation import Conversation
from app.models.message import Message


def create_conversation(
    db: Session,
    user_id: int,
    title: str | None
):
    try:
        conversation = Conversation(
            user_id=user_id,
            title=title
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return conversation

    except SQLAlchemyError:
        db.rollback()
        raise


def create_message(
    db: Session,
    conversation_id: int,
    role: str,
    content: str
):
    try:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )

        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    except SQLAlchemyError:
        db.rollback()
        raise


def get_conversation(
    db: Session,
    conversation_id: int,
    user_id: int
):
    try:
        return (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
            .first()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


def get_last_10_messages(
    db: Session,
    conversation_id: int
):
    try:
        messages = (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation_id
            )
            .order_by(Message.created_at.desc())
            .limit(10)
            .all()
        )

        return list(reversed(messages))

    except SQLAlchemyError:
        db.rollback()
        raise


# ============ NAYE FUNCTIONS ============

def get_conversations_by_user(
    db: Session,
    user_id: int
):
    """User ki saari conversations, latest update wali sabse upar."""
    try:
        return (
            db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


def get_all_messages(
    db: Session,
    conversation_id: int
):
    """Ek conversation ke saare messages, purane se naye order mein."""
    try:
        return (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


def update_conversation_title(
    db: Session,
    conversation_id: int,
    user_id: int,
    title: str
):
    """Sirf title update karo, agar conversation is user ki hai to."""
    try:
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

    except SQLAlchemyError:
        db.rollback()
        raise


def delete_conversation(
    db: Session,
    conversation_id: int,
    user_id: int
):
    """Conversation delete karo (messages/documents cascade delete honge model ki wajah se)."""
    try:
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

    except SQLAlchemyError:
        db.rollback()
        raise