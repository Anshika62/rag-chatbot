from sqlalchemy import func
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
        last_conversation_number = (
            db.query(
                func.max(Conversation.conversation_number)
            )
            .filter(
                Conversation.user_id == user_id
            )
            .scalar()
        )

        next_conversation_number = (
            (last_conversation_number or 0) + 1
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
        last_message_number = (
            db.query(
                func.max(Message.message_number)
            )
            .filter(
                Message.conversation_id == conversation_id
            )
            .scalar()
        )

        next_message_number = (
            (last_message_number or 0) + 1
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
            .order_by(
                Message.created_at.desc()
            )
            .limit(10)
            .all()
        )

        return list(reversed(messages))

    except SQLAlchemyError:
        db.rollback()
        raise


def get_conversations_by_user(
    db: Session,
    user_id: int
):
    try:
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

    except SQLAlchemyError:
        db.rollback()
        raise


def get_all_messages(
    db: Session,
    conversation_id: int
):
    try:
        return (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation_id
            )
            .order_by(
                Message.created_at.asc()
            )
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