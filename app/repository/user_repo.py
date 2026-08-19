from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.models.user import User


def get_user_by_username(
    db: Session,
    username: str
):
    try:
        return (
            db.query(User)
            .filter(User.username == username)
            .first()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


def get_user_by_email(
    db: Session,
    email: str
):
    try:
        return (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

    except SQLAlchemyError:
        db.rollback()
        raise


def create_user(
    db: Session,
    username: str,
    email: str,
    hashed_password: str
):
    try:
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return user

    except SQLAlchemyError:
        db.rollback()
        raise