from fastapi import status
from sqlalchemy.orm import Session

from app.core.response import AppException  
from app.repository.user_repo import (
    get_user_by_username,
    get_user_by_email,
    create_user
)

from app.utils.password import (
    hash_password,
    verify_password
)

def signup_user(
    db: Session,
    username: str,
    email: str,
    password: str
):
    try:
        existing_user = get_user_by_username(
            db,
            username
        )

        if existing_user:
            raise AppException(
                status_code=status.HTTP_409_CONFLICT,
                message="Username already exists",
                error_code="USERNAME_ALREADY_EXISTS"
            )

        existing_email = get_user_by_email(
            db,
            email
        )

        if existing_email:
            raise AppException(
                status_code=status.HTTP_409_CONFLICT,
                message="Email already registered",
                error_code="EMAIL_ALREADY_EXISTS"
            )

        hashed_password = hash_password(password)

        user = create_user(
            db,
            username,
            email,
            hashed_password
        )

        return user

    except AppException:
        raise

    except Exception:
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Unable to create user",
            error_code="USER_CREATION_FAILED"
        )


def authenticate_user(
    db: Session,
    email: str,
    password: str
):
    try:
        user = get_user_by_email(
            db,
            email
        )

        if not user:
            raise AppException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                message="Invalid email or password",
                error_code="AUTH_INVALID_CREDENTIALS"
            )

        if not verify_password(
            password,
            user.hashed_password
        ):
            raise AppException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                message="Invalid email or password",
                error_code="AUTH_INVALID_CREDENTIALS"
            )

        return user

    except AppException:
        raise

    except Exception:
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Unable to authenticate user",
            error_code="AUTH_FAILED"
        )