from fastapi import APIRouter, Depends, status
from app.core.database import get_db
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    verify_refresh_token
)
from app.core.response import success_response
from app.service.auth_service import (
    signup_user,
    authenticate_user
)
from app.schemas.auth_schema import (
    SignupRequest,
    LoginRequest,
    APIResponse
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post(
    "/signup",
    response_model=APIResponse
)
def signup(
    data: SignupRequest,
    db=Depends(get_db)
):
    user = signup_user(
        db,
        data.username,
        data.email,
        data.password
    )

    return success_response(
        message="User created successfully",
        data={
            "username": user.username,
            "email": user.email
        },
        status_code=status.HTTP_201_CREATED
    )


@router.post("/login")
def login(
    data: LoginRequest,
    db=Depends(get_db)
):
    user = authenticate_user(
        db,
        data.email,
        data.password
    )

    access_token = create_access_token(
        {"sub": user.email}
    )

    refresh_token = create_refresh_token(
        {"sub": user.email}
    )

    return success_response(
        message="Login successful",
        data={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        },
        status_code=status.HTTP_200_OK
    )

@router.get(
    "/me",
    response_model=APIResponse
)
def get_current_user(
    email: str = Depends(verify_token)
):
    return success_response(
        message="Authentication successful",
        data={
            "email": email
        },
        status_code=status.HTTP_200_OK
    )

@router.post(
    "/refresh",
    response_model=APIResponse
)
def refresh_access_token(
    email: str = Depends(verify_refresh_token)
):
    new_access_token = create_access_token(
        {"sub": email}
    )

    return success_response(
        message="Access token refreshed successfully",
        data={
            "access_token": new_access_token,
            "token_type": "bearer"
        },
        status_code=status.HTTP_200_OK
    )