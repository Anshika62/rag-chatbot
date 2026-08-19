from typing import Any
from pydantic import BaseModel, EmailStr


class APIResponse(BaseModel):
    success: bool
    status_code: int
    message: str
    data: Any = None
    error_code: str | None = None


class SignupRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class SignupData(BaseModel):
    username: str
    email: EmailStr

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenData(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class CurrentUserData(BaseModel):
    username: str