from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status
)

from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import verify_token

from app.repository.user_repo import get_user_by_email
from app.repository.conversation_repo import get_conversation

from app.service.rag_service import process_document


router = APIRouter()


@router.post("/files")
def upload_file(
    file: UploadFile = File(...),
    conversation_id: int | None = None,
    db: Session = Depends(get_db),
    email: str = Depends(verify_token)
):
    # 1. Get logged-in user using email
    user = get_user_by_email(
        db=db,
        email=email
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # 2. Conversation ID is required
    if conversation_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="conversation_id is required"
        )

    # 3. Verify conversation belongs to logged-in user
    conversation = get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=user.id
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # 4. Process document
    return process_document(
        file=file,
        db=db,
        conversation_id=conversation_id
    )