from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import verify_token

from app.repository.user_repo import get_user_by_email
from app.repository.conversation_repo import get_conversation
from app.repository.document_repo import get_owned_document_by_id



def get_current_user(
    db: Session = Depends(get_db),
    email: str = Depends(verify_token),
):
    """
    Return the currently authenticated user.
    """

    user = get_user_by_email(
        db=db,
        email=email,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user



def get_current_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Return the conversation only if it belongs
    to the currently authenticated user.
    """

    conversation = get_conversation(
        db=db,
        conversation_id=str(conversation_id),
        user_id=str(user.id),
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    return conversation



def get_current_document(
    document_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Return the document only if it belongs
    to the currently authenticated user.
    """

    document = get_owned_document_by_id(
        db=db,
        doc_id=str(document_id),
        user_id=str(user.id),
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document