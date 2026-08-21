from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    UploadFile,
    File,
    Form,
    status,
)

from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import verify_token
from app.core.response import success_response

from app.repository.user_repo import get_user_by_email
from app.repository import document_repo

from app.schemas.document import (
    FolderCreate,
    DocumentUpdate,
    DocumentOut,
)

from app.service import doc_service


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


def _get_current_user(
    db: Session,
    email: str,
):
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


# ============================================================
# CREATE FOLDER
# ============================================================


@router.post(
    "/folder",
    response_model=DocumentOut,
)
def create_folder(
    payload: FolderCreate,
    db: Session = Depends(get_db),
    email: str = Depends(verify_token),
):
    user = _get_current_user(
        db,
        email,
    )

    folder = doc_service.create_folder_service(
        db=db,
        file_name=payload.file_name,
        parent_id=payload.parent_id,
        user_id=user.id,
    )

    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Parent folder not found",
        )

    return success_response(
        message="Folder created successfully",
        data=DocumentOut.model_validate(
            folder
        ).model_dump(mode="json"),
        status_code=status.HTTP_201_CREATED,
    )


# ============================================================
# UPLOAD DOCUMENT
# ============================================================


@router.post(
    "/upload",
    response_model=DocumentOut,
)
def upload_document(
    file: UploadFile = File(...),
    parent_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    email: str = Depends(verify_token),
):
    user = _get_current_user(
        db,
        email,
    )

    doc = doc_service.upload_document_service(
        db=db,
        file=file,
        parent_id=parent_id,
        conversation_id=conversation_id,
        user_id=user.id,
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Parent folder not found",
        )

    return success_response(
        message="Document uploaded successfully",
        data=DocumentOut.model_validate(
            doc
        ).model_dump(mode="json"),
        status_code=status.HTTP_201_CREATED,
    )


# ============================================================
# LIST DOCUMENTS
# ============================================================


@router.get("")
def list_documents(
    parent_id: Optional[str] = None,
    page: int = Query(
        1,
        ge=1,
        description="Page number, starts at 1",
    ),
    page_size: int = Query(
        20,
        ge=1,
        le=100,
        description="Items per page (max 100)",
    ),
    db: Session = Depends(get_db),
    email: str = Depends(verify_token),
):
    user = _get_current_user(
        db,
        email,
    )

    skip = (page - 1) * page_size

    items, total = document_repo.list_documents_by_parent(
        db,
        parent_id,
        user.id,
        skip=skip,
        limit=page_size,
    )

    total_pages = (
        (total + page_size - 1) // page_size
        if total
        else 0
    )

    return success_response(
        message="Documents fetched successfully",
        data={
            "items": [
                DocumentOut.model_validate(
                    item
                ).model_dump(mode="json")
                for item in items
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        },
    )


# ============================================================
# GET SINGLE DOCUMENT
# ============================================================


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    email: str = Depends(verify_token),
):
    user = _get_current_user(
        db,
        email,
    )

    doc = document_repo.get_owned_document_by_id(
        db,
        document_id,
        user.id,
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return success_response(
        message="Document fetched successfully",
        data=DocumentOut.model_validate(
            doc
        ).model_dump(mode="json"),
    )


# ============================================================
# UPDATE DOCUMENT METADATA
# ============================================================


@router.put(
    "/{document_id}",
    response_model=DocumentOut,
)
def update_document(
    document_id: str,
    payload: DocumentUpdate,
    db: Session = Depends(get_db),
    email: str = Depends(verify_token),
):
    user = _get_current_user(
        db,
        email,
    )

    doc = doc_service.update_document_service(
        db=db,
        doc_id=document_id,
        user_id=user.id,
        file_name=payload.file_name,
        mime_type=payload.mime_type,
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return success_response(
        message="Document updated successfully",
        data=DocumentOut.model_validate(
            doc
        ).model_dump(mode="json"),
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================


@router.delete(
    "/{document_id}",
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    email: str = Depends(verify_token),
):
    user = _get_current_user(
        db,
        email,
    )

    deleted = doc_service.delete_document_service(
        db=db,
        doc_id=document_id,
        user_id=user.id,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return success_response(
        message="Document deleted successfully",
        data=None,
    )