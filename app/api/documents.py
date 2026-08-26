from typing import Optional
import logging
import os

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
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.response import success_response
from app.core.dependency import (
    get_current_user,
    get_current_document,
)

from app.models.document import Document

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

logger = logging.getLogger(__name__)


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
    user=Depends(get_current_user),
):
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
    user=Depends(get_current_user),
):
    """
    Upload a document.

    Scope rules:

    1. conversation_id is NULL
       -> Global document.
       -> Accessible from all conversations.

    2. conversation_id contains a valid conversation ID
       -> Conversation-scoped document.
       -> Accessible only from that conversation.

    parent_id is independent from conversation scope.
    """

    # --------------------------------------------------------
    # NORMALIZE EMPTY FORM VALUES
    # --------------------------------------------------------

    if parent_id is not None:
        parent_id = parent_id.strip() or None

    if conversation_id is not None:
        conversation_id = conversation_id.strip() or None

    # --------------------------------------------------------
    # DEBUG REQUEST
    # --------------------------------------------------------

    logger.info(
        "DOCUMENT UPLOAD REQUEST | "
        "file=%r | "
        "parent_id=%r | "
        "conversation_id=%r | "
        "user_id=%r",
        file.filename,
        parent_id,
        conversation_id,
        user.id,
    )

    try:
        # ----------------------------------------------------
        # SERVICE
        # ----------------------------------------------------

        doc = doc_service.upload_document_service(
            db=db,
            file=file,
            parent_id=parent_id,
            conversation_id=conversation_id,
            user_id=user.id,
        )

    except PermissionError as exc:

        logger.warning(
            "DOCUMENT UPLOAD FORBIDDEN | "
            "file=%r | "
            "parent_id=%r | "
            "conversation_id=%r | "
            "user_id=%r | "
            "error=%s",
            file.filename,
            parent_id,
            conversation_id,
            user.id,
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except ValueError as exc:

        logger.warning(
            "DOCUMENT UPLOAD BAD REQUEST | "
            "file=%r | "
            "parent_id=%r | "
            "conversation_id=%r | "
            "user_id=%r | "
            "error=%s",
            file.filename,
            parent_id,
            conversation_id,
            user.id,
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "DOCUMENT UPLOAD FAILED | "
            "file=%r | "
            "parent_id=%r | "
            "conversation_id=%r | "
            "user_id=%r",
            file.filename,
            parent_id,
            conversation_id,
            user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to upload document",
        ) from exc

    # --------------------------------------------------------
    # PARENT FOLDER NOT FOUND
    # --------------------------------------------------------

    if not doc:

        logger.warning(
            "DOCUMENT UPLOAD | Parent folder not found | "
            "file=%r | parent_id=%r | user_id=%r",
            file.filename,
            parent_id,
            user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Parent folder not found",
        )

    # --------------------------------------------------------
    # DEBUG CREATED DOCUMENT
    # --------------------------------------------------------

    logger.info(
        "DOCUMENT CREATED | "
        "document_id=%r | "
        "file=%r | "
        "parent_id=%r | "
        "conversation_id=%r | "
        "user_id=%r | "
        "status=%r",
        str(doc.id),
        doc.file_name,
        doc.parent_id,
        doc.conversation_id,
        doc.user_id,
        doc.status,
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

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
    user=Depends(get_current_user),
):
    skip = (page - 1) * page_size

    items, total = document_repo.list_documents_by_parent(
        db=db,
        parent_id=parent_id,
        user_id=user.id,
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
    document: Document = Depends(get_current_document),
):
    return success_response(
        message="Document fetched successfully",
        data=DocumentOut.model_validate(
            document
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
    document: Document = Depends(get_current_document),
    db: Session = Depends(get_db),
):
    doc = doc_service.update_document_service(
        db=db,
        doc=document,
        file_name=payload.file_name,
        mime_type=payload.mime_type,
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
    document: Document = Depends(get_current_document),
    db: Session = Depends(get_db),
):
    doc_service.delete_document_service(
        db=db,
        doc=document,
    )

    return success_response(
        message="Document deleted successfully",
        data=None,
    )


# ============================================================
# DOWNLOAD / VIEW DOCUMENT FILE
# ============================================================


@router.get(
    "/{document_id}/file",
)
def get_document_file(
    document: Document = Depends(get_current_document),
):
    file_path = os.path.join(
        "Uploads",
        f"{document.id}_{document.file_name}",
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk",
        )

    return FileResponse(
        file_path,
        media_type=(
            document.mime_type
            or "application/octet-stream"
        ),
        filename=document.file_name,
    )