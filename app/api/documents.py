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
import os
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import json
from fastapi.responses import FileResponse, StreamingResponse
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


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    parent_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    try:
        resolved = doc_service.resolve_upload_context(
            db=db,
            parent_id=parent_id,
            conversation_id=conversation_id,
            user_id=user.id,
        )

    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Parent folder not found",
        )

    resolved_parent_id, resolved_conversation_id = resolved

    def event_stream():
        for event in doc_service.upload_document_stream_service(
            file=file,
            parent_id=resolved_parent_id,
            conversation_id=resolved_conversation_id,
            user_id=user.id,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
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

@router.get("/{document_id}/file")
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
        media_type=document.mime_type or "application/octet-stream",
        filename=document.file_name,
    )