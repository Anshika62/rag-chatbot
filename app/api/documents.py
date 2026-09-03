from typing import Optional
import logging
import os
import json

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
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.response import success_response
from app.core.dependency import (
    get_current_user,
    get_current_document,
)

from app.models.document import Document

from app.repository import document_repo
from app.repository.conversation_repo import create_conversation

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
# UPLOAD DOCUMENT — SSE STREAMED
# ============================================================


@router.post(
    "/upload",
)
def upload_document(
    file: UploadFile = File(...),
    parent_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    is_conversation_new: bool = Form(False),
    is_chat: bool = Form(False),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Upload a document using Server-Sent Events.

    Scope rules:

    1. is_new_conv=True
       -> Create a new conversation.
       -> Document is attached to that new conversation.

    2. is_new_conv=False + conversation_id provided
       -> Conversation-scoped document.
       -> Accessible only from that conversation.

    3. is_new_conv=False + conversation_id=None
       -> Global document.
       -> Accessible from all conversations.

    parent_id is independent from conversation scope, except that
    conversation-scoped documents cannot be placed inside a
    global folder.

    is_chat:
        True when this upload originates from WITHIN an existing
        chat/conversation window (as opposed to the global
        document library). This is a HARD SCOPE GUARD, not just a
        UI hint: a chat-context upload must always resolve to a
        conversation-scoped document, never to a global one. If
        it were allowed to silently fall through to
        conversation_id=None, that document would become globally
        accessible and its content would leak into (and be
        answerable from) every other conversation — exactly the
        bug this guard prevents. It's only skipped when
        is_conversation_new=True, since in that case a fresh
        conversation_id is about to be created and attached below
        anyway.

    SSE events emitted by the service:

        uploading
        processing
        ready

    or:

        failed
    """

    # --------------------------------------------------------
    # NORMALIZE EMPTY FORM VALUES
    # --------------------------------------------------------

    if parent_id is not None:
        parent_id = parent_id.strip() or None

    if conversation_id is not None:
        conversation_id = conversation_id.strip() or None

    # --------------------------------------------------------
    # GUARD: chat-context uploads must never silently become
    # global documents.
    #
    # If this upload is coming from within an existing chat
    # (is_chat=True) and it is NOT creating a brand new
    # conversation, a conversation_id MUST already be present.
    # Without this check, a missing/dropped conversation_id from
    # the frontend would silently fall through to scope rule #3
    # (global document) instead of failing loudly — which is
    # exactly how a conversation-specific document was ending up
    # answerable from every other conversation.
    # --------------------------------------------------------

    if is_chat and not is_conversation_new and not conversation_id:

        logger.warning(
            "DOCUMENT UPLOAD REJECTED | "
            "is_chat=True but no conversation_id provided | "
            "file=%r | user_id=%r",
            file.filename,
            user.id,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "conversation_id is required when uploading "
                "a document from an existing chat"
            ),
        )

    # --------------------------------------------------------
    # CREATE NEW CONVERSATION IF REQUESTED
    # --------------------------------------------------------

    if is_conversation_new:

        conversation = create_conversation(
            db=db,
            user_id=user.id,
            title=file.filename or "New Conversation",
        )

        conversation_id = str(conversation.id)

        logger.info(
            "NEW CONVERSATION CREATED FOR DOCUMENT UPLOAD | "
            "conversation_id=%s | file=%r | user_id=%r",
            conversation_id,
            file.filename,
            user.id,
        )

    # --------------------------------------------------------
    # DEBUG REQUEST
    # --------------------------------------------------------

    logger.info(
        "DOCUMENT UPLOAD REQUEST | "
        "file=%r | "
        "parent_id=%r | "
        "conversation_id=%r | "
        "is_conversation_new=%r | "
        "is_chat=%r | "
        "user_id=%r",
        file.filename,
        parent_id,
        conversation_id,
        is_conversation_new,
        is_chat,
        user.id,
    )

    # --------------------------------------------------------
    # VALIDATE BEFORE OPENING SSE STREAM
    # --------------------------------------------------------

    try:
        resolved_context = doc_service.resolve_upload_context(
            db=db,
            parent_id=parent_id,
            conversation_id=conversation_id,
            user_id=str(user.id),
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

    except Exception as exc:

        logger.exception(
            "DOCUMENT UPLOAD VALIDATION FAILED | "
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
            detail="Unable to validate document upload",
        ) from exc

    # --------------------------------------------------------
    # PARENT FOLDER NOT FOUND
    # --------------------------------------------------------

    if not resolved_context:

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

    resolved_parent_id, resolved_conversation_id = (
        resolved_context
    )

    # --------------------------------------------------------
    # SSE EVENT ENCODER
    # --------------------------------------------------------

    def event_stream():

        try:

            for event in doc_service.upload_document_stream_service(
                file=file,
                parent_id=resolved_parent_id,
                conversation_id=resolved_conversation_id,
                user_id=str(user.id),
            ):

                yield (
                    f"data: {json.dumps(event, default=str)}\n\n"
                )

        except Exception as exc:

            logger.exception(
                "DOCUMENT SSE STREAM FAILED | "
                "file=%r | user_id=%r",
                file.filename,
                user.id,
            )

            error_event = {
                "status": "failed",
                "message": "Document upload failed",
            }

            yield (
                f"data: {json.dumps(error_event)}\n\n"
            )

    # --------------------------------------------------------
    # STREAM RESPONSE
    # --------------------------------------------------------

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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