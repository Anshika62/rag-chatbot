from fastapi import APIRouter
from app.api.auth import router as auth_router
from app.api.files import router as files_router
from app.api.conversation import router as conversation_router
from app.api.documents import router as documents_router


router = APIRouter()

router.include_router(auth_router)
router.include_router(files_router)
router.include_router(conversation_router)
router.include_router(documents_router)