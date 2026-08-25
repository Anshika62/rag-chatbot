import logging

from fastapi import FastAPI
from app.api.routes import router
from app.core.response import setup_exception_handlers

from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# LOGGING
#
# Without this, the root logger has no handler attached, so
# every logger.info/warning/exception(...) call across the app
# (doc_service, rag_service, image_tool, search_kb, etc.) is
# silently dropped — nothing shows up in the terminal, even for
# real failures. This just wires up a basic handler; it doesn't
# touch any of the actual logger.* call sites.
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI()

origins = [
    "http://localhost:3000",   # React/Next frontend
    "http://localhost:5173",   # Vite frontend
    "https://your-frontend-domain.com",  # deployed frontend
    "https://chat-bot-three-topaz.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


setup_exception_handlers(app)

app.include_router(router)