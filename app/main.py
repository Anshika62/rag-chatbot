import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.response import setup_exception_handlers


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


app = FastAPI()


# ============================================================
# GENERATED IMAGES
#
# Generated images are stored by image_generation_tool.py
# inside ./generated-images.
#
# This exposes them through:
# /generated-images/<filename>
#
# Example:
# /generated-images/abc123.png
# ============================================================

GENERATED_IMAGE_DIR = Path("generated-images")
GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/generated-images",
    StaticFiles(directory=GENERATED_IMAGE_DIR),
    name="generated-images",
)


# ============================================================
# CORS
# ============================================================

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://your-frontend-domain.com",
    "https://chat-bot-three-topaz.vercel.app",
    "https://chat-bot.bhawsarvinayak55.workers.dev/",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# EXCEPTION HANDLERS
# ============================================================

setup_exception_handlers(app)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(router)