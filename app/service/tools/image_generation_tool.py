import base64
import logging
import os
import uuid
from pathlib import Path

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


CLOUDFLARE_ACCOUNT_ID = (
    os.getenv("CLOUDFLARE_ACCOUNT_ID") or ""
).strip()

CLOUDFLARE_API_TOKEN = (
    os.getenv("CLOUDFLARE_API_TOKEN") or ""
).strip()

IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

GENERATED_IMAGE_DIR = Path("generated-images")
GENERATED_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def _get_image_endpoint() -> str:
    if not CLOUDFLARE_ACCOUNT_ID:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID is not configured."
        )

    return (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{IMAGE_MODEL}"
    )


def _generate_image(prompt: str) -> bytes:
    if not CLOUDFLARE_API_TOKEN:
        raise RuntimeError(
            "CLOUDFLARE_API_TOKEN is not configured."
        )

    if not prompt or not prompt.strip():
        raise ValueError(
            "Image prompt cannot be empty."
        )

    headers = {
        "Authorization": (
            f"Bearer {CLOUDFLARE_API_TOKEN}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt.strip(),
        "steps": 4,
    }

    response = requests.post(
        _get_image_endpoint(),
        headers=headers,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(
            f"Cloudflare image generation failed: {data}"
        )

    image_base64 = (
        data.get("result", {})
        .get("image")
    )

    if not image_base64:
        raise RuntimeError(
            "Cloudflare returned no generated image."
        )

    try:
        return base64.b64decode(
            image_base64
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to decode generated image."
        ) from exc


@tool
def generate_image(prompt: str) -> dict:
    """
    Generate a new image from a text description.

    Use this tool ONLY when the user explicitly asks
    to generate, create, draw, make, or produce a new image.

    Do NOT use this tool for:
    - analyzing uploaded images
    - analyzing document images
    - searching for existing images
    """

    clean_prompt = prompt.strip()

    logger.info(
        "IMAGE GENERATION START | prompt=%s",
        clean_prompt,
    )

    image_bytes = _generate_image(
        clean_prompt
    )

    filename = (
        f"{uuid.uuid4().hex}.png"
    )

    file_path = (
        GENERATED_IMAGE_DIR / filename
    )

    file_path.write_bytes(
        image_bytes
    )

    logger.info(
        "IMAGE GENERATION SUCCESS | "
        "filename=%s size=%s",
        filename,
        len(image_bytes),
    )

    return {
        "success": True,
        "filename": filename,
        "file_path": str(file_path),
        "url": (
            f"/generated-images/{filename}"
        ),
        "prompt": clean_prompt,
        "model": IMAGE_MODEL,
    }