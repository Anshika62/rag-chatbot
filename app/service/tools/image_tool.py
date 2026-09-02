import io
import logging
import os
from typing import Any

from PIL import Image
import pillow_avif  # noqa: F401  (registers AVIF support with Pillow)

import requests
from langchain_core.tools import tool

from app.repository import document_repo
from app.core.database import SessionLocal


logger = logging.getLogger(__name__)


# Local upload directory convention. Kept as a plain literal (matching
# the existing convention already used in search_kb.py) rather than
# importing it from doc_service.py, since doc_service.py imports this
# module (generate_image_caption) — importing back would create a
# circular import.
UPLOAD_DIR = "Uploads"


# ============================================================
# CLOUDFLARE WORKERS AI VISION CLIENT
#
# Replaces the previous Gemini-based captioning. Public function
# signatures below (generate_image_caption, create_image_tool,
# create_document_image_analysis_tool) are UNCHANGED — doc_service.py
# and conversation_tool.py need zero edits.
#
# API used: Cloudflare Workers AI REST API
#     POST https://api.cloudflare.com/client/v4/accounts/
#          {account_id}/ai/run/@cf/llava-hf/llava-1.5-7b-hf
#
# Required environment variables:
#     CLOUDFLARE_ACCOUNT_ID
#     CLOUDFLARE_API_TOKEN
# ============================================================

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

VISION_MODEL = "@cf/google/gemma-4-26b-a4b-it"

VISION_ENDPOINT = (
    "https://api.cloudflare.com/client/v4/accounts/"
    f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{VISION_MODEL}"
)

VISION_HEADERS = {
    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
    "Content-Type": "application/json",
}


# ============================================================
# QUOTA / RATE-LIMIT EXCEPTION
#
# Kept as its own class (same name/behavior as before) so
# doc_service.py's except ImageCaptionQuotaExceededError blocks
# keep working unchanged. Cloudflare's free-tier limit is a
# shared daily "Neurons" budget rather than Gemini's per-model
# daily request quota, but it fails the same way from the
# caller's point of view (HTTP 429 / rate-limit error), so the
# same stop-early handling applies.
# ============================================================


class ImageCaptionQuotaExceededError(RuntimeError):
    """Raised when the image-captioning provider's quota is exhausted."""


def _is_quota_exhausted(exc: Exception) -> bool:
    message = str(exc)
    return (
        "429" in message
        or "rate limit" in message.lower()
        or "quota" in message.lower()
    )


# ============================================================
# UNIVERSAL IMAGE LOADING — WITH AUTOMATIC RESIZE/COMPRESS
#
# FIX (2026-09-01): Cloudflare Workers AI's llava-1.5-7b-hf model
# rejects payloads over a certain size with HTTP 413. High-res
# phone photos / AVIF images re-encoded as full-size PNG can
# easily exceed that limit.
#
# This now happens AUTOMATICALLY for every image, every time —
# nothing manual needed per-upload:
#   1. Downscale to a max dimension (longest side <= MAX_DIMENSION)
#      while preserving aspect ratio.
#   2. Encode as JPEG (much smaller than PNG for photos) instead
#      of PNG.
#   3. If still above the safe byte budget, progressively lower
#      JPEG quality (and shrink further if needed) until it fits.
#
# This makes the pipeline size-agnostic: any image Pillow can
# decode, of any original resolution, will be sent as a payload
# safely under Cloudflare's limit.
# ============================================================

# Cloudflare Workers AI doesn't publish an exact byte limit for
# this endpoint, but 413s have been observed well under 10MB.
# Keeping meaningful headroom below that.
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024  # 4 MB safety budget
MAX_DIMENSION = 1568  # longest side, in pixels — plenty for vision/OCR-style captioning
MIN_DIMENSION = 512   # don't shrink below this even under heavy compression
JPEG_QUALITY_START = 85
JPEG_QUALITY_FLOOR = 40


def _load_image_as_png_bytes(image_path: str) -> bytes:
    """
    Load an image from disk and return encoded bytes that are safe
    to send to the Cloudflare Workers AI vision endpoint.

    Despite the name (kept for backward compatibility with existing
    callers), this now encodes as JPEG when downscaling/compression
    is needed, since JPEG is dramatically smaller than PNG for
    photographic content and Cloudflare's vision model doesn't care
    about the source format. If the image already comfortably fits
    within the byte budget as PNG, it's still returned as PNG.
    """

    with Image.open(image_path) as img:

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # First, try the original size as PNG (cheap path for
        # already-small images — no quality loss).
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        if len(png_bytes) <= MAX_PAYLOAD_BYTES:
            return png_bytes

        # Too big — switch to the resize/compress path.
        working = img.convert("RGB")  # JPEG has no alpha channel

        width, height = working.size
        longest_side = max(width, height)

        if longest_side > MAX_DIMENSION:
            scale = MAX_DIMENSION / float(longest_side)
            new_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
            working = working.resize(new_size, Image.LANCZOS)

        quality = JPEG_QUALITY_START

        while True:
            buffer = io.BytesIO()
            working.save(buffer, format="JPEG", quality=quality, optimize=True)
            jpeg_bytes = buffer.getvalue()

            if len(jpeg_bytes) <= MAX_PAYLOAD_BYTES:
                return jpeg_bytes

            # Still too big: first try lowering quality...
            if quality > JPEG_QUALITY_FLOOR:
                quality -= 15
                continue

            # ...then, if quality is already at the floor, shrink
            # dimensions further and reset quality.
            width, height = working.size
            longest_side = max(width, height)

            if longest_side <= MIN_DIMENSION:
                # Can't shrink further without destroying the image;
                # return the best-effort smallest version we have.
                return jpeg_bytes

            scale = 0.75
            new_size = (
                max(MIN_DIMENSION, int(width * scale)),
                max(MIN_DIMENSION, int(height * scale)),
            )
            working = working.resize(new_size, Image.LANCZOS)
            quality = JPEG_QUALITY_START


DEFAULT_CAPTION_PROMPT = (
    "Describe this image in detail so it can be used as "
    "searchable knowledge-base content. Include any visible "
    "text, numbers, charts, diagrams, tables, objects, or "
    "people present in the image. Be factual and specific."
)


def _run_vision_model(
    image_bytes: bytes,
    prompt: str,
    max_tokens: int = 1024,
) -> str:
    """
    Shared low-level call to Cloudflare Workers AI Gemma 4
    vision model.

    Raises:
        ImageCaptionQuotaExceededError:
            On HTTP 429 / quota / rate-limit errors.
        RuntimeError:
            On other Cloudflare/API errors.
    """

    import base64

    try:
        # Convert image bytes to a data URL.
        # Gemma 4 vision accepts the image as base64 image data.
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{image_base64}"

        response = requests.post(
            VISION_ENDPOINT,
            headers=VISION_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url,
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
            timeout=60,
        )

        # ---------------------------------------------------------
        # RATE LIMIT / QUOTA
        # ---------------------------------------------------------
        if response.status_code == 429:
            raise ImageCaptionQuotaExceededError(
                "Cloudflare Workers AI rate limit / "
                "daily Neurons quota exceeded."
            )

        # ---------------------------------------------------------
        # PAYLOAD TOO LARGE
        # ---------------------------------------------------------
        if response.status_code == 413:
            raise RuntimeError(
                "Cloudflare Workers AI rejected the image as "
                "too large (413) even after resize/compress. "
                "Consider lowering MAX_PAYLOAD_BYTES."
            )

        # ---------------------------------------------------------
        # OTHER HTTP ERRORS
        # ---------------------------------------------------------
        response.raise_for_status()

        payload = response.json()

        # ---------------------------------------------------------
        # CLOUDFLARE ENVELOPE ERRORS
        # ---------------------------------------------------------
        if not payload.get("success", True):

            errors = payload.get("errors") or []

            error_text = str(errors)

            if _is_quota_exhausted(
                RuntimeError(error_text)
            ):
                raise ImageCaptionQuotaExceededError(
                    f"Cloudflare Workers AI quota exhausted: "
                    f"{errors}"
                )

            raise RuntimeError(
                f"Cloudflare Workers AI returned an error: "
                f"{errors}"
            )

        # ---------------------------------------------------------
        # RESPONSE PARSING
        #
        # Chat-completion style response:
        #
        # result -> choices -> message -> content
        # ---------------------------------------------------------
        result = payload.get("result") or {}

        choices = result.get("choices") or []

        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content") or ""

            if isinstance(text, str):
                return text.strip()

        # ---------------------------------------------------------
        # FALLBACK FOR OTHER RESPONSE SHAPES
        # ---------------------------------------------------------
        text = (
            result.get("response")
            or result.get("description")
            or ""
        )

        if isinstance(text, str):
            return text.strip()

        return ""

    except ImageCaptionQuotaExceededError:
        raise

    except requests.RequestException as exc:

        if _is_quota_exhausted(exc):
            raise ImageCaptionQuotaExceededError(
                f"Cloudflare Workers AI quota exhausted: {exc}"
            ) from exc

        raise RuntimeError(
            f"Cloudflare Workers AI request failed: {exc}"
        ) from exc

def generate_image_caption(
    image_path: str,
    prompt: str = DEFAULT_CAPTION_PROMPT,
) -> str:
    """
    Generate a detailed text description of an image using
    Cloudflare Workers AI vision (@cf/llava-hf/llava-1.5-7b-hf).
    Used to make image content (standalone uploads or images
    embedded inside PDFs) searchable inside the RAG knowledge
    base.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        image_bytes = _load_image_as_png_bytes(image_path)

        return _run_vision_model(
            image_bytes=image_bytes,
            prompt=prompt,
        ) or ""

    except ImageCaptionQuotaExceededError:

        # Don't log a full traceback for every image — the
        # caller (doc_service) logs ONE summary warning and
        # stops trying further images for this run.
        logger.warning(
            "IMAGE CAPTION QUOTA EXCEEDED: image_path=%s",
            image_path,
        )

        raise

    except Exception as exc:

        logger.exception(
            "IMAGE CAPTION ERROR: image_path=%s error=%s",
            image_path,
            str(exc),
        )

        raise RuntimeError(
            f"Unable to caption image: {exc}"
        ) from exc


# ============================================================
# CHAT-ATTACHED IMAGE TOOL (single-turn, used by LangChain)
# ============================================================


def create_image_tool(
    image_paths: list[str],
):
    image_paths = [
        path
        for path in image_paths
        if path and os.path.exists(path)
    ]

    @tool
    def analyze_image(
        question: str,
    ) -> dict[str, Any]:
        """
        Analyze the uploaded image or images using Cloudflare
        Workers AI vision.
        """

        if not question or not question.strip():
            raise ValueError(
                "Image question cannot be empty."
            )

        if not image_paths:
            return {
                "success": False,
                "analysis": "No image is available for analysis.",
            }

        try:

            # Cloudflare's llava model takes a single image per
            # request, so with multiple attached images we ask
            # about each one and combine the answers.
            analyses = []

            for image_path in image_paths:

                image_bytes = _load_image_as_png_bytes(
                    image_path
                )

                text = _run_vision_model(
                    image_bytes=image_bytes,
                    prompt=question.strip(),
                )

                analyses.append(text)

            return {
                "success": True,
                "analysis": "\n\n---\n\n".join(analyses),
            }

        except ImageCaptionQuotaExceededError as exc:

            logger.warning(
                "IMAGE ANALYSIS QUOTA EXCEEDED: error=%s",
                str(exc),
            )

            return {
                "success": False,
                "analysis": (
                    "Image analysis quota is currently "
                    "exhausted. Please try again later."
                ),
            }

        except Exception as exc:

            logger.exception(
                "IMAGE ANALYSIS ERROR: error=%s",
                str(exc),
            )

            raise RuntimeError(
                f"Unable to analyze image: {exc}"
            ) from exc

    return analyze_image


# ============================================================
# PREVIOUSLY-UPLOADED DOCUMENT IMAGE TOOL
#
# analyze_image (above) only covers images attached directly to
# the CURRENT chat message. It has no way to look up an image
# that was extracted from a PDF/document during a PAST upload.
#
# search_knowledge_base can locate such an image (by content_type
# or semantic match) and returns its document_id + cached
# caption/OCR text, but that cached text was generated once at
# ingestion time with a generic prompt — it cannot answer a
# specific question about the image. This tool closes that gap:
# given a document_id already surfaced by search_knowledge_base,
# it loads the ACTUAL image bytes from disk and asks the vision
# model the user's specific question.
# ============================================================


def create_document_image_analysis_tool(
    user_id: str,
    conversation_id: str | None,
):
    user_id = str(user_id)

    conversation_id = (
        str(conversation_id)
        if conversation_id
        else None
    )

    @tool
    def analyze_document_image(
        document_id: str,
        question: str,
    ) -> dict[str, Any]:
        """
        Analyze a SPECIFIC image that was previously extracted from
        an uploaded document/PDF (or uploaded standalone), using
        Cloudflare Workers AI vision, to answer a specific question
        about it.

        Use this — instead of relying only on the cached
        description text returned by search_knowledge_base —
        whenever the user asks to interpret, explain, or describe
        what an already-uploaded image, diagram, chart, table, or
        figure actually shows. Examples: "Explain the architecture
        diagram on page 4", "What does this chart show?", "Which
        components are shown in this diagram?".

        Typical flow: call search_knowledge_base first (with
        content_type="image") to find the relevant image and its
        document_id, then call this tool with that document_id and
        the user's specific question.

        Args:
            document_id: The document_id of the image, exactly as
                returned by a previous search_knowledge_base call.
                Never guess or invent a document_id.
            question: The specific question to ask about the image.
        """

        if not document_id or not document_id.strip():

            raise ValueError(
                "document_id cannot be empty."
            )

        if not question or not question.strip():

            raise ValueError(
                "Image question cannot be empty."
            )

        document_id = document_id.strip()

        db = SessionLocal()

        try:

            image_doc = (
                document_repo.get_accessible_document_by_id(
                    db=db,
                    doc_id=document_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            )

            if not image_doc:

                return {
                    "success": False,
                    "analysis": (
                        "That image document was not found, or "
                        "is not accessible from this conversation."
                    ),
                }

            if (
                not image_doc.mime_type
                or not image_doc.mime_type.startswith("image/")
            ):

                return {
                    "success": False,
                    "analysis": (
                        "The referenced document is not an image."
                    ),
                }

            image_path = os.path.join(
                UPLOAD_DIR,
                f"{image_doc.id}_{image_doc.file_name}",
            )

            if not os.path.exists(image_path):

                logger.warning(
                    "DOCUMENT IMAGE ANALYSIS: file missing on "
                    "disk for document_id=%s path=%s",
                    document_id,
                    image_path,
                )

                return {
                    "success": False,
                    "analysis": (
                        "The image file could not be found on "
                        "disk."
                    ),
                }

            try:

                image_bytes = _load_image_as_png_bytes(
                    image_path
                )

                analysis_text = _run_vision_model(
                    image_bytes=image_bytes,
                    prompt=question.strip(),
                )

            except ImageCaptionQuotaExceededError:

                logger.warning(
                    "DOCUMENT IMAGE ANALYSIS QUOTA "
                    "EXCEEDED: document_id=%s",
                    document_id,
                )

                return {
                    "success": False,
                    "analysis": (
                        "Image analysis quota is currently "
                        "exhausted. Please try again later."
                    ),
                }

            return {
                "success": True,
                "analysis": analysis_text or "",
                "document_id": str(image_doc.id),
                "parent_document_id": str(
                    image_doc.parent_id
                    or image_doc.id
                ),
                "filename": image_doc.file_name,
                "content_type": image_doc.mime_type,
            }

        except ValueError:

            raise

        except Exception as exc:

            logger.exception(
                "DOCUMENT IMAGE ANALYSIS ERROR: "
                "document_id=%s error=%s",
                document_id,
                str(exc),
            )

            raise RuntimeError(
                f"Unable to analyze document image: {exc}"
            ) from exc

        finally:

            db.close()

    return analyze_document_image