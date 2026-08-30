import io
import logging
import os
from typing import Any

from PIL import Image
import pillow_avif  # noqa: F401  (registers AVIF support with Pillow)

from google import genai
from google.genai import types
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


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# QUOTA-SPECIFIC EXCEPTION
#
# Gemini's free tier enforces a PER-DAY request quota (not
# per-minute), so once it's hit, retrying the next image in the
# same run will just fail again immediately. Callers that process
# many images in a loop (e.g. a PDF with many embedded images)
# can catch this specifically to stop early instead of hammering
# the API — and logging — once per remaining image.
# ============================================================


class ImageCaptionQuotaExceededError(RuntimeError):
    """Raised when the image-captioning provider's quota is exhausted."""


def _is_quota_exhausted(exc: Exception) -> bool:
    message = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in message
        or "429" in message
        or "quota" in message.lower()
    )


# ============================================================
# UNIVERSAL IMAGE LOADING
#
# Instead of guessing mime type from file extension (fragile,
# breaks on avif/heic/bmp/etc.), we open the file with Pillow
# and always re-encode it as PNG before sending to Gemini.
# This makes the pipeline format-agnostic: any format Pillow
# can decode will work, without adding new cases every time.
# ============================================================


def _load_image_as_png_bytes(image_path: str) -> bytes:

    with Image.open(image_path) as img:

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        return buffer.getvalue()


DEFAULT_CAPTION_PROMPT = (
    "Describe this image in detail so it can be used as "
    "searchable knowledge-base content. Include any visible "
    "text, numbers, charts, diagrams, tables, objects, or "
    "people present in the image. Be factual and specific."
)


def generate_image_caption(
    image_path: str,
    prompt: str = DEFAULT_CAPTION_PROMPT,
) -> str:
    """
    Generate a detailed text description of an image using
    Gemini vision. Used to make image content (standalone
    uploads or images embedded inside PDFs) searchable inside
    the RAG knowledge base.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        image_bytes = _load_image_as_png_bytes(image_path)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                prompt,
            ],
        )

        return response.text or ""

    except Exception as exc:

        if _is_quota_exhausted(exc):

            # Don't log a full traceback for every image — the
            # caller (doc_service) logs ONE summary warning and
            # stops trying further images for this run.
            logger.warning(
                "IMAGE CAPTION QUOTA EXCEEDED: image_path=%s",
                image_path,
            )

            raise ImageCaptionQuotaExceededError(
                f"Image-captioning quota exhausted: {exc}"
            ) from exc

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
        Analyze the uploaded image or images using Gemini vision.
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
            contents = []

            for image_path in image_paths:

                image_bytes = _load_image_as_png_bytes(
                    image_path
                )

                contents.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    )
                )

            contents.append(question.strip())

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents,
            )

            return {
                "success": True,
                "analysis": response.text or "",
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
# it loads the ACTUAL image bytes from disk and asks Gemini
# vision the user's specific question.
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
        Gemini vision, to answer a specific question about it.

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

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png",
                        ),
                        question.strip(),
                    ],
                )

            except Exception as exc:

                if _is_quota_exhausted(exc):

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

                raise

            return {
                "success": True,
                "analysis": response.text or "",
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