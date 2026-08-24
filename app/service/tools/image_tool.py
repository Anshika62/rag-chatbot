import io
import logging
import os
from typing import Any

from PIL import Image
import pillow_avif  # noqa: F401  (registers AVIF support with Pillow)

from google import genai
from google.genai import types
from langchain_core.tools import tool


logger = logging.getLogger(__name__)


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
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