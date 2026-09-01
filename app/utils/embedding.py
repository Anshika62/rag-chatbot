from typing import List
import os
import time
import numpy as np
import requests


class EmbeddingManager:
    """
    Generates text embeddings using Cloudflare Workers AI
    (@cf/baai/bge-small-en-v1.5 — 384-dimensional, same dimension
    as the previous sentence-transformers/all-MiniLM-L6-v2 model,
    so no changes are needed anywhere else: doc_service.py,
    vector_store.py, and the Qdrant collection all keep working
    exactly as before).

    Public interface (generate_embedding: List[str] -> np.ndarray
    of shape (N, 384), L2-normalized) is UNCHANGED from the
    HuggingFace version. Every caller (rag_clients.py,
    doc_service.py, search_kb.py) needs zero changes.

    API used: Cloudflare Workers AI REST API
        POST https://api.cloudflare.com/client/v4/accounts/
             {account_id}/ai/run/@cf/baai/bge-small-en-v1.5

    Required environment variables:
        CLOUDFLARE_ACCOUNT_ID
        CLOUDFLARE_API_TOKEN
            (needs the "Workers AI" -> "Read" or "Edit" permission
            when you create the token in the Cloudflare dashboard)
    """

    def __init__(
        self,
        model_name: str = "@cf/baai/bge-small-en-v1.5",
    ):
        try:
            self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
            self.api_token = os.getenv("CLOUDFLARE_API_TOKEN")

            if not self.account_id:
                raise ValueError(
                    "CLOUDFLARE_ACCOUNT_ID environment variable "
                    "is not set"
                )

            if not self.api_token:
                raise ValueError(
                    "CLOUDFLARE_API_TOKEN environment variable "
                    "is not set"
                )

            self.model_name = model_name

            self.endpoint = (
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{self.account_id}/ai/run/{self.model_name}"
            )

            self.headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }

        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize Cloudflare Workers AI "
                f"embedding client: {exc}"
            ) from exc

    def generate_embedding(
        self,
        texts: List[str]
    ) -> np.ndarray:

        try:
            if not texts:
                raise ValueError(
                    "Text list cannot be empty"
                )

            cleaned_texts = [
                text.strip()
                for text in texts
                if text and text.strip()
            ]

            if not cleaned_texts:
                raise ValueError(
                    "Text list contains no valid text"
                )

            # ====================================================
            # SINGLE BATCHED CALL
            #
            # Same as the HuggingFace version: all chunks go in ONE
            # request. Cloudflare's bge-small-en-v1.5 accepts "text"
            # as a list of strings and returns embeddings in the
            # same order, one HTTP round-trip for the whole batch
            # instead of one call per chunk.
            # ====================================================

            result = None
            last_error = None

            for attempt in range(3):
                try:
                    response = requests.post(
                        self.endpoint,
                        headers=self.headers,
                        json={"text": cleaned_texts},
                        timeout=30,
                    )

                    response.raise_for_status()

                    payload = response.json()

                    if not payload.get("success"):
                        raise RuntimeError(
                            f"Cloudflare Workers AI returned an "
                            f"error: {payload.get('errors')}"
                        )

                    result = payload["result"]["data"]
                    break

                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))

            if result is None:
                raise RuntimeError(
                    f"Unable to generate embeddings after retries: {last_error}"
                ) from last_error

            embeddings = np.asarray(
                result,
                dtype=np.float32
            )

            # Defensive: normalize shape before proceeding so
            # downstream code always gets a consistent
            # (num_texts, 384) array, same as before.
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            if embeddings.ndim != 2:
                raise RuntimeError(
                    f"Unexpected embedding shape: "
                    f"{embeddings.shape}"
                )

            # bge-small-en-v1.5 produces 384-dimensional vectors,
            # same as the previous all-MiniLM-L6-v2 model — this
            # check (and the Qdrant collection dimension) stays
            # unchanged.
            if embeddings.shape[1] != 384:
                raise RuntimeError(
                    f"Unexpected embedding dimension: "
                    f"{embeddings.shape[1]}. "
                    f"Expected 384."
                )

            # Normalize each embedding row (unchanged from before).
            norms = np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True
            )

            norms[norms == 0] = 1

            embeddings = embeddings / norms

            return embeddings.astype(np.float32)

        except ValueError:
            raise

        except Exception as exc:
            raise RuntimeError(
                f"Unable to generate embeddings: {exc}"
            ) from exc