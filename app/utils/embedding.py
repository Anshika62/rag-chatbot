from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        try:
            self.model = SentenceTransformer(
                model_name
            )

        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize embedding model: {exc}"
            )

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

            embeddings = self.model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            # Single text can return shape (384,)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            if embeddings.ndim != 2:
                raise RuntimeError(
                    f"Unexpected embedding shape: "
                    f"{embeddings.shape}"
                )

            # all-MiniLM-L6-v2 produces 384-dimensional vectors
            if embeddings.shape[1] != 384:
                raise RuntimeError(
                    f"Unexpected embedding dimension: "
                    f"{embeddings.shape[1]}. "
                    f"Expected 384."
                )

            return embeddings

        except ValueError:
            raise

        except Exception as exc:
            raise RuntimeError(
                f"Unable to generate embeddings: {exc}"
            )