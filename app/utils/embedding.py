from typing import List
import os

import numpy as np
from huggingface_hub import InferenceClient


class EmbeddingManager:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        try:
            self.api_key = os.getenv("HF_TOKEN")

            if not self.api_key:
                raise ValueError(
                    "HF_TOKEN environment variable is not set"
                )

            self.model_name = model_name

            self.client = InferenceClient(
                token=self.api_key
            )

        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize Hugging Face embedding client: {exc}"
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

            embeddings = []

            for text in cleaned_texts:

                result = self.client.feature_extraction(
                    text,
                    model=self.model_name
                )

                embedding = np.asarray(
                    result,
                    dtype=np.float32
                )

                # Remove extra dimensions if returned
                if embedding.ndim > 1:
                    embedding = embedding.reshape(-1)

                if embedding.ndim != 1:
                    raise RuntimeError(
                        f"Unexpected embedding shape: "
                        f"{embedding.shape}"
                    )

                # all-MiniLM-L6-v2 produces 384-dimensional vectors
                if embedding.shape[0] != 384:
                    raise RuntimeError(
                        f"Unexpected embedding dimension: "
                        f"{embedding.shape[0]}. "
                        f"Expected 384."
                    )

                # Normalize embedding
                norm = np.linalg.norm(embedding)

                if norm > 0:
                    embedding = embedding / norm

                embeddings.append(embedding)

            return np.asarray(
                embeddings,
                dtype=np.float32
            )

        except ValueError:
            raise

        except Exception as exc:
            raise RuntimeError(
                f"Unable to generate embeddings: {exc}"
            ) from exc