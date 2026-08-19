from typing import List

import numpy as np
from huggingface_hub import InferenceClient


class EmbeddingManager:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        try:
            self.model_name = model_name
            self.client = InferenceClient(
                provider="hf-inference"
            )

        except Exception:
            raise RuntimeError(
                "Unable to initialize embedding service"
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

            result = self.client.feature_extraction(
                texts,
                model=self.model_name
            )

            embeddings = np.asarray(result)

            if embeddings.ndim == 3:
                embeddings = embeddings.mean(axis=1)

            elif embeddings.ndim == 2:
                if len(texts) == 1:
                    embeddings = embeddings.mean(axis=0, keepdims=True)

            elif embeddings.ndim != 2:
                raise RuntimeError(
                    "Unexpected embedding shape"
                )

            return embeddings.astype(np.float32)

        except ValueError:
            raise

        except Exception as exc:
            raise RuntimeError(
                f"Unable to generate embeddings: {exc}"
            )