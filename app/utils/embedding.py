from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(
                model_name,
                device="cpu"
            )

        except Exception:
            raise RuntimeError(
                "Unable to load embedding model"
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

            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                batch_size=1
            )

            return np.asarray(embeddings)

        except ValueError:
            raise

        except Exception:
            raise RuntimeError(
                "Unable to generate embeddings"
            )