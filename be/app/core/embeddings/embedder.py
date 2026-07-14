from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.infra.logging import logger


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._model: Optional[SentenceTransformer] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s (device: %s)", self.model_name, self.device)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        logger.info("Embedding model loaded (dim=%d)", self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        if not self._model:
            self.load()
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not self._model:
            self.load()
        if not texts:
            return np.array([], dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if not self._model:
            self.load()
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)
