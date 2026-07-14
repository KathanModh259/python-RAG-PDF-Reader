import numpy as np
import pytest

from app.core.embeddings.embedder import EmbeddingModel


class TestEmbeddingModel:
    def test_load_and_encode(self):
        model = EmbeddingModel(model_name="BAAI/bge-small-en-v1.5", device="cpu")
        model.load()
        texts = ["This is a test", "Another document"]
        embeddings = model.encode(texts)
        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32

    def test_encode_query(self):
        model = EmbeddingModel(model_name="BAAI/bge-small-en-v1.5", device="cpu")
        model.load()
        emb = model.encode_query("What is the law?")
        assert emb.shape == (1, 384)

    def test_empty_input(self):
        model = EmbeddingModel(model_name="BAAI/bge-small-en-v1.5", device="cpu")
        model.load()
        embeddings = model.encode([])
        assert embeddings.size == 0
