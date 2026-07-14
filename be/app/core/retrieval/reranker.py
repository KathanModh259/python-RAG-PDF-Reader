from typing import Any

from app.infra.logging import logger


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker loaded: %s", self.model_name)
        except ImportError:
            logger.warning("CrossEncoder not available, using fallback scoring")

    def rerank(self, query: str, results: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not results:
            return []

        if self._model is None:
            self.load()

        if self._model is None:
            return results[:top_k]

        pairs = [(query, r.get("text", "")) for r in results]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.error("Reranking failed: %s, using original order", e)
            return results[:top_k]

        for r, score in zip(results, scores):
            r["rerank_score"] = float(score)

        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return results[:top_k]
