import re
from collections import Counter
from typing import Any, Optional

import numpy as np

from app.infra.logging import logger


class HybridSearch:
    def __init__(self, vector_store, alpha: float = 0.5):
        self.vector_store = vector_store
        self.alpha = alpha
        self._keyword_index: dict[str, list[int]] = {}
        self._chunks: list[dict[str, Any]] = []

    def build_keyword_index(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._keyword_index.clear()
        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "").lower()
            tokens = re.findall(r"\b[a-zA-Z]{4,}\b", text)
            for token in set(tokens):
                if token not in self._keyword_index:
                    self._keyword_index[token] = []
                self._keyword_index[token].append(idx)
        logger.info("Keyword index built with %d terms", len(self._keyword_index))

    def save_index(self, path: str) -> None:
        import pickle
        data = {"chunks": self._chunks, "keyword_index": self._keyword_index}
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Keyword index saved to %s (%d terms, %d chunks)", path, len(self._keyword_index), len(self._chunks))

    def load_index(self, path: str) -> bool:
        import pickle, os
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data["chunks"]
            self._keyword_index = data["keyword_index"]
            logger.info("Keyword index loaded from %s (%d terms, %d chunks)", path, len(self._keyword_index), len(self._chunks))
            return True
        except Exception as e:
            logger.warning("Failed to load keyword index: %s", e)
            return False

    def search(self, query: str, query_emb: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        vector_results = self.vector_store.search(query_emb, top_k=top_k * 2)
        keyword_results = self._keyword_search(query, top_k=top_k)

        combined = self._fuse(vector_results, keyword_results, query)
        return combined[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_lower = query.lower()
        score_map: dict[int, float] = Counter()

        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", query_lower)
        for token in tokens:
            if token in self._keyword_index:
                for idx in self._keyword_index[token]:
                    score_map[idx] += len(token) * 1.0

        legal_refs = re.findall(r"(?:article|section|chapter)\s+\d+", query_lower)
        for ref in legal_refs:
            norm = ref.strip()
            for keyword, indices in self._keyword_index.items():
                if norm in keyword or keyword in norm:
                    for idx in indices:
                        score_map[idx] += 5.0

        results = []
        for idx, score in score_map.most_common(top_k):
            if idx < len(self._chunks):
                chunk = dict(self._chunks[idx])
                chunk["keyword_score"] = score
                chunk["search_type"] = "keyword"
                results.append(chunk)
        return results

    def _fuse(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        query: str,
    ) -> list[dict[str, Any]]:
        fusion: dict[str, dict] = {}

        for rank, r in enumerate(vector_results):
            key = r.get("id", r.get("text", ""))
            if key not in fusion:
                fusion[key] = dict(r)
                fusion[key]["rank_sum"] = 0.0
                fusion[key]["from_vector"] = True
            fusion[key]["rank_sum"] += 1.0 / (rank + 1)
            fusion[key]["vector_score"] = r.get("score", 0.0)

        for rank, r in enumerate(keyword_results):
            key = r.get("id", r.get("text", ""))
            if key not in fusion:
                fusion[key] = dict(r)
                fusion[key]["rank_sum"] = 0.0
                fusion[key]["from_vector"] = False
            fusion[key]["rank_sum"] += 1.0 / (rank + 1)
            fusion[key]["keyword_score"] = r.get("keyword_score", 0.0)

        max_kw = max((r.get("keyword_score", 0) for r in keyword_results), default=0)
        for key, item in fusion.items():
            vec_score = item.get("vector_score", 0.0)
            kw_score = item.get("keyword_score", 0.0)
            norm_kw = (kw_score / max_kw) if max_kw > 0 and kw_score > 0 else 0.0
            item["combined_score"] = self.alpha * vec_score + (1 - self.alpha) * norm_kw

        sorted_items = sorted(
            fusion.values(),
            key=lambda x: x.get("combined_score", 0),
            reverse=True,
        )
        return sorted_items
