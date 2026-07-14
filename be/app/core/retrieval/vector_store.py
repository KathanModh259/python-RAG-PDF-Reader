from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.infra.logging import logger


class VectorStore:
    def __init__(self, store_type: str = "chroma", persist_dir: Optional[Path] = None):
        self.store_type = store_type
        self.persist_dir = persist_dir
        self._collection = None
        self._client = None

    def initialize(self, collection_name: str = "legal_docs") -> None:
        if self.store_type == "chroma":
            self._init_chroma(collection_name)
        else:
            self._init_faiss(collection_name)
        logger.info("VectorStore initialized: %s (%s)", collection_name, self.store_type)

    def _init_chroma(self, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb is required for vector storage")
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _init_faiss(self, collection_name: str) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss is required for vector storage")
        self._faiss_index = None
        self._faiss_id_map = {}
        self._faiss_next_id = 0
        self._faiss_dim = None

    def add(self, embeddings: np.ndarray, texts: list[str], metadatas: list[dict]) -> None:
        if self.store_type == "chroma":
            self._add_chroma(embeddings, texts, metadatas)
        else:
            self._add_faiss(embeddings, texts, metadatas)

    def _add_chroma(self, embeddings: np.ndarray, texts: list[str], metadatas: list[dict]) -> None:
        import hashlib
        source = metadatas[0].get("source", "unknown") if metadatas else "unknown"
        ids = [
            hashlib.md5(f"{source}:{i}:{text[:64]}".encode("utf-8", errors="replace")).hexdigest()[:16]
            for i, text in enumerate(texts)
        ]
        existing = set(self._collection.get()["ids"])
        new_data = [
            (ids[i], embeddings[i].tolist(), texts[i], metadatas[i])
            for i in range(len(texts)) if ids[i] not in existing
        ]
        if not new_data:
            return
        new_ids, new_embs, new_texts, new_metas = zip(*new_data)
        self._collection.add(
            embeddings=list(new_embs),
            documents=list(new_texts),
            metadatas=list(new_metas),
            ids=list(new_ids),
        )
        logger.debug("Added %d new chunks (skipped %d existing)", len(new_ids), len(texts) - len(new_ids))

    def _add_faiss(self, embeddings: np.ndarray, texts: list[str], metadatas: list[dict]) -> None:
        import faiss
        dim = embeddings.shape[1]
        if self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_dim = dim
        self._faiss_index.add(embeddings)
        for i in range(len(texts)):
            self._faiss_id_map[self._faiss_next_id] = {
                "text": texts[i],
                "metadata": metadatas[i],
            }
            self._faiss_next_id += 1

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        if self.store_type == "chroma":
            return self._search_chroma(query_emb, top_k)
        return self._search_faiss(query_emb, top_k)

    def _search_chroma(self, query_emb: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        results = self._collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=min(top_k, self._collection.count()),
        )
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": results["distances"][0][i] if results["distances"] else 0.0,
                "id": results["ids"][0][i],
            })
        return items

    def _search_faiss(self, query_emb: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if self._faiss_index is None:
            return []
        scores, indices = self._faiss_index.search(query_emb, min(top_k, self._faiss_index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self._faiss_id_map:
                item = self._faiss_id_map[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results

    def search_by_text(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        import re
        refs = re.findall(r"(?:article|section|art|sec|chapter)\s+(\d+[A-Za-z]*)", query, re.IGNORECASE)
        if not refs:
            return []
        ref = refs[0]
        try:
            all_data = self._collection.get()
            matched = []
            pat = re.compile(r'(?:^|[ \n(])' + re.escape(ref) + r'[\.\)\s:]')
            for i in range(len(all_data["ids"])):
                txt = all_data["documents"][i]
                meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
                src = meta.get("source", "")
                if pat.search(txt):
                    matched.append({
                        "text": txt,
                        "metadata": meta,
                        "score": 0.98,
                        "id": all_data["ids"][i],
                        "search_type": "literal",
                        "source_priority": 0 if "constitution" in src else 1,
                    })
            matched.sort(key=lambda x: (x.get("source_priority", 1), -x.get("score", 0)))
            return matched[:top_k]
        except Exception as e:
            return []

    @property
    def count(self) -> int:
        if self.store_type == "chroma" and self._collection:
            return self._collection.count()
        if self._faiss_index is not None:
            return self._faiss_index.ntotal
        return 0
