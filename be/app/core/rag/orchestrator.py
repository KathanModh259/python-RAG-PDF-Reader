from typing import Any, Optional

import numpy as np

from app.core.embeddings.embedder import EmbeddingModel
from app.core.llm.runner import LLMRunner
from app.core.preprocessing.chunker import LegalChunker
from app.core.rag.prompts import make_qa_prompt
from app.core.retrieval.hybrid_search import HybridSearch
from app.core.retrieval.reranker import CrossEncoderReranker
from app.core.retrieval.vector_store import VectorStore
from app.infra.config import settings
from app.infra.logging import logger


class RAGOrchestrator:
    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        hybrid_search: HybridSearch,
        llm: LLMRunner,
        chunker: Optional[LegalChunker] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_store.initialize()
        self.hybrid_search = hybrid_search
        self.llm = llm
        self.chunker = chunker or LegalChunker()
        self.reranker = reranker
        self.top_k = top_k
        self._chunks: list[dict[str, Any]] = []
        self._rebuild_keyword_index()

    def index_document(self, text: str, source: str = "") -> int:
        chunks = self.chunker.chunk(text, source=source)
        if not chunks:
            logger.warning("No chunks generated from document")
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "heading": c.get("heading", ""),
                "type": c.get("type", ""),
                "source": c.get("source", source),
            }
            for c in chunks
        ]

        embeddings = self.embedder.encode(texts)
        self.vector_store.add(embeddings, texts, metadatas)
        self._chunks.extend(chunks)
        self.hybrid_search.build_keyword_index(self._chunks)

        logger.info("Indexed %d chunks from %s", len(chunks), source)
        return len(chunks)

    def _rebuild_keyword_index(self) -> None:
        try:
            import os
            persist_dir = getattr(self.vector_store, 'persist_dir', None)
            if persist_dir:
                cache_path = os.path.join(str(persist_dir), "keyword_index.pkl")
                if self.hybrid_search.load_index(cache_path):
                    return
            coll = getattr(self.vector_store, '_collection', None)
            if coll is None:
                return
            count = coll.count()
            if count == 0:
                return
            all_data = coll.get()
            self._chunks = [
                {
                    "text": all_data["documents"][i],
                    "metadata": (all_data["metadatas"][i] if all_data["metadatas"] else {}),
                    "id": all_data["ids"][i],
                }
                for i in range(len(all_data["ids"]))
            ]
            self.hybrid_search.build_keyword_index(self._chunks)
            if persist_dir:
                self.hybrid_search.save_index(cache_path)
        except Exception as e:
            pass

    def query(
        self,
        question: str,
        mode: str = "standard",
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        k = top_k or self.top_k

        literal_results = self.vector_store.search_by_text(question, top_k=3)
        used_literal = len(literal_results) > 0

        query_emb = self.embedder.encode_query(question)
        hybrid_k = k * 3
        results = self.hybrid_search.search(question, query_emb, top_k=hybrid_k)

        if used_literal:
            seen_ids = {r.get("id", "") for r in literal_results}
            for r in literal_results:
                r["combined_score"] = 1.0
            results = literal_results + [r for r in results if r.get("id", "") not in seen_ids]

        if not results:
            return {
                "answer": "No relevant documents found. Please index documents first.",
                "sources": [],
                "confidence": 0.0,
            }

        results = self._boost_keyword_matches(question, results)
        if self.reranker and settings.reranker_enabled:
            results = self.reranker.rerank(question, results, top_k=k)
        else:
            results = results[:k]

        context = self._format_context(results)
        answer = self.llm.generate(make_qa_prompt(question, context, mode=mode))
        answer = self._sanitize_citations(answer, results)

        if answer is None:
            answer = (
                "LLM is not available. Enable it in settings and download a model "
                "(python scripts/download_model.py).\n\n"
                "Relevant document excerpts:\n\n" + context
            )

        sources = [
            {
                "text": r.get("text", "")[:200],
                "heading": r.get("metadata", {}).get("heading", ""),
                "source": r.get("metadata", {}).get("source", ""),
                "score": r.get("combined_score", r.get("score", 0)),
            }
            for r in results[:5]
        ]

        confidence = min(
            sum(s["score"] for s in sources if s["score"]) / max(len(sources), 1),
            1.0,
        )

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

    def _sanitize_citations(self, answer: str, results: list[dict]) -> str:
        import re
        context_text = " ".join(r.get("text", "") for r in results).lower()
        cited_sections = set(re.findall(r'(?:Section|section|Sec|sec)\s+(\d+[A-Za-z]*)', answer))
        fake_sections = [s for s in cited_sections if s not in context_text]
        if fake_sections:
            for sec in fake_sections:
                answer = re.sub(
                    r'(?:Section|section|Sec|sec)\s+' + re.escape(sec) + r'\b\s*',
                    '',
                    answer,
                )
            answer = re.sub(r'  +', ' ', answer).strip()
            answer += "\n\n[Note: Some cited section numbers were not found in the source documents and have been removed.]"
        return answer

    def _boost_keyword_matches(self, query: str, results: list[dict]) -> list[dict]:
        import re
        if any(r.get("search_type") == "literal" for r in results):
            return results
        keywords = set()
        legal_refs = re.findall(r"(?:article|section|art|sec|chapter)\s+(\d+[A-Za-z]*)", query, re.IGNORECASE)
        if legal_refs:
            keywords.update(ref.lower() for ref in legal_refs)
        exact_phrases = re.findall(r'"([^"]+)"', query)
        if exact_phrases:
            keywords.update(p.lower() for p in exact_phrases)
        all_words = set(re.findall(r'\b([a-zA-Z]{5,})\b', query.lower()))
        all_words -= {"legal", "india", "indian", "would", "could", "should",
                       "shall", "have", "been", "being", "about", "what", "where",
                       "there", "which", "after", "before", "under", "without", "within"}
        keywords.update(all_words)
        if not keywords:
            return results
        matched = []
        unmatched = []
        for r in results:
            text = (r.get("text", "") + " " + r.get("metadata", {}).get("heading", ""))
            text_lower = text.lower()
            kw_count = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
            if kw_count > 0:
                occur = sum(text_lower.count(kw) for kw in keywords)
                r["combined_score"] = 0.5 + 0.1 * kw_count + min(occur * 0.02, 0.3)
                matched.append(r)
            else:
                unmatched.append(r)
        matched.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        unmatched.sort(key=lambda x: x.get("combined_score", x.get("score", 0)), reverse=True)
        return matched + unmatched

    def _format_context(self, results: list[dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            heading = r.get("metadata", {}).get("heading", "")
            source = r.get("metadata", {}).get("source", "")
            text = r.get("text", "")
            header = f"[{i}]"
            if heading:
                header += f" {heading}"
            if source:
                header += f" ({source})"
            parts.append(f"{header}\n{text}")
        return "\n\n".join(parts)
