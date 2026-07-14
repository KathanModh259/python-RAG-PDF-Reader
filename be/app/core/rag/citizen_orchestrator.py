from typing import Any, Optional

from app.core.anti_exploitation.scam_detector import (
    FeeSanityChecker,
    RightsReminder,
    ScamDetector,
)
from app.core.embeddings.embedder import EmbeddingModel
from app.core.llm.prompts.citizen_framework import build_citizen_prompt
from app.core.llm.runner import LLMRunner
from app.core.preprocessing.chunker import LegalChunker
from app.core.rag.orchestrator import RAGOrchestrator
from app.core.retrieval.hybrid_search import HybridSearch
from app.core.retrieval.vector_store import VectorStore
from app.domain.citizen_protection.emergency import (
    EMERGENCY_HELPLINES,
    detect_urgency,
    get_helplines_text,
)
from app.domain.citizen_protection.models import (
    ActionStep,
    EightPartResponse,
    LegalCitation,
    RedFlag,
    RiskLevel,
    Urgency,
)
from app.infra.logging import logger


class CitizenProtectionOrchestrator(RAGOrchestrator):
    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        hybrid_search: HybridSearch,
        llm: LLMRunner,
        chunker: Optional[LegalChunker] = None,
        top_k: int = 5,
    ):
        super().__init__(embedder, vector_store, hybrid_search, llm, chunker, top_k=top_k)
        self.scam_detector = ScamDetector()
        self.fee_checker = FeeSanityChecker()
        self.rights_reminder = RightsReminder()

    def respond_to_citizen(
        self,
        user_input: str,
        document_text: Optional[str] = None,
        is_follow_up: bool = False,
        needs_draft: bool = False,
    ) -> dict[str, Any]:
        has_urgency = detect_urgency(user_input)

        context = ""
        if document_text:
            self.index_document(document_text, source="user_document")
            query_emb = self.embedder.encode_query(user_input)
            results = self.hybrid_search.search(user_input, query_emb, top_k=self.top_k)
            context = self._format_context(results)

        scam_flags = []
        fee_analysis = None
        relevant_rights = []

        if document_text:
            scam_flags = self.scam_detector.analyze(document_text)
            matter_type = self.fee_checker.detect_matter_type(document_text)
            if matter_type:
                fee_analysis = {"detected_matter": matter_type}
            relevant_rights = self.rights_reminder.get_relevant_rights(
                document_text + " " + user_input
            )

        prompt = build_citizen_prompt(
            user_input=user_input,
            context=context,
            is_follow_up=is_follow_up,
            needs_draft=needs_draft,
            detected_urgency=has_urgency,
        )

        answer = self.llm.generate(prompt)

        if answer is None:
            answer = self._build_fallback_response(user_input, context, has_urgency)

        return {
            "answer": answer,
            "sources": self._parse_citations(answer),
            "scam_flags": scam_flags,
            "fee_analysis": fee_analysis,
            "relevant_rights": relevant_rights,
            "helplines": get_helplines_text() if has_urgency else "",
        }

    def _build_fallback_response(
        self, user_input: str, context: str, has_urgency: bool
    ) -> str:
        parts = []

        if has_urgency:
            parts.append("EMERGENCY - PLEASE CONTACT HELPLINES")
            parts.append("=" * 40)
            for name, number, desc in EMERGENCY_HELPLINES:
                parts.append(f"{name}: {number} - {desc}")
            parts.append("")

        parts.append("1. WHAT THIS SITUATION ACTUALLY IS")
        parts.append(
            "I understand you are facing a legal situation based on what you described."
        )
        parts.append("")

        parts.append("2. WHAT HAS HAPPENED TO YOU (IN SIMPLE WORDS)")
        parts.append(
            "Since the AI language model is not currently available, here is what I "
            "found in the legal documents that may be relevant to your situation:"
        )
        parts.append("")

        parts.append("3. HOW SERIOUS IS THIS?")
        parts.append("Please check with a legal aid clinic for a proper assessment.")
        parts.append("")

        parts.append("4. WHAT YOU MUST DO NEXT")
        parts.append("1. Call NALSA helpline 15100 for free legal advice (Do today)")
        parts.append("2. Visit your nearest District Legal Services Authority (This week)")
        parts.append("3. Keep all documents safe (Do today)")
        parts.append("")

        parts.append("7. LEGAL BASIS (FOR VERIFICATION)")
        if context:
            parts.append("Relevant document excerpts:")
            parts.append(context[:2000])
        else:
            parts.append("Search the knowledge base by indexing legal documents.")
        parts.append("")

        parts.append("8. FINAL CONCLUSION")
        parts.append(
            "You have taken the first step by seeking help. Your rights exist to protect you. "
            "Free legal aid is available. You are not alone in this."
        )
        parts.append("")
        parts.append(
            "DISCLAIMER: I am an AI assistant built to help you understand your legal "
            "situation in simple words. I am not a lawyer and this is not legal representation."
        )

        return "\n\n".join(parts)

    def _parse_citations(self, text: str) -> list[dict]:
        import re

        sources = []
        section_pattern = r"(?:Section|Article|Rule|Clause)\s+(\d+[A-Za-z]*(?:\.\d+)*)"
        act_pattern = r"((?:Constitution|Act|Code|Sanhita|Adhiniyam|Ordinance)[^,.]+)"

        for match in re.finditer(section_pattern, text):
            sources.append({
                "citation": match.group(0),
                "text": text[max(0, match.start() - 50) : match.end() + 100],
            })

        for match in re.finditer(act_pattern, text):
            if not any(s["citation"] == match.group(1) for s in sources):
                sources.append({
                    "citation": match.group(1).strip(),
                    "text": text[max(0, match.start() - 50) : match.end() + 100],
                })

        return sources[:10]

    def get_emergency_numbers(self) -> list[tuple[str, str, str]]:
        return EMERGENCY_HELPLINES
