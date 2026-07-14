from pathlib import Path

from dependency_injector import containers, providers

from app.core.anti_exploitation.scam_detector import FeeSanityChecker, RightsReminder, ScamDetector
from app.core.embeddings.embedder import EmbeddingModel
from app.core.ingestion.loader import DocumentLoader
from app.core.llm.runner import LLMRunner
from app.core.localization.detector import LanguageDetector
from app.core.preprocessing.chunker import LegalChunker
from app.core.preprocessing.cleaner import TextCleaner
from app.core.rag.citizen_orchestrator import CitizenProtectionOrchestrator
from app.core.rag.orchestrator import RAGOrchestrator
from app.core.retrieval.hybrid_search import HybridSearch
from app.core.retrieval.reranker import CrossEncoderReranker
from app.core.retrieval.vector_store import VectorStore
from app.core.second_opinion.evaluator import SecondOpinionEngine
from app.core.voice.engine import VoiceEngine
from app.domain.cases.tracker import CaseTracker
from app.domain.templates.renderer import TemplateRenderer
from app.infra.config import Settings


class ApplicationContainer(containers.DeclarativeContainer):
    config = providers.Singleton(Settings)
    settings = providers.Configuration()

    embedder = providers.Singleton(
        EmbeddingModel,
        model_name=config.provided.embedding_model,
        device=config.provided.embedding_device,
    )

    vector_store = providers.Singleton(
        VectorStore,
        store_type=config.provided.vector_store_type,
        persist_dir=config.provided.vector_store_path,
    )

    hybrid_search = providers.Factory(
        HybridSearch,
        vector_store=vector_store,
        alpha=0.7,
    )

    llm_runner = providers.Singleton(
        LLMRunner,
        enabled=config.provided.llm_enabled,
        model_path=config.provided.llm_model_path,
        n_ctx=config.provided.llm_n_ctx,
        n_threads=config.provided.llm_n_threads,
        max_tokens=config.provided.llm_max_tokens,
        temperature=config.provided.llm_temperature,
        use_gpu=config.provided.llm_use_gpu,
        ollama_base_url=config.provided.ollama_base_url,
        ollama_model=config.provided.ollama_model,
    )

    chunker = providers.Factory(
        LegalChunker,
        chunk_size=config.provided.chunk_size,
        chunk_overlap=config.provided.chunk_overlap,
    )

    cleaner = providers.Factory(TextCleaner)

    document_loader = providers.Factory(
        DocumentLoader,
        ocr_engine=config.provided.ocr_engine,
    )

    reranker = providers.Singleton(
        CrossEncoderReranker,
        model_name=config.provided.reranker_model,
    )

    rag_orchestrator = providers.Singleton(
        RAGOrchestrator,
        embedder=embedder,
        vector_store=vector_store,
        hybrid_search=hybrid_search,
        llm=llm_runner,
        chunker=chunker,
        reranker=reranker,
        top_k=config.provided.retrieval_top_k,
    )

    citizen_orchestrator = providers.Singleton(
        CitizenProtectionOrchestrator,
        embedder=embedder,
        vector_store=vector_store,
        hybrid_search=hybrid_search,
        llm=llm_runner,
        chunker=chunker,
        top_k=config.provided.retrieval_top_k,
    )

    scam_detector = providers.Singleton(ScamDetector)
    fee_checker = providers.Singleton(FeeSanityChecker)
    rights_reminder = providers.Singleton(RightsReminder)
    second_opinion = providers.Singleton(SecondOpinionEngine)
    language_detector = providers.Singleton(LanguageDetector)
    template_renderer = providers.Singleton(TemplateRenderer)
    case_tracker = providers.Singleton(CaseTracker)
    voice_engine = providers.Singleton(VoiceEngine)
