# Architecture

## Overview

Legal AI follows a clean architecture pattern with strict separation of concerns. The application is designed for offline-first operation with swappable components via dependency injection.

```
                     +-----------+
                     |   User    |
                     +-----+-----+
                           |
              +------------+------------+
              |                         |
        +-----v-----+           +-------v-------+
        |  PyQt6 UI |           |  FastAPI API  |
        +-----+-----+           +-------+-------+
              |                         |
        +-----v-------------------------v-------+
        |         RAG Orchestrator               |
        |   (app/core/rag/orchestrator.py)       |
        +--+--------+--------+--------+---------+
           |        |        |        |
      +----v---+ +-v------+ +-v----+ +v--------+
      |Hybrid  | |Vector  | |Cross-| |LLM      |
      |Search  | |Store   | |Enc.  | |Runner   |
      +--------+ +--------+ |Rerank| +---------+
                            +------+
```

## Module Boundaries

### app/infra/ — Infrastructure Layer
- **config.py**: pydantic-settings BaseSettings, reads from .env or defaults
- **logging.py**: Structured logging to console + file
- **storage.py**: Local filesystem abstraction for raw/processed data
- **crypto.py**: AES-256 encryption via Fernet (PBKDF2 key derivation)

### app/domain/ — Domain Layer
- **act.py**: Act, Chapter dataclasses
- **section.py**: LegalSection, Clause dataclasses
- **judgment.py**: Judgment dataclass with citation metadata

### app/core/ — Core Business Logic
- **ingestion/loader.py**: Multi-format document parser (PDF, DOCX, HTML, EPUB, images with OCR)
- **preprocessing/cleaner.py**: Text normalization, OCR fix, header/footer removal
- **preprocessing/chunker.py**: Legal-aware splitting by Article/Section/Chapter
- **embeddings/embedder.py**: SentenceTransformer wrapper with lazy loading
- **retrieval/vector_store.py**: ChromaDB/FAISS abstraction
- **retrieval/hybrid_search.py**: Dense + keyword fusion search
- **retrieval/reranker.py**: Cross-encoder re-ranking
- **llm/runner.py**: llama-cpp-python wrapper
- **rag/orchestrator.py**: End-to-end RAG pipeline
- **rag/prompts.py**: Prompt templates for different query modes

### app/api/ — API Layer
- **service.py**: FastAPI application factory. Endpoints: /health, /query, /upload, /stats

### app/ui/ — Presentation Layer
- **main_window.py**: PyQt6 main window with chat display, question input, source panel

### app/di/ — Dependency Injection
- **container.py**: DeclarativeContainer wiring all dependencies

## Data Flow

1. **Ingestion**: Document -> DocumentLoader -> TextCleaner -> LegalChunker -> Embed -> VectorStore
2. **Query**: Question -> EmbedQuery -> HybridSearch -> CrossEncoderReranker -> ContextBuilder -> LLMPrompt -> LLM -> Answer
3. **Storage**: All data local. Vector DB persisted to disk. Document store optionally encrypted.
