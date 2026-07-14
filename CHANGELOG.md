# Changelog

## [0.2.0] - 2026-07-14

### Added
- **Nyaya Mitra Citizen Protection Framework**:
  - `app/core/anti_exploitation/scam_detector.py` for scanning documents/contracts for predatory practices, red flags, and fee sanity checks.
  - Emergency legal aid hotlines (NALSA/DLSA) with automated prompt-based urgency detection.
  - Interactive citizen rights and protections guidance.
- **Modern React + Tauri Desktop UI**:
  - Complete Vite-based frontend in `fe/` featuring React, TypeScript, and TailwindCSS v4.
  - Cross-platform desktop integration with Tauri (`src-tauri`).
  - Implemented interactive screens for Chat, Document Library, Guided Flows, Panic Mode (emergency alerts), Anti-Exploitation analysis, and Settings.
- **Multilingual Support & Localization (l10n)**:
  - Localized system components and voice interfaces to support regional languages.
- **Voice Engine (STT/TTS)**:
  - Fully local Speech-to-Text (STT) and Text-to-Speech (TTS) integration.
- **Second Opinion Evaluator**:
  - Independent assessment engine for RAG outputs to check grounding and correctness.
- **Templates & Case Tracking**:
  - Document templates generator and legal case progress tracker.

### Changed
- Shifted architecture to a decoupled Frontend-Backend model (FastAPI API server + Tauri React application).
- Configured default settings optimized for low-resource environments (e.g. TinyLlama, lightweight embedding models).
- Resolved hardware device allocation options (`auto` -> CPU/CUDA fallback).

## [0.1.0] - 2026-07-09

### Added
- Complete repository audit and cleanup (Phase 0)
- CLEANUP_REPORT.md documenting every file's disposition
- .gitignore with Python/Windows/IDE patterns
- pyproject.toml (Poetry) replacing requirements.txt
- Clean architecture layout under `app/`:
  - `app/api/` — FastAPI local-only service
  - `app/core/ingestion/` — Document loader (PDF, DOCX, HTML, EPUB, images, text)
  - `app/core/preprocessing/` — TextCleaner + LegalChunker
  - `app/core/embeddings/` — SentenceTransformer embedding model
  - `app/core/retrieval/` — VectorStore (Chroma/FAISS) + HybridSearch
  - `app/core/llm/` — llama-cpp-python local LLM runner
  - `app/core/rag/` — RAG orchestrator with prompt engineering
  - `app/domain/` — Legal entities (Act, Section, Judgment, Clause)
  - `app/infra/` — Configuration, logging, storage, crypto
  - `app/di/` — Dependency injection container
  - `app/ui/` — PyQt6 desktop UI (main window with chat/sources)
  - `app/main.py` — CLI entry point (GUI or API mode)
- `app/core/training/` — Fine-tuning stub for LoRA/QLoRA
- Initial test suite: test_cleaner, test_chunker, test_loader, test_embeddings
- `scripts/` and `data/` skeleton directories

### Changed
- Refactored from flat Streamlit/Ollama monolith to modular clean architecture
- Config moved from plain Python to pydantic-settings

### Removed
- cli.py, simple_cli.py, ultra_simple.py (redundant CLIs)
- ENHANCED_FEATURES.md, FASTAPI_DOCS.md, SETUP_GUIDE.md (absorbed into docs/)
- requirements.txt (replaced by pyproject.toml)
- __pycache__/ (compiled bytecode)
- Tracked Arabic PDF (should not have been committed)

