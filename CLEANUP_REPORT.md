# Repository Audit & Cleanup Report

> Generated: 2026-07-09
> Phase 0 of migration from experimental PDF analyzer to production Legal AI desktop app.

---

## File-by-File Audit

| # | File | Inferred Purpose | Verdict | Action |
|---|------|-----------------|---------|--------|
| 1 | `pdf_analyzer.py` | Main Streamlit app: PDF extract, chunk, embed, query via Ollama | **REFACTOR** | Split into `app/core/ingestion/`, `app/core/embeddings/`, `app/core/rag/` |
| 2 | `cli.py` | Thin CLI wrapper around `PDFAnalyzer` | **DELETE** | Logic duplicated in `pdf_analyzer.py`; no unique value |
| 3 | `simple_cli.py` | Simpler CLI, nearly identical to `cli.py` | **DELETE** | Redundant third variant of same logic |
| 4 | `ultra_simple.py` | Minimal flat-script variant of the same pipeline | **DELETE** | Duplicate; no classes, no reusability |
| 5 | `enhanced_legal_rag_fixed.py` | Enhanced RAG with semantic chunking, hybrid search, advanced prompts | **KEEP** | Core algorithmic asset; refactor into `app/core/rag/` |
| 6 | `fastapi_app.py` | FastAPI REST wrapper importing `enhanced_legal_rag` (missing module) | **REFACTOR** | Dead import; refactor into `app/api/` with proper DI |
| 7 | `config.py` | Plain Python constants for Ollama model, chunk sizes, etc. | **REFACTOR** | Convert to `pydantic-settings` in `app/infra/config.py` |
| 8 | `requirements.txt` | 16 pinned deps (torch, fastapi, streamlit, ollama, etc.) | **DELETE** | Consolidate into `pyproject.toml` (Poetry) |
| 9 | `test_api.py` | HTTP test script for FastAPI endpoints | **REFACTOR** | Move to `tests/test_api/` with pytest fixtures |
| 10 | `run_fastapi.bat` | Batch launcher for FastAPI server | **MERGE** | Content → `scripts/` directory |
| 11 | `test_api.bat` | Batch launcher for test script | **MERGE** | Content → `scripts/` directory |
| 12 | `README.md` | Project README (outdated — references Streamlit, Ollama-only) | **REWRITE** | New README for production app |
| 13 | `ENHANCED_FEATURES.md` | Feature description for enhanced RAG | **DELETE** | Content absorbed into `docs/` |
| 14 | `FASTAPI_DOCS.md` | API documentation for FastAPI wrapper | **DELETE** | Content absorbed into `docs/` |
| 15 | `SETUP_GUIDE.md` | Setup instructions (hardcoded `d:\INTERNSHIP\` paths) | **DELETE** | Contains absolute local paths; replaced by `docs/BUILD.md` |
| 16 | `__pycache__/` | Python bytecode cache | **DELETE** | `*.pyc` must not be committed |
| 17 | `CLEANUP_REPORT.md` | This file | **KEEP** | Permanent record of audit |

---

## Dead Code / Dead Imports Found

1. **`fastapi_app.py:19`** — `from enhanced_legal_rag import EnhancedLegalRAG`  
   → Module `enhanced_legal_rag.py` was deleted in commit `1818f03`. The file `enhanced_legal_rag_fixed.py` exists but is not imported. This is a **dead import** that will raise `ModuleNotFoundError` at runtime.

2. **`enhanced_legal_rag_fixed.py:6-13`** — imports `pdfplumber`, `numpy`, `faiss`, `sentence_transformers`, `ollama`  
   → These are the core libraries, but the file has a duplicated `main()` that mixes CLI concerns with library code.

3. **`pdf_analyzer.py:9`** — `import streamlit as st`  
   → Ties the core logic to Streamlit, making it non-reusable outside the web context. Must be separated.

---

## Cleanup Actions Taken

| Action | Details |
|--------|---------|
| Deleted `cli.py` | Redundant CLI; all logic in `pdf_analyzer.py` |
| Deleted `simple_cli.py` | Three-way duplicate of the same pipeline |
| Deleted `ultra_simple.py` | Unmaintainable flat script variant |
| Deleted `ENHANCED_FEATURES.md` | Absorbed into future `docs/` |
| Deleted `FASTAPI_DOCS.md` | Absorbed into future `docs/` |
| Deleted `SETUP_GUIDE.md` | Contained hardcoded `d:\INTERNSHIP\` local path |
| Deleted `__pycache__/` | Compiled bytecode (2 `.pyc` files) |
| Created `.gitignore` | Python + Windows + IDE patterns |
| Created `pyproject.toml` | Poetry-managed dependencies (replaces `requirements.txt`) |
| Created `scripts/` | For build/package tools |

---

## Dependency Migration

`requirements.txt` (16 deps) → `pyproject.toml` (Poetry)

**Removed (not needed for target architecture):**
- `streamlit` — replaced by PyQt6 desktop UI
- `ollama` — replaced by `llama-cpp-python` (embedded local LLM)
- `pdfplumber` — replaced by `PyMuPDF` (fitz) for broader format support
- `torch` — not directly needed; SentenceTransformers bundles it

**Added for target architecture:**
- `PyMuPDF` — PDF & image handling
- `python-docx` — DOCX support
- `ebooklib` — EPUB support
- `python-pptx` — PPTX slides
- `pillow` — image processing
- `pytesseract` / `paddleocr` — OCR fallback
- `llama-cpp-python` — local LLM inference (no Ollama dependency)
- `chromadb` — local vector store (alternative to FAISS)
- `sentence-transformers` — embeddings
- `fastapi` + `uvicorn` — internal API
- `PyQt6` — desktop UI
- `pydantic-settings` — configuration management
- `dependency-injector` — DI container
- `cryptography` — AES-256 encryption
- `pytest` + `pytest-cov` — testing
- `ruff` + `mypy` + `black` — linting/formatting

---

## Git History (for reference)

```
1818f03 Delete enhanced_legal_rag.py
9278643 Changes done
32ff046 Fixed some bugs
17107ce Completed the task
```

---

## Next Steps After Phase 0

1. Create target directory structure (`app/`, `tests/`, `scripts/`, `docs/`, `data/`)
2. Implement `app/infra/config.py` with `pydantic-settings`
3. Implement `app/domain/` legal entities
4. Implement `app/core/ingestion/` document loaders
5. Implement `app/core/embeddings/` local embedding
6. Implement `app/core/retrieval/` vector DB
7. Implement `app/core/llm/` local LLM runner
8. Implement `app/core/rag/` RAG orchestration
9. Implement `app/ui/` PyQt6 desktop app
10. Implement `app/api/` local FastAPI service
11. Implement `scripts/build_knowledge_base.py`
12. Package as `.exe` with PyInstaller + Inno Setup
