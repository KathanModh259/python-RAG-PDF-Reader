# Nyaya Mitra Backend API & RAG Engine

The backend of **Nyaya Mitra** is a Python-based Clean Architecture application. It provides document ingestion, preprocessing (cleaning and chunking), hybrid search (ChromaDB + BM25), and local LLM execution via `llama-cpp-python`. It exposes a local API server built with FastAPI.

---

## Features

- **Local LLM Execution**: Uses `llama-cpp-python` to run quantized GGUF models (e.g. Phi-3-mini, TinyLlama) locally. No internet connection or external servers (like Ollama) are required.
- **Hybrid Retrieval**: Employs dense vector embeddings via `BAAI/bge-small-en-v1.5` and sparse BM25 index matching, reranked with a cross-encoder (`BAAI/bge-reranker-base`).
- **Legal-aware Processing**: Customized parser and chunker split legal documents by chapters, sections, and clauses.
- **Citizen Protections Module**: Includes scam detectors, emergency response detectors, and fee validators.
- **FastAPI Endpoints**: Local REST endpoints for querying, document uploading, and system health status.
- **Fallback PyQt6 Interface**: Includes a desktop window interface (`--mode gui`) for direct pyqt6 execution if needed.

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Poetry (Package manager)

### Steps

1. **Install Dependencies**:
   ```bash
   poetry install
   ```

2. **Configure Settings**:
   Create a `.env` file in the `be/` folder to configure variables:
   ```env
   LLM_ENABLED=True
   RERANKER_ENABLED=True
   ENCRYPTION_ENABLED=False
   API_HOST=127.0.0.1
   API_PORT=8765
   ```

3. **Download Local LLM Model**:
   Downloads the default quantized model:
   ```bash
   poetry run python scripts/download_model.py
   ```

4. **Build Knowledge Base**:
   Indexes the preset Indian legal texts into the vector store:
   ```bash
   poetry run python scripts/build_knowledge_base.py
   ```

5. **Start backend API server**:
   ```bash
   poetry run python app/main.py --mode api
   ```

---

## API Endpoints

The FastAPI server exposes the following endpoints (default base URL: `http://127.0.0.1:8765`):

- **`GET /health`**: Health status indicating whether the LLM is loaded and how many documents are indexed.
- **`POST /query`**: Accepts a JSON body containing `question` and `mode`.
- **`POST /upload`**: Multi-part file upload supporting PDFs, DOCX, TXT, EPUB, etc., which are cleaned, chunked, and indexed on-the-fly.
- **`GET /stats`**: Returns statistics about indexed documents and the vector store.

---

## Running Tests

We use `pytest` for unit testing:
```bash
poetry run pytest
```

