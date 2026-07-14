# Nyaya Mitra (Legal AI Assistant)

**Nyaya Mitra** is an offline-first, citizen-centric Legal AI assistant and desktop application designed specifically for Indian legal contexts. It operates 100% locally with zero external network dependencies, ensuring complete privacy and offline utility.

Grounded in the Constitution of India, Central Bare Acts, and landmark legal cases, Nyaya Mitra assists citizens in understanding legal scenarios, identifying predatory practices, tracking legal aid workflows, and translating complex legalese into plain language.

---

## Key Features

- **Decoupled Architecture**: 
  - **Local API Backend (`/be`)**: A robust Python RAG engine powered by FastAPI, `llama-cpp-python`, and local vector stores.
  - **Desktop Frontend App (`/fe`)**: A modern, high-performance UI built with React, TypeScript, TailwindCSS v4, and Tauri.
- **Citizen Protection & Anti-Exploitation**:
  - **Contract & Document Scanner**: Detects unfair clauses, hidden liabilities, and scams locally.
  - **Fee Sanity Checker**: Cross-references local legal fee databases to ensure users aren't overcharged.
- **Emergency Panic Mode**:
  - Automatically detects urgent/high-risk prompts and displays direct links to Indian legal aid services (e.g. NALSA, DLSA hotlines).
- **Multilingual Support & Voice Integration**:
  - Localized system prompts and screens for regional Indian languages.
  - Speech-to-Text (STT) and Text-to-Speech (TTS) engine running locally.
- **Decentralized Legal RAG**:
  - Grounded legal search combining dense semantic embeddings (BGE-small) with sparse BM25 keyword search, reranked by a cross-encoder model.
  - Splits documents based on legal syntax (Articles, Sections, Chapters).
- **Second Opinion Evaluator**:
  - Validates model responses against source citations to prevent hallucinations and check reasoning.
- **Templates & Case Tracking**:
  - Generates standardized legal draft templates.
  - Tracks court case stages and hearing status locally.

---

## Directory Structure

```
python-RAG-PDF-Reader/
├── be/                       # Backend Application
│   ├── app/
│   │   ├── api/              # FastAPI local API endpoints
│   │   ├── core/             # RAG, ingestion, embedding, LLM, anti-exploitation, voice
│   │   ├── di/               # dependency injection container
│   │   ├── domain/           # Legal data models (Acts, Sections, Judgments, Cases)
│   │   ├── infra/            # Config (.env settings), logging, storage, crypto
│   │   └── ui/               # PyQt6 fallback desktop UI
│   ├── data/                 # Raw/processed legal documents, vector DB, models
│   ├── scripts/              # Setup, indexing, and packaging utilities
│   └── pyproject.toml        # Poetry python configuration
│
├── fe/                       # Frontend Application
│   ├── src/                  # React screens, components, custom hooks
│   ├── src-tauri/            # Tauri desktop configuration & Rust build files
│   ├── package.json          # Node dependencies and scripts
│   └── vite.config.ts        # Vite configuration
│
└── docs/                     # Detailed architectural and usage documents
```

---

## Getting Started

### 1. Setting up the Backend (`/be`)

Go to the [Backend Documentation](file:///c:/Kathan/exp/python-RAG-PDF-Reader/be/README.md) for more details.

```bash
cd be

# Install dependencies using Poetry
poetry install

# Download the default quantized model (Phi-3-mini or TinyLlama)
poetry run python scripts/download_model.py

# Build the vector store database
poetry run python scripts/build_knowledge_base.py

# Run the backend API server
poetry run python app/main.py --mode api
```

The FastAPI server will run locally at `http://127.0.0.1:8765`.

### 2. Setting up the Frontend (`/fe`)

Go to the [Frontend Documentation](file:///c:/Kathan/exp/python-RAG-PDF-Reader/fe/README.md) for more details.

```bash
cd fe

# Install Node dependencies
npm install

# Run the web development server (Vite)
npm run dev

# Or run it wrapped inside Tauri (Desktop Mode)
npm run tauri dev
```

---

## Model Specifications

- **Embedding Model**: `BAAI/bge-small-en-v1.5` (~100MB, local)
- **Reranker Model**: `BAAI/bge-reranker-base` (local)
- **Local LLM**: `Phi-3-mini-4k-instruct-GGUF` (or `TinyLlama-1.1B` as a lightweight alternative)
- **Inference Library**: `llama-cpp-python` (CPU execution by default, supports CUDA hardware acceleration)

---

## Disclaimer

This tool is designed to provide informational assistance to citizens and is not a substitute for professional legal advice or legal representation. Always consult a qualified advocate for legal matters.

---

## License

MIT

