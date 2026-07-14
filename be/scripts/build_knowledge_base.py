#!/usr/bin/env python3
"""
Build the Indian Legal Knowledge Base.

Downloads (or accepts local), structures, chunks, embeds, and persists
Indian legal corpora into the local vector store.

Public-domain sources:
  - Constitution of India (legislative.gov.in)
  - India Code (indiacode.nic.in) for Central Bare Acts
  - Supreme Court judgments (public domain via sci.gov.in)

Usage:
    python scripts/build_knowledge_base.py
    python scripts/build_knowledge_base.py --data-dir ./my_docs
    python scripts/build_knowledge_base.py --force-download
"""

import argparse
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.preprocessing.chunker import LegalChunker
from app.core.preprocessing.cleaner import TextCleaner
from app.core.embeddings.embedder import EmbeddingModel
from app.core.retrieval.vector_store import VectorStore
from app.infra.config import settings
from app.infra.logging import logger


KNOWLEDGE_SOURCES = [
    # ── Foundational ─────────────────────────────────────────────
    {
        "name": "Constitution of India",
        "short": "COI",
        "url": "https://www.indiacode.nic.in/bitstream/123456789/19150/1/constitution_of_india.pdf",
        "filename": "constitution_of_india.pdf",
        "status": "in_force",
    },

    # ── Legacy Criminal Laws (still needed for pending cases) ────
    {
        "name": "Indian Penal Code, 1860",
        "short": "IPC",
        "url": "https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf",
        "filename": "indian_penal_code_1860.pdf",
        "status": "repealed_but_active_for_pre_2024_cases",
        "replaced_by": "BNS",
    },
    {
        "name": "Code of Criminal Procedure, 1973",
        "short": "CrPC",
        "url": "https://www.indiacode.nic.in/bitstream/123456789/15272/1/the_code_of_criminal_procedure%2C_1973.pdf",
        "filename": "code_of_criminal_procedure_1973.pdf",
        "status": "repealed_but_active_for_pre_2024_cases",
        "replaced_by": "BNSS",
    },
    {
        "name": "Indian Evidence Act, 1872",
        "short": "IEA",
        "url": "https://www.indiacode.nic.in/bitstream/123456789/15351/1/iea_1872.pdf",
        "filename": "indian_evidence_act_1872.pdf",
        "status": "repealed_but_active_for_pre_2024_cases",
        "replaced_by": "BSA",
    },

    # ── New Sanhitas (MANDATORY for 2026) ────────────────────────
    {
        "name": "Bharatiya Nyaya Sanhita, 2023",
        "short": "BNS",
        "url": "https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf",
        "filename": "bharatiya_nyaya_sanhita_2023.pdf",
        "status": "in_force_from_2024_07_01",
    },
    {
        "name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "short": "BNSS",
        "url": "https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf",
        "filename": "bharatiya_nagarik_suraksha_sanhita_2023.pdf",
        "status": "in_force_from_2024_07_01",
    },
    {
        "name": "Bharatiya Sakshya Adhiniyam, 2023",
        "short": "BSA",
        "url": "https://www.mha.gov.in/sites/default/files/2024-04/250882_english_01042024_0.pdf",
        "filename": "bharatiya_sakshya_adhiniyam_2023.pdf",
        "status": "in_force_from_2024_07_01",
    },
]



def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        logger.info("Already cached: %s", dest.name)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest.name)
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=dest.name, total=total, unit="B", unit_scale=True
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        logger.warning("Download failed for %s: %s", url, e)
        logger.info("You can manually download and place in %s", dest.parent)
        return False


def build_index(force_download: bool = False, data_dir: Path | None = None) -> None:
    logger.info("Starting knowledge base build")

    raw_dir = data_dir or settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    for source in KNOWLEDGE_SOURCES:
        dest = raw_dir / source["filename"]
        if force_download or not dest.exists():
            download_file(source["url"], dest)

    embedder = EmbeddingModel(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    embedder.load()

    vector_store = VectorStore(
        store_type=settings.vector_store_type,
        persist_dir=settings.vector_store_path,
    )
    vector_store.initialize()

    cleaner = TextCleaner()
    chunker = LegalChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    from app.core.ingestion.loader import DocumentLoader
    loader = DocumentLoader(ocr_engine=settings.ocr_engine)

    total_chunks = 0
    pdf_files = list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.txt"))
    if not pdf_files:
        logger.warning("No documents found in %s", raw_dir)
        logger.info("Place PDF/TXT files in %s and re-run", raw_dir)
        return

    for doc_path in pdf_files:
        logger.info("Processing %s", doc_path.name)
        text = loader.load(doc_path)
        if not text:
            logger.warning("Skipping %s (no text extracted)", doc_path.name)
            continue

        cleaned = cleaner.clean(text)
        chunks = chunker.chunk(cleaned, source=doc_path.name)

        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "heading": c.get("heading", ""),
                "type": c.get("type", ""),
                "source": c.get("source", doc_path.name),
            }
            for c in chunks
        ]

        embeddings = embedder.encode(texts)
        vector_store.add(embeddings, texts, metadatas)
        total_chunks += len(chunks)
        logger.info("Added %d chunks from %s", len(chunks), doc_path.name)

    logger.info("Knowledge base build complete. Total chunks: %d", total_chunks)

    manifest = {
        "total_chunks": total_chunks,
        "embedding_model": settings.embedding_model,
        "vector_store_type": settings.vector_store_type,
        "documents_processed": len(pdf_files),
    }
    manifest_path = settings.processed_dir / "kb_manifest.json"
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Indian Legal Knowledge Base")
    parser.add_argument("--force-download", action="store_true", help="Re-download documents")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to directory with legal PDFs/TXTs")
    args = parser.parse_args()
    build_index(force_download=args.force_download, data_dir=args.data_dir)
