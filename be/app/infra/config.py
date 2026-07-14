from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        arbitrary_types_allowed=True,
    )

    app_name: str = "Legal AI"
    app_version: str = "0.1.0"
    debug: bool = False

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("data/models")
    cache_dir: Path = Path("data/cache")
    log_dir: Path = Path("logs")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: Literal["cpu", "cuda", "auto"] = "cpu"
    embedding_dimension: int = 384

    vector_store_type: Literal["chroma", "faiss"] = "chroma"
    vector_store_path: Path = Path("data/processed/vectordb_new")

    llm_enabled: bool = False
    llm_model_path: Optional[Path] = None
    llm_model_repo: str = "QuantFactory/Phi-3-mini-4k-instruct-GGUF"
    llm_model_file: str = "Phi-3-mini-4k-instruct.Q4_K_M.gguf"
    llm_model_alt_repo: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    llm_model_alt_file: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    llm_n_ctx: int = 2048
    llm_n_threads: int = 2
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1
    llm_use_gpu: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"

    chunk_size: int = 384
    chunk_overlap: int = 64
    retrieval_top_k: int = 5
    rerank_top_k: int = 3

    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"

    encryption_enabled: bool = False
    encryption_key_file: Optional[Path] = None

    telemetry_opt_in: bool = False
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [
        ".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt",
        ".html", ".htm", ".epub", ".md",
        ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ]

    first_run: bool = True
    disclaimer_accepted: bool = False

    api_host: str = "127.0.0.1"
    api_port: int = 8765

    ocr_engine: Literal["paddle", "tesseract"] = "paddle"
    ocr_languages: str = "en"

    index_data_path: Path = Path("data/processed/index")


settings = Settings()
