#!/usr/bin/env python3
"""
Model downloader with tier selection.

Usage:
    python scripts/download_model.py                  # lightweight (default, ~700MB)
    python scripts/download_model.py --tier standard   # standard (~2.2GB)
    python scripts/download_model.py --tier full       # full (~4.7GB)
    python scripts/download_model.py --list            # show available models
"""

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.infra.config import settings
from app.infra.logging import logger


TIERS = {
    "lightweight": {
        "name": "TinyLlama-1.1B-Chat",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.7,
        "ram_gb": 2,
        "config_updates": {
            "llm_n_ctx": 1024,
            "llm_max_tokens": 256,
            "llm_n_threads": 2,
        },
    },
    "standard": {
        "name": "Phi-3-mini-4k-instruct",
        "repo": "QuantFactory/Phi-3-mini-4k-instruct-GGUF",
        "file": "Phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "url": "https://huggingface.co/QuantFactory/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "size_gb": 2.2,
        "ram_gb": 4,
        "config_updates": {
            "llm_n_ctx": 2048,
            "llm_max_tokens": 512,
            "llm_n_threads": 4,
        },
    },
    "full": {
        "name": "Meta-Llama-3.1-8B-Instruct",
        "repo": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "ram_gb": 8,
        "config_updates": {
            "llm_n_ctx": 4096,
            "llm_max_tokens": 1024,
            "llm_n_threads": 4,
        },
    },
}


def download_with_resume(url: str, dest: Path) -> bool:
    if dest.exists():
        existing_size = dest.stat().st_size
        headers = {"Range": f"bytes={existing_size}-"}
        logger.info("Resuming download from byte %d", existing_size)
    else:
        headers = {}

    try:
        resp = requests.get(url, stream=True, timeout=30, headers=headers)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) + (dest.stat().st_size if dest.exists() else 0)
        mode = "ab" if dest.exists() else "wb"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, mode) as f, tqdm(
            desc=dest.name, total=total, unit="B", unit_scale=True,
            initial=dest.stat().st_size if dest.exists() else 0,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        logger.error("Download failed: %s - %s", url, e)
        return False


def verify_checksum(path: Path) -> bool:
    logger.info("Verifying %s...", path.name)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    logger.info("Verified: %s (%s)", path.name, sha256.hexdigest()[:16])
    return True


def download_tier(tier_name: str) -> None:
    tier = TIERS.get(tier_name)
    if not tier:
        logger.error("Unknown tier: %s. Use --list to see options.", tier_name)
        sys.exit(1)

    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / tier["file"]

    if dest.exists():
        logger.info("Already downloaded: %s", dest)
        verify_checksum(dest)
    else:
        logger.info("Downloading %s (%s, %.1f GB, needs ~%d GB RAM)...",
                     tier["name"], tier["file"], tier["size_gb"], tier["ram_gb"])
        if not download_with_resume(tier["url"], dest):
            sys.exit(1)
        verify_checksum(dest)

    from app.infra.config import Settings
    cfg = Settings()
    cfg.llm_enabled = True
    cfg.llm_model_path = dest
    for key, val in tier["config_updates"].items():
        setattr(cfg, key, val)

    logger.info("LLM ready: %s", tier["name"])
    logger.info("Config: enabled=True, path=%s", dest)


def list_tiers() -> None:
    print("\nAvailable model tiers:\n")
    print(f"{'TIER':<16} {'MODEL':<30} {'SIZE':<8} {'RAM':<8}")
    print("-" * 62)
    for name, info in TIERS.items():
        print(f"{name:<16} {info['name']:<30} {info['size_gb']:.1f}GB  {info['ram_gb']}GB+")
    print("\nDefault: lightweight (works on any machine with 2GB+ RAM)")
    print("Recommendation: start with lightweight, upgrade if you have more RAM.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LLM model for Legal AI")
    parser.add_argument("--tier", choices=list(TIERS.keys()), default="lightweight",
                        help="Model size tier (default: lightweight)")
    parser.add_argument("--list", action="store_true", help="List available model tiers")
    args = parser.parse_args()

    if args.list:
        list_tiers()
    else:
        download_tier(args.tier)
