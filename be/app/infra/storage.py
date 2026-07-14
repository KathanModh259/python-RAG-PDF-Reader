import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.infra.logging import logger


class DocumentRecord:
    def __init__(
        self,
        path: Path,
        source: str,
        doc_type: str,
        act_name: Optional[str] = None,
        section_no: Optional[str] = None,
        page: Optional[int] = None,
    ):
        self.path = path
        self.source = source
        self.doc_type = doc_type
        self.act_name = act_name
        self.section_no = section_no
        self.page = page
        self.hash = self._compute_hash()
        self.ingested_at = datetime.now(timezone.utc).isoformat()

    def _compute_hash(self) -> str:
        if self.path.exists():
            return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
        return ""

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "source": self.source,
            "doc_type": self.doc_type,
            "act_name": self.act_name,
            "section_no": self.section_no,
            "page": self.page,
            "hash": self.hash,
            "ingested_at": self.ingested_at,
        }


class LocalStorage:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("LocalStorage initialized at %s", base_path)

    def store_raw(self, subdir: str, filename: str, data: bytes) -> Path:
        dest = self.base_path / "raw" / subdir
        dest.mkdir(parents=True, exist_ok=True)
        filepath = dest / filename
        filepath.write_bytes(data)
        logger.debug("Stored raw file: %s", filepath)
        return filepath

    def store_processed(self, filename: str, data: bytes) -> Path:
        dest = self.base_path / "processed"
        dest.mkdir(parents=True, exist_ok=True)
        filepath = dest / filename
        filepath.write_bytes(data)
        return filepath

    def file_exists(self, subdir: str, filename: str) -> bool:
        return (self.base_path / "raw" / subdir / filename).exists()
