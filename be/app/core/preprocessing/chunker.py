import re
from typing import Any, Optional

from app.infra.logging import logger


class LegalChunker:
    SECTION_PATTERNS = [
        (r"(?i)(Article\s+[IVXLCDM\d]+[A-Za-z\d]*(?:\.\d+)*)", "article"),
        (r"(?i)(Section\s+\d+[A-Za-z\d]*(?:\.\d+)*)", "section"),
        (r"(?i)(Chapter\s+[IVXLCDM\d]+)", "chapter"),
        (r"(?i)(Schedule\s+[IVXLCDM\d]+)", "schedule"),
        (r"(?i)(Part\s+[IVXLCDM\d]+)", "part"),
        (r"(?i)(Clause\s+\d+)", "clause"),
        (r"(?i)(Rule\s+\d+)", "rule"),
        (r"^(?:\d+\.\s*)(.+)", "numbered"),
        (r"^(?:\(\d+\)\s*)(.+)", "parenthetical"),
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, source: str = "") -> list[dict[str, Any]]:
        sections = self._split_by_legal_structure(text)
        if not sections:
            sections = [{"text": text, "heading": "Document", "type": "document"}]

        chunks = []
        for sec in sections:
            sec_text = sec["text"]
            sec_words = len(sec_text.split())
            if sec_words <= self.chunk_size:
                chunks.append({
                    "text": sec_text,
                    "heading": sec.get("heading", ""),
                    "type": sec.get("type", "unknown"),
                    "source": source,
                })
            else:
                sub_chunks = self._split_large_section(sec_text, sec, source)
                chunks.extend(sub_chunks)

        logger.info("Chunked into %d segments", len(chunks))
        return chunks

    def _split_by_legal_structure(self, text: str) -> list[dict[str, str]]:
        lines = text.split("\n")
        sections = []
        current: Optional[dict[str, str]] = None

        for line in lines:
            matched = None
            for pattern, sec_type in self.SECTION_PATTERNS:
                m = re.match(pattern, line.strip())
                if m:
                    matched = (m.group(1), sec_type)
                    break

            if matched:
                if current and current["text"].strip():
                    sections.append(current)
                current = {
                    "heading": matched[0],
                    "type": matched[1],
                    "text": line,
                }
            elif current:
                current["text"] += "\n" + line
            else:
                if not current:
                    current = {"heading": "Preamble", "type": "preamble", "text": ""}
                current["text"] += "\n" + line

        if current and current["text"].strip():
            sections.append(current)

        return sections

    def _split_large_section(
        self, text: str, section_info: dict, source: str
    ) -> list[dict[str, Any]]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "heading": section_info.get("heading", ""),
                "type": section_info.get("type", "unknown"),
                "source": source,
                "sub_chunk": True,
            })
        return chunks
