import re
from typing import Optional


class TextCleaner:
    def clean(self, text: str) -> str:
        text = self._normalize_whitespace(text)
        text = self._fix_ocr_issues(text)
        text = self._normalize_legal_refs(text)
        text = self._strip_headers_footers(text)
        return text.strip()

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\t+", " ", text)
        return text.strip()

    def _fix_ocr_issues(self, text: str) -> str:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
        text = re.sub(r"l\/l", "II", text)
        text = re.sub(r"([0-9]),([0-9]{3})", r"\1\2", text)
        return text

    def _normalize_legal_refs(self, text: str) -> str:
        text = re.sub(r"(?i)(Article|Section|Chapter|Schedule)\s+(\d+)", r"\1 \2", text)
        text = re.sub(r"(?i)(Clause|Rule|Regulation)\s+(\d+)", r"\1 \2", text)
        text = re.sub(
            r"(?i)(Part|Volume|Title)\s+([IVXLCDM]+)",
            lambda m: f"{m.group(1).capitalize()} {m.group(2).upper()}",
            text,
        )
        return text

    def _strip_headers_footers(self, text: str) -> str:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned.append("")
                continue
            if re.match(r"^\d+\s*$", stripped):
                continue
            if re.match(r"^Page\s+\d+\s+of\s+\d+$", stripped, re.IGNORECASE):
                continue
            if len(stripped) < 5 and stripped.isdigit():
                continue
            cleaned.append(stripped)
        return "\n".join(cleaned)
