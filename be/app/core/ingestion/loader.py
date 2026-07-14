from pathlib import Path
from typing import Optional

from app.infra.logging import logger


class DocumentLoader:
    SUPPORTED_FORMATS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "doc",
        ".txt": "text",
        ".rtf": "text",
        ".odt": "text",
        ".html": "html",
        ".htm": "html",
        ".epub": "epub",
        ".md": "markdown",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".tiff": "image",
        ".tif": "image",
    }

    def __init__(self, ocr_engine: str = "paddle"):
        self.ocr_engine = ocr_engine

    def load(self, path: Path) -> Optional[str]:
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            logger.warning("Unsupported format: %s", ext)
            return None

        format_type = self.SUPPORTED_FORMATS[ext]
        loader_map = {
            "pdf": self._load_pdf,
            "docx": self._load_docx,
            "text": self._load_text,
            "html": self._load_html,
            "epub": self._load_epub,
            "markdown": self._load_text,
            "image": self._load_image,
        }

        loader = loader_map.get(format_type)
        if not loader:
            return None

        try:
            text = loader(path)
            if text and text.strip():
                logger.info("Loaded %s (%d chars)", path.name, len(text))
                return text
            logger.warning("No text extracted from %s", path.name)
            return None
        except Exception as e:
            logger.error("Failed to load %s: %s", path.name, str(e))
            return None

    def _load_pdf(self, path: Path) -> Optional[str]:
        try:
            import fitz
        except ImportError:
            logger.error("PyMuPDF not installed")
            return None
        text_parts = []
        with fitz.open(path) as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")
        return "\n\n".join(text_parts) if text_parts else None

    def _load_docx(self, path: Path) -> Optional[str]:
        try:
            from docx import Document
        except ImportError:
            logger.error("python-docx not installed")
            return None
        doc = Document(path)
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras) if paras else None

    def _load_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding="latin-1")
            except Exception:
                return path.read_text(encoding="cp1252")

    def _load_html(self, path: Path) -> Optional[str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 not installed")
            return None
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    def _load_epub(self, path: Path) -> Optional[str]:
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            logger.error("ebooklib not installed")
            return None
        book = epub.read_epub(str(path))
        texts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(item.get_body_content(), "lxml")
                    texts.append(soup.get_text(separator="\n", strip=True))
                except Exception:
                    pass
        return "\n\n".join(texts) if texts else None

    def _load_image(self, path: Path) -> Optional[str]:
        if self.ocr_engine == "paddle":
            return self._ocr_paddle(path)
        return self._ocr_tesseract(path)

    def _ocr_paddle(self, path: Path) -> Optional[str]:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.warning("PaddleOCR not installed, falling back to tesseract")
            return self._ocr_tesseract(path)
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            result = ocr.ocr(str(path))
            lines = []
            if result and result[0]:
                for line_info in result[0]:
                    text = line_info[1][0]
                    if text.strip():
                        lines.append(text.strip())
            return "\n".join(lines) if lines else None
        except Exception as e:
            logger.error("PaddleOCR failed: %s", str(e))
            return None

    def _ocr_tesseract(self, path: Path) -> Optional[str]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            logger.error("pytesseract or pillow not installed")
            return None
        try:
            image = Image.open(path)
            text = pytesseract.image_to_string(image, lang="eng")
            return text.strip() or None
        except Exception as e:
            logger.error("Tesseract OCR failed: %s", str(e))
            return None
