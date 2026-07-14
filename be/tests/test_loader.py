from pathlib import Path

import pytest

from app.core.ingestion.loader import DocumentLoader


@pytest.fixture
def loader():
    return DocumentLoader()


class TestDocumentLoader:
    def test_load_text_file(self, loader, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello World", encoding="utf-8")
        result = loader.load(f)
        assert result == "Hello World"

    def test_load_markdown_file(self, loader, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nContent", encoding="utf-8")
        result = loader.load(f)
        assert result is not None
        assert "Title" in result

    def test_unsupported_format(self, loader, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("test")
        result = loader.load(f)
        assert result is None

    def test_supported_formats(self, loader):
        assert ".pdf" in loader.SUPPORTED_FORMATS
        assert ".docx" in loader.SUPPORTED_FORMATS
        assert ".txt" in loader.SUPPORTED_FORMATS
        assert ".html" in loader.SUPPORTED_FORMATS
        assert ".md" in loader.SUPPORTED_FORMATS
