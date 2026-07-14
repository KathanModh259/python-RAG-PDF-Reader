import pytest

from app.core.preprocessing.chunker import LegalChunker


@pytest.fixture
def chunker():
    return LegalChunker(chunk_size=100, chunk_overlap=20)


class TestLegalChunker:
    def test_chunk_simple_text(self, chunker):
        text = "This is a simple document. It has no legal structure."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all("text" in c for c in chunks)

    def test_chunk_with_articles(self, chunker):
        text = (
            "Article 1 - Preliminary\n"
            "This Act may be called the Test Act, 2024.\n"
            "Article 2 - Definitions\n"
            "In this Act, unless context otherwise requires..."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        article_chunks = [c for c in chunks if "article" in c.get("type", "")]
        assert len(article_chunks) >= 1

    def test_chunk_with_sections(self, chunker):
        text = (
            "Section 1 Short title\n"
            "This Act may be called the Test Act.\n"
            "Section 2 Definitions\n"
            "In this Act, the following definitions apply..."
        )
        chunks = chunker.chunk(text)
        section_chunks = [c for c in chunks if "section" in c.get("type", "")]
        assert len(section_chunks) >= 1

    def test_chunk_large_section(self, chunker):
        words = ["word"] * 300
        text = "Section 1 Test\n" + " ".join(words)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_empty_text(self, chunker):
        chunks = chunker.chunk("")
        assert len(chunks) >= 1
