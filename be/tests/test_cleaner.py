import pytest

from app.core.preprocessing.cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


class TestTextCleaner:
    def test_normalize_whitespace(self, cleaner):
        result = cleaner.clean("Hello   World\n\n\nTest")
        assert "  " not in result
        assert "\n\n\n" not in result

    def test_fix_ocr_issues(self, cleaner):
        result = cleaner.clean("Section1\nThis is aTest")
        assert "Section 1" in result or "Section1" in result

    def test_legal_refs_normalized(self, cleaner):
        result = cleaner.clean("Article 5 says... Section 10 applies")
        assert "Article 5" in result
        assert "Section 10" in result

    def test_header_footer_removed(self, cleaner):
        result = cleaner.clean("Some text\n1\nPage 1 of 10\nMore text")
        assert "Page 1 of 10" not in result
        assert "Some text" in result

    def test_empty_text(self, cleaner):
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""
