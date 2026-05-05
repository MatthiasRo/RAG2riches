"""Tests for cleaning module."""

from rag2riches.cleaning import clean_documents, default_clean_text
from rag2riches.types import Document


class TestDefaultCleanText:
    """Test default text cleaning function."""

    def test_normalize_whitespace(self):
        """Test that multiple spaces are normalized."""
        text = "The   quick    brown   fox"
        cleaned = default_clean_text(text)
        assert cleaned == "The quick brown fox"

    def test_remove_extra_newlines(self):
        """Test that multiple newlines are normalized."""
        text = "Line 1\n\n\nLine 2\n\n\nLine 3"
        cleaned = default_clean_text(text)
        assert cleaned == "Line 1\nLine 2\nLine 3"

    def test_strip_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        text = "   test text   "
        cleaned = default_clean_text(text)
        assert cleaned == "test text"

    def test_mixed_whitespace(self):
        """Test cleaning with mixed whitespace issues."""
        text = "  The  quick   \n\n brown  fox  \n  jumps  "
        cleaned = default_clean_text(text)
        assert cleaned == "The quick\nbrown fox\njumps"


class TestCleanDocuments:
    """Test document cleaning."""

    def test_clean_documents_basic(self):
        """Test basic document cleaning."""
        docs = [
            Document(
                document_id="doc1",
                source_path="test.csv",
                text="  Text  with   spaces  ",
                metadata={},
            ),
            Document(
                document_id="doc2",
                source_path="test.csv",
                text="Another\n\n\ntext",
                metadata={},
            ),
        ]

        cleaned = clean_documents(docs)

        assert cleaned[0].text == "Text with spaces"
        assert cleaned[1].text == "Another\ntext"

    def test_clean_documents_preserves_raw(self):
        """Test that raw text is preserved."""
        docs = [
            Document(
                document_id="doc1",
                source_path="test.csv",
                text="  original  ",
                metadata={},
            ),
        ]

        cleaned = clean_documents(docs, preserve_raw=True)

        assert cleaned[0].raw == "  original  "
        assert cleaned[0].text == "original"

    def test_clean_documents_custom_function(self):
        """Test cleaning with custom function."""
        docs = [
            Document(
                document_id="doc1",
                source_path="test.csv",
                text="hello world",
                metadata={},
            ),
        ]

        def uppercase_clean(text):
            return text.upper()

        cleaned = clean_documents(docs, clean_fn=uppercase_clean)

        assert cleaned[0].text == "HELLO WORLD"

