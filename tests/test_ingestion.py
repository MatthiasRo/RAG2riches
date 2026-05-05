"""Tests for ingestion module."""

import tempfile
from pathlib import Path

import pytest

from rag2riches.ingestion import CSVIngester, TXTIngester, ingest_documents
from rag2riches.types import Document


class TestCSVIngester:
    """Test CSV ingestion."""

    def test_csv_basic_ingestion(self):
        """Test basic CSV ingestion."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        csv_path = fixtures_dir / "speeches_sample.csv"

        ingester = CSVIngester(
            text_column="speech_text",
            metadata_columns=["party", "year", "speaker"],
        )
        docs = ingester.ingest(csv_path)

        assert len(docs) == 6
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.text for d in docs)
        assert docs[0].metadata["party"] in ["Democrat", "Republican"]
        assert docs[0].metadata["year"] in [2020, 2022]

    def test_csv_document_id_generation(self):
        """Test automatic document ID generation."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        csv_path = fixtures_dir / "speeches_sample.csv"

        ingester = CSVIngester(
            text_column="speech_text",
            metadata_columns=["party"],
            auto_generate_ids=True,
        )
        docs = ingester.ingest(csv_path)

        # Check that IDs are generated and unique
        assert len(set(d.document_id for d in docs)) == len(docs)
        assert all(d.document_id for d in docs)

    def test_csv_missing_text_column_raises_error(self):
        """Test that missing text column raises ValueError."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        csv_path = fixtures_dir / "speeches_sample.csv"

        ingester = CSVIngester(text_column="nonexistent_column")

        with pytest.raises(ValueError, match="not found"):
            ingester.ingest(csv_path)

    def test_csv_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        ingester = CSVIngester(text_column="speech_text")

        with pytest.raises(FileNotFoundError):
            ingester.ingest("nonexistent.csv")


class TestTXTIngester:
    """Test TXT directory ingestion."""

    def test_txt_directory_ingestion(self):
        """Test ingesting multiple TXT files from a directory."""
        fixtures_dir = Path(__file__).parent / "fixtures"

        ingester = TXTIngester()
        docs = ingester.ingest(fixtures_dir)

        # Should find the two sample documents
        assert len(docs) >= 2
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.text for d in docs)
        assert all("filename" in d.metadata for d in docs)

    def test_txt_nonexistent_directory_raises_error(self):
        """Test that nonexistent directory raises error."""
        ingester = TXTIngester()

        with pytest.raises(NotADirectoryError):
            ingester.ingest("/nonexistent/path")


class TestIngestionFunction:
    """Test convenience ingestion function."""

    def test_ingest_csv(self):
        """Test ingest_documents with CSV format."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        csv_path = fixtures_dir / "speeches_sample.csv"

        docs = ingest_documents(
            csv_path,
            format="csv",
            text_column="speech_text",
            metadata_columns=["party", "year"],
        )

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_ingest_txt(self):
        """Test ingest_documents with TXT format."""
        fixtures_dir = Path(__file__).parent / "fixtures"

        docs = ingest_documents(fixtures_dir, format="txt")

        assert len(docs) >= 2

    def test_ingest_unsupported_format_raises_error(self):
        """Test that unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            ingest_documents(
                "test.csv",
                format="unsupported",
                text_column="text",
            )

    def test_ingest_csv_missing_text_column_raises_error(self):
        """Test that CSV without text_column raises error."""
        with pytest.raises(ValueError, match="text_column is required"):
            ingest_documents("test.csv", format="csv")

