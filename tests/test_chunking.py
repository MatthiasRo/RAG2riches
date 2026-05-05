"""Tests for chunking module."""

import pytest

from rag2riches.chunking import chunk_text, chunks_from_documents
from rag2riches.types import Chunk, Document


class TestChunkText:
    """Test text chunking function."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=250, chunk_overlap=50)

        # Should produce multiple chunks
        assert len(chunks) > 1
        # All chunks should be at most chunk_size
        assert all(len(c) <= 250 for c in chunks)
        # Concatenation with overlap should reconstruct most of the text
        assert len("".join(chunks)) > len(text)  # Due to overlap

    def test_chunk_text_short_text(self):
        """Test chunking text shorter than chunk_size."""
        text = "short text"
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_exact_size(self):
        """Test chunking text exactly matching chunk_size."""
        text = "a" * 250
        chunks = chunk_text(text, chunk_size=250, chunk_overlap=50)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        text = "test"

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text(text, chunk_size=0)

        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            chunk_text(text, chunk_size=10, chunk_overlap=10)

    def test_chunk_text_preserves_content(self):
        """Test that chunking preserves all content."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

        # Join chunks with overlap removed (simulate merging)
        reconstructed = chunks[0]
        for i in range(1, len(chunks)):
            # Remove the overlap from subsequent chunks
            reconstructed += chunks[i][50:]  # Skip first 50 chars (overlap)

        # Should contain most of the original text
        assert text[:100] in reconstructed or text[:50] in reconstructed


class TestChunksFromDocuments:
    """Test creating chunks from documents."""

    def test_chunks_from_documents_basic(self):
        """Test basic chunk creation from documents."""
        docs = [
            Document(
                document_id="doc1",
                source_path="test.csv",
                text="a" * 1000,
                metadata={"party": "A", "year": 2020},
            ),
            Document(
                document_id="doc2",
                source_path="test.csv",
                text="b" * 500,
                metadata={"party": "B", "year": 2020},
            ),
        ]

        chunks = chunks_from_documents(docs, chunk_size=250, chunk_overlap=50)

        assert len(chunks) > 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunks_inherit_metadata(self):
        """Test that chunks inherit document metadata."""
        doc = Document(
            document_id="doc1",
            source_path="test.csv",
            text="test text " * 100,
            metadata={"party": "Democratic", "year": 2020, "speaker": "Alice"},
        )

        chunks = chunks_from_documents([doc], chunk_size=100, chunk_overlap=10)

        # All chunks should have the document's metadata
        for chunk in chunks:
            assert chunk.metadata["party"] == "Democratic"
            assert chunk.metadata["year"] == 2020
            assert chunk.metadata["speaker"] == "Alice"
            assert chunk.document_id == "doc1"

    def test_chunks_have_unique_ids(self):
        """Test that each chunk gets a unique ID."""
        doc = Document(
            document_id="doc1",
            source_path="test.csv",
            text="a" * 1000,
            metadata={},
        )

        chunks = chunks_from_documents([doc], chunk_size=200, chunk_overlap=50)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunks_have_correct_indices(self):
        """Test that chunk_index is correct."""
        doc = Document(
            document_id="doc1",
            source_path="test.csv",
            text="a" * 1000,
            metadata={},
        )

        chunks = chunks_from_documents([doc], chunk_size=200, chunk_overlap=50)

        # Chunk indices should be sequential starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_custom_chunking_function(self):
        """Test using a custom chunking function."""
        doc = Document(
            document_id="doc1",
            source_path="test.csv",
            text="Sentence one. Sentence two. Sentence three.",
            metadata={},
        )

        # Custom function splits by period
        def split_by_period(text):
            return [s.strip() + "." for s in text.split(".") if s.strip()]

        chunks = chunks_from_documents([doc], chunk_fn=split_by_period)

        assert len(chunks) == 3
        assert chunks[0].text == "Sentence one."
        assert chunks[1].text == "Sentence two."
        assert chunks[2].text == "Sentence three."

