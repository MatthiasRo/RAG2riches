"""Tests for export module."""

import json
import tempfile
from pathlib import Path

import pytest

from rag2riches.export import (
    export_chunks_csv,
    export_chunks_json,
    export_responses_csv,
    export_responses_json,
)
from rag2riches.types import Chunk, ResponseRecord


class TestExportChunksCSV:
    """Test CSV chunk export."""

    def test_export_chunks_csv_basic(self):
        """Test basic CSV export."""
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="text1",
                metadata={"party": "D", "year": 2020},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d2",
                text="text2",
                metadata={"party": "R", "year": 2020},
                chunk_index=0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "chunks.csv"
            export_chunks_csv(chunks, output_path)

            assert output_path.exists()

            # Read and verify
            with open(output_path) as f:
                lines = f.readlines()
                assert len(lines) == 3  # Header + 2 chunks

    def test_export_chunks_csv_without_text(self):
        """Test CSV export excluding text."""
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="very long text" * 100,
                metadata={"party": "D"},
                chunk_index=0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "chunks.csv"
            export_chunks_csv(chunks, output_path, include_text=False)

            assert output_path.exists()

            # Verify text is not in output
            content = output_path.read_text()
            assert "very long text" not in content


class TestExportChunksJSON:
    """Test JSONL chunk export."""

    def test_export_chunks_json_basic(self):
        """Test basic JSONL export."""
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="text1",
                metadata={"party": "D", "year": 2020},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d2",
                text="text2",
                metadata={"party": "R", "year": 2020},
                chunk_index=0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "chunks.jsonl"
            export_chunks_json(chunks, output_path)

            assert output_path.exists()

            # Read and verify
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2

            # Parse JSON lines
            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data["chunk_id"] == chunks[i].chunk_id
                assert data["text"] == chunks[i].text


class TestExportResponsesCSV:
    """Test CSV response export."""

    def test_export_responses_csv_basic(self):
        """Test basic response CSV export."""
        responses = [
            ResponseRecord(
                response_id="r1",
                query_id="q1",
                query_text="How is climate discussed?",
                cell_id="party=D|year=2020",
                cell_filter={"party": "D", "year": 2020},
                retrieved_chunk_ids=["c1", "c2"],
                retrieved_context="Context text",
                response_text="Generated response",
                model_name="gpt-3.5-turbo",
                embedding_model_name="text-embedding-3-small",
            ),
            ResponseRecord(
                response_id="r2",
                query_id="q1",
                query_text="How is climate discussed?",
                cell_id="party=R|year=2020",
                cell_filter={"party": "R", "year": 2020},
                retrieved_chunk_ids=["c3"],
                retrieved_context="Different context",
                response_text="Different response",
                model_name="gpt-3.5-turbo",
                embedding_model_name="text-embedding-3-small",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "responses.csv"
            export_responses_csv(responses, output_path)

            assert output_path.exists()

            # Read and verify
            with open(output_path) as f:
                lines = f.readlines()
                assert len(lines) == 3  # Header + 2 responses

    def test_export_responses_csv_without_context(self):
        """Test response CSV export excluding context."""
        responses = [
            ResponseRecord(
                response_id="r1",
                query_id="q1",
                query_text="Test",
                cell_id="cell1",
                retrieved_context="Long context" * 100,
                response_text="Response",
                model_name="gpt-3.5-turbo",
                embedding_model_name="text-embedding-3-small",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "responses.csv"
            export_responses_csv(responses, output_path, include_context=False)

            content = output_path.read_text()
            assert "Long context" not in content


class TestExportResponsesJSON:
    """Test JSONL response export."""

    def test_export_responses_json_basic(self):
        """Test basic response JSONL export."""
        responses = [
            ResponseRecord(
                response_id="r1",
                query_id="q1",
                query_text="Test query",
                cell_id="cell1",
                cell_filter={"field": "value"},
                retrieved_chunk_ids=["c1"],
                retrieved_context="Context",
                response_text="Response",
                model_name="gpt-3.5-turbo",
                embedding_model_name="text-embedding-3-small",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "responses.jsonl"
            export_responses_json(responses, output_path)

            assert output_path.exists()

            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["response_id"] == "r1"
            assert data["response_text"] == "Response"

