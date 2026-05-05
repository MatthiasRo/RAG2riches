"""Tests for metadata module."""

import pytest

from rag2riches.metadata import (
    chunks_dataframe,
    chunks_for_cell,
    construct_cell_id,
    construct_cells,
    get_unique_metadata_values,
)
from rag2riches.types import Cell, Chunk, Document


class TestConstructCellId:
    """Test cell ID construction."""

    def test_construct_cell_id_single_field(self):
        """Test constructing cell ID with one field."""
        cell_id = construct_cell_id({"party": "Democratic"})
        assert cell_id == "party=Democratic"

    def test_construct_cell_id_multiple_fields(self):
        """Test constructing cell ID with multiple fields."""
        cell_id = construct_cell_id({"party": "Democratic", "year": 2020})
        # Fields should be sorted alphabetically
        assert cell_id == "party=Democratic|year=2020"

    def test_construct_cell_id_ordering(self):
        """Test that cell IDs are consistent regardless of input order."""
        id1 = construct_cell_id({"year": 2020, "party": "D"})
        id2 = construct_cell_id({"party": "D", "year": 2020})
        assert id1 == id2


class TestConstructCells:
    """Test cell construction from chunks."""

    def test_construct_cells_basic(self):
        """Test basic cell construction."""
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
                metadata={"party": "D", "year": 2020},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c3",
                document_id="d3",
                text="text3",
                metadata={"party": "R", "year": 2020},
                chunk_index=0,
            ),
        ]

        cells = construct_cells(chunks, cell_fields=["party", "year"])

        assert len(cells) == 2
        cell_ids = {c.cell_id for c in cells}
        assert "party=D|year=2020" in cell_ids
        assert "party=R|year=2020" in cell_ids

    def test_construct_cells_three_fields(self):
        """Test cell construction with three fields."""
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="text1",
                metadata={"party": "D", "year": 2020, "outlet": "Times"},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d2",
                text="text2",
                metadata={"party": "D", "year": 2020, "outlet": "Post"},
                chunk_index=0,
            ),
        ]

        cells = construct_cells(chunks, cell_fields=["party", "year", "outlet"])

        assert len(cells) == 2

    def test_construct_cells_missing_field_warning(self):
        """Test that chunks missing required fields are skipped."""
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
                metadata={"party": "D"},  # Missing 'year'
                chunk_index=0,
            ),
        ]

        cells = construct_cells(chunks, cell_fields=["party", "year"])

        # Should only have one cell (c1)
        assert len(cells) == 1


class TestChunksForCell:
    """Test filtering chunks by cell."""

    def test_chunks_for_cell_basic(self):
        """Test basic chunk filtering."""
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
                metadata={"party": "D", "year": 2022},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c3",
                document_id="d3",
                text="text3",
                metadata={"party": "R", "year": 2020},
                chunk_index=0,
            ),
        ]

        cell = Cell(
            cell_id="party=D|year=2020",
            fields={"party": "D", "year": 2020},
        )

        matching = chunks_for_cell(chunks, cell)

        assert len(matching) == 1
        assert matching[0].chunk_id == "c1"

    def test_chunks_for_cell_multiple_matches(self):
        """Test filtering returns multiple chunks from same cell."""
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
                document_id="d1",
                text="text2",
                metadata={"party": "D", "year": 2020},
                chunk_index=1,
            ),
            Chunk(
                chunk_id="c3",
                document_id="d2",
                text="text3",
                metadata={"party": "R", "year": 2020},
                chunk_index=0,
            ),
        ]

        cell = Cell(
            cell_id="party=D|year=2020",
            fields={"party": "D", "year": 2020},
        )

        matching = chunks_for_cell(chunks, cell)

        assert len(matching) == 2
        assert {m.chunk_id for m in matching} == {"c1", "c2"}


class TestGetUniqueMetadataValues:
    """Test extracting unique metadata values."""

    def test_get_unique_values_single_field(self):
        """Test getting unique values for a field."""
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="text",
                metadata={"party": "D"},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d2",
                text="text",
                metadata={"party": "D"},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c3",
                document_id="d3",
                text="text",
                metadata={"party": "R"},
                chunk_index=0,
            ),
        ]

        values = get_unique_metadata_values(chunks, "party")

        assert set(values) == {"D", "R"}


class TestChunksDataframe:
    """Test converting chunks to DataFrame."""

    def test_chunks_dataframe_basic(self):
        """Test basic chunks to DataFrame conversion."""
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

        df = chunks_dataframe(chunks)

        assert len(df) == 2
        assert list(df["chunk_id"]) == ["c1", "c2"]
        assert list(df["party"]) == ["D", "R"]
        assert list(df["year"]) == [2020, 2020]

