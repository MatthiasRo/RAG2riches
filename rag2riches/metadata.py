"""
Metadata utilities for constructing comparison cells.

This module provides tools for:
- Extracting unique values of metadata fields
- Constructing cell IDs from metadata combinations
- Building metadata filters for retrieval
"""

from typing import Any, Optional

import pandas as pd
from loguru import logger

from .types import Cell, Chunk


def get_unique_metadata_values(
    chunks: list[Chunk],
    field: str,
) -> list[Any]:
    """Get all unique values for a metadata field across chunks.

    Args:
        chunks: List of Chunk objects
        field: Metadata field name

    Returns:
        Sorted list of unique values
    """
    values = set()
    for chunk in chunks:
        if field in chunk.metadata:
            values.add(chunk.metadata[field])

    return sorted(list(values), key=str)


def construct_cell_id(field_values: dict[str, Any]) -> str:
    """Construct a cell ID from field-value pairs.

    Args:
        field_values: Dictionary of field -> value

    Returns:
        Cell ID string, e.g., "party=Democratic|year=2020|speaker=Alice"
    """
    items = []
    for key in sorted(field_values.keys()):
        value = field_values[key]
        items.append(f"{key}={value}")
    return "|".join(items)


def construct_cell_filter(field_values: dict[str, Any]) -> dict[str, Any]:
    """Construct a metadata filter for vector store queries.

    Args:
        field_values: Dictionary of field -> value

    Returns:
        Filter dictionary (format depends on vector store backend)
    """
    return field_values.copy()


def construct_cells(
    chunks: list[Chunk],
    cell_fields: list[str],
) -> list[Cell]:
    """Construct all unique cells from chunks and cell field definitions.

    A cell is a unique combination of values for the specified fields.

    Args:
        chunks: List of Chunk objects
        cell_fields: List of metadata field names that define cells

    Returns:
        List of unique Cell objects
    """
    logger.info(f"Constructing cells from {len(cell_fields)} fields: {cell_fields}")

    # Collect all combinations
    cell_dict = {}  # cell_id -> Cell

    for chunk in chunks:
        # Check that all required fields are present
        field_values = {}
        has_all_fields = True

        for field in cell_fields:
            if field not in chunk.metadata:
                has_all_fields = False
                break
            field_values[field] = chunk.metadata[field]

        if not has_all_fields:
            logger.warning(
                f"Chunk {chunk.chunk_id} missing required fields. "
                f"Required: {cell_fields}, found: {list(chunk.metadata.keys())}"
            )
            continue

        # Create cell ID and filter
        cell_id = construct_cell_id(field_values)
        cell_filter = construct_cell_filter(field_values)

        # Add to dictionary if not seen before
        if cell_id not in cell_dict:
            cell = Cell(cell_id=cell_id, fields=field_values, filter_expression=cell_filter)
            cell_dict[cell_id] = cell

    cells = list(cell_dict.values())
    logger.info(f"Constructed {len(cells)} unique cells")

    # Log cell summary
    for cell in cells[:5]:  # Log first 5
        logger.debug(f"  Cell: {cell.cell_id}")
    if len(cells) > 5:
        logger.debug(f"  ... and {len(cells) - 5} more")

    return cells


def chunks_for_cell(
    chunks: list[Chunk],
    cell: Cell,
) -> list[Chunk]:
    """Filter chunks to only those matching a cell's metadata.

    Args:
        chunks: List of all Chunk objects
        cell: Cell to filter for

    Returns:
        List of Chunks matching the cell
    """
    matching = []

    for chunk in chunks:
        matches = True
        for field, value in cell.fields.items():
            if chunk.metadata.get(field) != value:
                matches = False
                break
        if matches:
            matching.append(chunk)

    return matching


def chunks_dataframe(chunks: list[Chunk]) -> pd.DataFrame:
    """Convert chunks to a pandas DataFrame for inspection and export.

    Args:
        chunks: List of Chunk objects

    Returns:
        DataFrame with columns: chunk_id, document_id, text, chunk_index,
                               and flattened metadata columns
    """
    data = []

    for chunk in chunks:
        row = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
        # Add metadata columns
        row.update(chunk.metadata)
        data.append(row)

    df = pd.DataFrame(data)
    return df
