"""
Export utilities for RAG2riches data.

Supports exporting chunks and responses to CSV and JSON formats.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from .types import Chunk, ResponseRecord


def export_chunks_csv(
    chunks: list[Chunk],
    output_path: str | Path,
    include_text: bool = True,
) -> None:
    """Export chunks to CSV format.

    Args:
        chunks: List of Chunk objects
        output_path: Path to write CSV file
        include_text: If True, include chunk text in output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for chunk in chunks:
        row = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
        if include_text:
            row["text"] = chunk.text
        # Add metadata columns
        row.update(chunk.metadata)
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(chunks)} chunks to {output_path}")


def export_chunks_json(
    chunks: list[Chunk],
    output_path: str | Path,
) -> None:
    """Export chunks to JSON format (one chunk per line).

    Args:
        chunks: List of Chunk objects
        output_path: Path to write JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Exported {len(chunks)} chunks to {output_path}")


def export_responses_csv(
    responses: list[ResponseRecord],
    output_path: str | Path,
    include_context: bool = False,
    include_metadata: bool = True,
) -> None:
    """Export responses to CSV format.

    Args:
        responses: List of ResponseRecord objects
        output_path: Path to write CSV file
        include_context: If True, include retrieved context text
        include_metadata: If True, include response metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for resp in responses:
        row = {
            "response_id": resp.response_id,
            "query_id": resp.query_id,
            "query_text": resp.query_text,
            "cell_id": resp.cell_id,
            "response_text": resp.response_text,
            "model_name": resp.model_name,
            "embedding_model": resp.embedding_model_name,
            "retrieved_chunk_count": len(resp.retrieved_chunk_ids),
            "timestamp": resp.timestamp.isoformat(),
        }

        if include_context:
            row["retrieved_context"] = resp.retrieved_context

        # Add cell fields as separate columns for easier analysis
        if resp.cell_filter:
            row.update(resp.cell_filter)

        if include_metadata:
            # Store metadata as JSON string
            row["metadata"] = json.dumps(resp.metadata) if resp.metadata else ""

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(responses)} responses to {output_path}")


def export_responses_json(
    responses: list[ResponseRecord],
    output_path: str | Path,
) -> None:
    """Export responses to JSON format (one response per line).

    Args:
        responses: List of ResponseRecord objects
        output_path: Path to write JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for resp in responses:
            record = {
                "response_id": resp.response_id,
                "query_id": resp.query_id,
                "query_text": resp.query_text,
                "cell_id": resp.cell_id,
                "cell_filter": resp.cell_filter,
                "retrieved_chunk_ids": resp.retrieved_chunk_ids,
                "retrieved_context": resp.retrieved_context,
                "response_text": resp.response_text,
                "model_name": resp.model_name,
                "embedding_model_name": resp.embedding_model_name,
                "timestamp": resp.timestamp.isoformat(),
                "metadata": resp.metadata,
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Exported {len(responses)} responses to {output_path}")


def export_chunks(
    chunks: list[Chunk],
    output_path: str | Path,
    format: str = "csv",
) -> None:
    """Export chunks in specified format.

    Args:
        chunks: List of Chunk objects
        output_path: Path to write file
        format: One of "csv" or "json"
    """
    if format == "csv":
        export_chunks_csv(chunks, output_path)
    elif format == "json":
        export_chunks_json(chunks, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def export_responses(
    responses: list[ResponseRecord],
    output_path: str | Path,
    format: str = "csv",
) -> None:
    """Export responses in specified format.

    Args:
        responses: List of ResponseRecord objects
        output_path: Path to write file
        format: One of "csv" or "json"
    """
    if format == "csv":
        export_responses_csv(responses, output_path)
    elif format == "json":
        export_responses_json(responses, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

