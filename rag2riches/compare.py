"""
Comparative runner for RAG2riches.

Runs a query across all metadata-defined cells and returns structured responses.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .checkpointing import CheckpointManager
from .generation import Generator
from .metadata import construct_cells
from .retrieval import Retriever
from .types import Chunk, QuerySpec, ResponseRecord


class ComparativeRunner:
    """Run comparative queries across metadata-defined cells."""

    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def run_query_across_cells(
        self,
        query_text: str,
        chunks: list[Chunk],
        cell_fields: list[str],
        k: int = 5,
        additional_instructions: str | None = None,
        generation_model: str | None = None,
        query_id: str | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> list[ResponseRecord]:
        """Run a single query across all cells.

        Args:
            query_text: Question to ask.
            chunks: All chunks with metadata.
            cell_fields: Metadata fields that define cells.
            k: Retrieval top-k per cell.
            additional_instructions: Extra generation instructions.
            generation_model: Optional model name for generator.

        Returns:
            List of ResponseRecord objects.
        """
        if not query_text:
            raise ValueError("query_text must be non-empty")
        if not cell_fields:
            raise ValueError("cell_fields must be non-empty")

        query_id = query_id or "q1"

        checkpoint_manager: CheckpointManager | None = None
        processed: set[tuple[str, str]] = set()
        if checkpoint_path is not None:
            checkpoint_manager = CheckpointManager(path=Path(checkpoint_path))
            if resume:
                processed = checkpoint_manager.processed_keys()

        cells = construct_cells(chunks, cell_fields=cell_fields)
        logger.info(f"Running query across {len(cells)} cells")

        responses: list[ResponseRecord] = []
        for cell in cells:
            if resume and (query_id, cell.cell_id) in processed:
                logger.info(f"Skipping already-processed cell {cell.cell_id}")
                continue
            try:
                results = self.retriever.retrieve(
                    query_text=query_text,
                    cell_filter=cell.filter_expression,
                    k=k,
                )
                retrieved_chunks = [r.chunk for r in results]
                retrieved_ids = [c.chunk_id for c in retrieved_chunks]
                retrieved_context = "\n\n".join(c.text for c in retrieved_chunks)

                response_text = self.generator.generate(
                    query_text=query_text,
                    retrieved_chunks=retrieved_chunks,
                    additional_instructions=additional_instructions,
                    model=generation_model,
                )

                generator_metadata = {}
                if hasattr(self.generator, "last_metadata"):
                    generator_metadata = getattr(self.generator, "last_metadata") or {}

                embedder_model = getattr(self.retriever.embedder, "model_name", "unknown")
                generator_model = (
                    generation_model or getattr(self.generator, "model_name", "mock")
                )

                responses.append(
                    ResponseRecord(
                        query_id=query_id,
                        query_text=query_text,
                        cell_id=cell.cell_id,
                        cell_filter=cell.fields,
                        retrieved_chunk_ids=retrieved_ids,
                        retrieved_context=retrieved_context,
                        response_text=response_text,
                        model_name=generator_model,
                        embedding_model_name=embedder_model,
                        timestamp=datetime.utcnow(),
                        metadata=generator_metadata,
                    )
                )
                if checkpoint_manager is not None:
                    checkpoint_manager.append([responses[-1]])
            except Exception as exc:
                logger.error(f"Cell {cell.cell_id} failed: {exc}")
                error_response = ResponseRecord(
                    query_id=query_id,
                    query_text=query_text,
                    cell_id=cell.cell_id,
                    cell_filter=cell.fields,
                    retrieved_chunk_ids=[],
                    retrieved_context="",
                    response_text="",
                    model_name=generation_model or getattr(self.generator, "model_name", "mock"),
                    embedding_model_name=getattr(
                        self.retriever.embedder, "model_name", "unknown"
                    ),
                    timestamp=datetime.utcnow(),
                    metadata={"error": str(exc)},
                )
                responses.append(error_response)
                if checkpoint_manager is not None:
                    checkpoint_manager.append([error_response])

        return responses

    def run_queries_across_cells(
        self,
        queries: list[QuerySpec],
        chunks: list[Chunk],
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> list[ResponseRecord]:
        """Run multiple queries across all cells.

        Args:
            queries: List of QuerySpec objects.
            chunks: All chunks with metadata.

        Returns:
            List of ResponseRecord objects.
        """
        responses: list[ResponseRecord] = []
        for query in queries:
            responses.extend(
                self.run_query_across_cells(
                    query_text=query.query_text,
                    chunks=chunks,
                    cell_fields=query.cell_fields,
                    k=query.retrieval_k,
                    additional_instructions=query.additional_instructions,
                    generation_model=query.generation_model,
                    query_id=query.query_id,
                    checkpoint_path=checkpoint_path,
                    resume=resume,
                )
            )
        return responses

