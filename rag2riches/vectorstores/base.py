"""
Base vector store interface for RAG2riches.

Defines the core operations required for vector storage and retrieval with
metadata pre-filtering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..types import Chunk, EmbeddingRecord


@dataclass
class SearchResult:
    """Search result returned by a vector store."""

    chunk: Chunk
    score: float


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def create_or_connect(self, path: str | None = None, table_name: str | None = None) -> None:
        """Create or connect to a persistent vector store.

        Args:
            path: Optional storage path.
            table_name: Optional table name.
        """

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[EmbeddingRecord]) -> None:
        """Add chunks and their embeddings to the store.

        Args:
            chunks: List of Chunk objects.
            embeddings: List of EmbeddingRecord objects.
        """

    @abstractmethod
    def similarity_search(
        self,
        query_vector: list[float],
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for the most similar chunks with optional metadata filtering.

        Args:
            query_vector: Vector representing the query.
            k: Number of results to return.
            filter: Metadata filter applied before similarity search.

        Returns:
            List of SearchResult objects sorted by descending score.
        """

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by ID."""

    @abstractmethod
    def persist(self) -> None:
        """Persist the current state to storage (if applicable)."""

    @abstractmethod
    def list_metadata_values(self, fields: list[str]) -> dict[str, list[Any]]:
        """List unique values for metadata fields.

        Args:
            fields: Metadata fields to list values for.

        Returns:
            Dictionary mapping field -> list of unique values.
        """

