"""
Retrieval module for RAG2riches.

Provides a simple Retriever class that embeds queries and performs
metadata-filtered similarity search in the configured vector store.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from .embeddings import Embedder
from .vectorstores import SearchResult, VectorStore


class Retriever:
    """Retriever that performs metadata-filtered similarity search."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, default_k: int = 5):
        if default_k <= 0:
            raise ValueError("default_k must be positive")
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_k = default_k

    def retrieve(
        self,
        query_text: str,
        cell_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve top-k chunks for a query within a metadata cell.

        Args:
            query_text: Query string used for retrieval.
            cell_filter: Metadata filter defining the cell.
            k: Number of results to return (uses default_k if None).

        Returns:
            List of SearchResult objects sorted by descending score.
        """
        if not query_text:
            raise ValueError("query_text must be non-empty")

        k = self.default_k if k is None else k
        if k <= 0:
            raise ValueError("k must be positive")

        logger.debug("Embedding query for retrieval")
        query_vector = self.embedder.embed_query(query_text)

        logger.debug("Running similarity search with metadata filter")
        results = self.vector_store.similarity_search(query_vector, k=k, filter=cell_filter)

        return results

