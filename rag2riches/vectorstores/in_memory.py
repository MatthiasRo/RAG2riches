"""
In-memory vector store for RAG2riches.

This lightweight implementation is intended for tests and small datasets.
It supports metadata pre-filtering before similarity search.
"""

from __future__ import annotations

import math
from typing import Any

from .base import SearchResult, VectorStore
from ..types import Chunk, EmbeddingRecord


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store with metadata filtering."""

    def __init__(self):
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def create_or_connect(self, path: str | None = None, table_name: str | None = None) -> None:
        # In-memory store does not persist data.
        return None

    def add_chunks(self, chunks: list[Chunk], embeddings: list[EmbeddingRecord]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        for chunk, emb in zip(chunks, embeddings):
            if chunk.chunk_id != emb.chunk_id:
                raise ValueError("chunk_id mismatch between chunk and embedding")
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = emb.vector

    def similarity_search(
        self,
        query_vector: list[float],
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            return []

        # Apply metadata filter before scoring
        filtered_chunk_ids = []
        for chunk_id, chunk in self._chunks.items():
            if self._matches_filter(chunk, filter):
                filtered_chunk_ids.append(chunk_id)

        results = []
        for chunk_id in filtered_chunk_ids:
            vec = self._vectors.get(chunk_id)
            if vec is None:
                continue
            score = self._cosine_similarity(query_vector, vec)
            results.append(SearchResult(chunk=self._chunks[chunk_id], score=score))

        # Sort by score descending and return top-k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def persist(self) -> None:
        # No-op for in-memory store.
        return None

    def list_metadata_values(self, fields: list[str]) -> dict[str, list[Any]]:
        values: dict[str, set[Any]] = {field: set() for field in fields}

        for chunk in self._chunks.values():
            for field in fields:
                if field in chunk.metadata:
                    values[field].add(chunk.metadata[field])

        return {field: sorted(list(vals), key=str) for field, vals in values.items()}

    def _matches_filter(self, chunk: Chunk, filter: dict[str, Any] | None) -> bool:
        if not filter:
            return True
        if "__clauses__" in filter:
            clauses = filter.get("__clauses__", [])
            logic = str(filter.get("__logic__", "and")).strip().lower()
            results = [self._matches_clause(chunk, clause) for clause in clauses]
            return any(results) if logic == "or" else all(results)

        for key, value in filter.items():
            if str(key).startswith("__"):
                continue
            if not self._matches_value(chunk.metadata.get(key), value):
                return False
        return True

    def _matches_clause(self, chunk: Chunk, clause: dict[str, Any]) -> bool:
        field = clause.get("field")
        if not field:
            return True
        value = chunk.metadata.get(field)
        operator = self._normalize_operator(str(clause.get("op", "=")).strip())
        expected = clause.get("value")
        return self._evaluate_clause(value, operator, expected)

    def _matches_value(self, actual: Any, expected: Any) -> bool:
        if isinstance(expected, dict) and expected.get("op") is not None:
            operator = self._normalize_operator(str(expected.get("op", "=")).strip())
            return self._evaluate_clause(actual, operator, expected.get("value"))
        if isinstance(expected, (list, tuple, set)):
            return actual in expected
        return actual == expected

    def _evaluate_clause(self, actual: Any, operator: str, expected: Any) -> bool:
        if operator == "is null":
            return actual is None
        if operator == "is not null":
            return actual is not None
        if actual is None:
            return False
        if operator == "in":
            if not isinstance(expected, (list, tuple, set)):
                return False
            return actual in expected
        if operator == "not in":
            if not isinstance(expected, (list, tuple, set)):
                return False
            return actual not in expected
        if operator == "=":
            return actual == expected
        if operator == "!=":
            return actual != expected
        try:
            if operator == ">":
                return actual > expected
            if operator == ">=":
                return actual >= expected
            if operator == "<":
                return actual < expected
            if operator == "<=":
                return actual <= expected
        except Exception:
            return False
        return False

    def _normalize_operator(self, operator: str) -> str:
        normalized = operator.strip().lower()
        mapping = {
            "==": "=",
            "eq": "=",
            "neq": "!=",
            "ne": "!=",
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
            "notin": "not in",
            "isnull": "is null",
            "isnotnull": "is not null",
        }
        return mapping.get(normalized, normalized)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError("Vector length mismatch")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

