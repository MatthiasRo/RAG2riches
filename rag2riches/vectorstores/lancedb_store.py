"""
LanceDB vector store implementation for RAG2riches.

This backend provides persistent vector storage with metadata pre-filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from .base import SearchResult, VectorStore
from ..types import Chunk, EmbeddingRecord


class LanceDBVectorStore(VectorStore):
    """LanceDB-backed vector store with metadata pre-filtering."""

    def __init__(self, path: str | Path | None = None, table_name: str = "chunks"):
        self._path = Path(path) if path is not None else None
        self._table_name = table_name
        self._db = None
        self._table = None
        self._reserved_columns = {
            "chunk_id",
            "document_id",
            "text",
            "vector",
            "chunk_index",
            "start_char",
            "end_char",
        }

    def create_or_connect(self, path: str | None = None, table_name: str | None = None) -> None:
        lancedb = _import_lancedb()

        if path is not None:
            self._path = Path(path)
        if table_name is not None:
            self._table_name = table_name

        if self._path is None:
            raise ValueError("path is required to create or connect to LanceDB")

        logger.info(f"Connecting to LanceDB at {self._path}")
        self._db = lancedb.connect(str(self._path))

        try:
            table_names = set(self._db.list_tables())
        except Exception:
            table_names = set()

        if self._table_name in table_names:
            self._table = self._db.open_table(self._table_name)
            logger.info(f"Opened existing table '{self._table_name}'")
        else:
            self._table = None
            logger.info(f"Table '{self._table_name}' not found; will create on first add")

    def add_chunks(self, chunks: list[Chunk], embeddings: list[EmbeddingRecord]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if self._db is None:
            raise RuntimeError("Call create_or_connect before add_chunks")

        rows: list[dict[str, Any]] = []
        for chunk, emb in zip(chunks, embeddings):
            if chunk.chunk_id != emb.chunk_id:
                raise ValueError("chunk_id mismatch between chunk and embedding")

            conflicts = self._reserved_columns.intersection(chunk.metadata.keys())
            if conflicts:
                raise ValueError(
                    "Metadata fields conflict with reserved columns: "
                    f"{sorted(conflicts)}"
                )

            row = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "vector": emb.vector,
            }
            row.update(chunk.metadata)
            rows.append(row)

        if self._table is None:
            self._table = self._db.create_table(self._table_name, data=rows, mode="overwrite")
            logger.info(f"Created LanceDB table '{self._table_name}' with {len(rows)} rows")
        else:
            self._table.add(rows)
            logger.info(f"Added {len(rows)} rows to LanceDB table '{self._table_name}'")

    def similarity_search(
        self,
        query_vector: list[float],
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._table is None or k <= 0:
            return []

        filter_expr = _build_filter_expression(filter)

        query = self._table.search(query_vector)
        if filter_expr:
            if hasattr(query, "where"):
                query = query.where(filter_expr)
            else:
                try:
                    query = self._table.search(query_vector, filter=filter_expr)
                except TypeError:
                    logger.warning("Filter expression not supported by this LanceDB version")

        results_df = query.limit(k).to_pandas()
        records = results_df.to_dict(orient="records")

        results: list[SearchResult] = []
        for row in records:
            score = _extract_score(row)
            chunk = _row_to_chunk(row, reserved=self._reserved_columns)
            results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        if self._table is None:
            return None

        escaped = chunk_id.replace("'", "''")
        try:
            df = self._table.to_pandas(filter=f"chunk_id = '{escaped}'")
        except TypeError:
            df = self._table.to_pandas()
            df = df[df.get("chunk_id") == chunk_id]

        if df.empty:
            return None

        row = df.iloc[0].to_dict()
        return _row_to_chunk(row, reserved=self._reserved_columns)

    def persist(self) -> None:
        # LanceDB persists on writes; no explicit flush required for typical use.
        return None

    def list_metadata_values(self, fields: list[str]) -> dict[str, list[Any]]:
        if self._table is None:
            return {field: [] for field in fields}

        try:
            df = self._table.to_pandas(columns=fields)
        except TypeError:
            df = self._table.to_pandas()
            df = df[fields]

        values: dict[str, list[Any]] = {}
        for field in fields:
            if field not in df.columns:
                values[field] = []
                continue
            series = df[field].dropna().unique().tolist()
            values[field] = sorted(series, key=str)

        return values


def _import_lancedb():
    try:
        import lancedb
    except ImportError as exc:
        raise ImportError(
            "LanceDB is required for LanceDBVectorStore. "
            "Install with: pip install rag2riches[vector]"
        ) from exc
    return lancedb


def _build_filter_expression(filter_dict: dict[str, Any] | None) -> str | None:
    if not filter_dict:
        return None

    clauses = []
    for key, value in filter_dict.items():
        if value is None:
            continue
        if isinstance(value, str):
            escaped = value.replace("'", "''")
            clauses.append(f"{key} = '{escaped}'")
        elif isinstance(value, bool):
            clauses.append(f"{key} = {str(value).lower()}")
        else:
            clauses.append(f"{key} = {value}")

    return " AND ".join(clauses) if clauses else None


def _extract_score(row: dict[str, Any]) -> float:
    if "_score" in row and row["_score"] is not None:
        return float(row["_score"])
    if "_distance" in row and row["_distance"] is not None:
        return -float(row["_distance"])
    return 0.0


def _row_to_chunk(row: dict[str, Any], reserved: set[str]) -> Chunk:
    metadata = {k: v for k, v in row.items() if k not in reserved and not k.startswith("_")}

    chunk_index = _coerce_int(row.get("chunk_index"))
    start_char = _coerce_optional_int(row.get("start_char"))
    end_char = _coerce_optional_int(row.get("end_char"))

    return Chunk(
        chunk_id=str(row.get("chunk_id")),
        document_id=str(row.get("document_id")),
        text=str(row.get("text", "")),
        metadata=metadata,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
    )


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        if value != value:  # NaN check
            return 0
    except Exception:
        pass
    return int(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN check
            return None
    except Exception:
        pass
    return int(value)

