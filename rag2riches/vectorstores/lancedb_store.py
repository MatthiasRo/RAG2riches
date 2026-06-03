"""
LanceDB vector store implementation for RAG2riches.

This backend provides persistent vector storage with metadata pre-filtering.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from loguru import logger

from .base import SearchResult, VectorStore
from ..types import Chunk, EmbeddingRecord
                                                                                                         

_MAX_IDS_PER_LANCEDB_FILTER = 500
                                             

class LanceDBVectorStore(VectorStore):
    """LanceDB-backed vector store with metadata pre-filtering."""

    def __init__(
        self,
        path: str | Path | None = None,
        table_name: str = "chunks",
        cache_chunk_ids: bool = True,
        full_scan_cache: bool = False,
    ):
        self._path = Path(path) if path is not None else None
        self._table_name = table_name
        self._db = None
        self._table = None
        self._cache_chunk_ids = cache_chunk_ids
        self._full_scan_cache = full_scan_cache
        self._chunk_id_cache: set[str] | None = None
        self._chunk_id_cache_complete = False
        self._warned_full_scan = False
        self._warned_full_scan_cache = False
        self._reserved_columns = {
            "chunk_id",
            "document_id",
            "text",
            "vector",
            "chunk_index",
            "start_char",
            "end_char",
            "embedding_model",
            "metadata_json",
        }

    def create_or_connect(self, path: str | None = None, table_name: str | None = None) -> None:
        lancedb = _import_lancedb()

        if path is not None:
            self._path = Path(path)
        if table_name is not None:
            self._table_name = table_name

        self._chunk_id_cache = None
        self._chunk_id_cache_complete = False
        self._warned_full_scan = False

        if self._path is None:
            raise ValueError("path is required to create or connect to LanceDB")

        logger.info(f"Connecting to LanceDB at {self._path}")
        self._db = lancedb.connect(str(self._path))

        table_names = _list_table_names(self._db)

        if self._table_name in table_names:
            self._table = self._db.open_table(self._table_name)
            logger.info(f"Opened existing table '{self._table_name}'")
        else:
            self._table = None
            logger.info(f"Table '{self._table_name}' not found; will create on first add")

        if self._cache_chunk_ids and self._full_scan_cache:
            if self._table is None:
                self._chunk_id_cache = set()
                self._chunk_id_cache_complete = True
            else:
                self._load_chunk_id_cache()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[EmbeddingRecord]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if self._db is None:
            raise RuntimeError("Call create_or_connect before add_chunks")
        if not chunks:
            return

        rows, vector_dim, metadata_type_map = _build_lancedb_rows(
            chunks=chunks,
            embeddings=embeddings,
            reserved_columns=self._reserved_columns,
        )

        if self._table is None:
            schema = _build_pyarrow_schema(rows, vector_dim, metadata_type_map)
            create_kwargs: dict[str, Any] = {"data": rows, "mode": "overwrite"}
            if schema is not None:
                create_kwargs["schema"] = schema
            self._table = self._db.create_table(self._table_name, **create_kwargs)
            logger.info(f"Created LanceDB table '{self._table_name}' with {len(rows)} rows")
        else:
            rows = _align_rows_to_existing_schema(self._table, rows)
            self._table.add(rows)
            logger.info(f"Added {len(rows)} rows to LanceDB table '{self._table_name}'")

        if self._cache_chunk_ids and self._chunk_id_cache_complete:
            if self._chunk_id_cache is None:
                self._chunk_id_cache = set()
            for chunk in chunks:
                self._chunk_id_cache.add(chunk.chunk_id)

    def similarity_search(
        self,
        query_vector: list[float],
        k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._table is None or k <= 0:
            return []

        filter_expr = _build_filter_expression(filter)

        query = self._table.search(_coerce_vector(query_vector))
        if filter_expr:
            query = query.where(filter_expr)

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

        filter_expr = _id_filter_expression("chunk_id", [chunk_id])
        df = self._table.search(None).where(filter_expr).limit(1).to_pandas()

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

        schema_names = _schema_field_names(self._table)
        top_level_fields = [field for field in fields if field in schema_names]
        missing_fields = [field for field in fields if field not in schema_names]

        values: dict[str, list[Any]] = {field: [] for field in fields}

        if top_level_fields:
            df = self._table.search(None).select(top_level_fields).to_pandas()
            for field in top_level_fields:
                if field not in df.columns:
                    continue
                series = df[field].dropna().unique().tolist()
                values[field] = sorted(series, key=str)

        if missing_fields and "metadata_json" in schema_names:
            df_meta = self._table.search(None).select(["metadata_json"]).to_pandas()
            collected: dict[str, set[Any]] = {field: set() for field in missing_fields}
            for raw in df_meta.get("metadata_json", []):
                if not raw:
                    continue
                try:
                    metadata = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(metadata, dict):
                    continue
                for field in missing_fields:
                    value = metadata.get(field)
                    if value is not None:
                        try:
                            collected[field].add(value)
                        except TypeError:
                            collected[field].add(json.dumps(value, sort_keys=True))
            for field in missing_fields:
                values[field] = sorted(collected[field], key=str)

        return values

    def get_existing_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return the subset of chunk IDs already present in the table."""
        if self._table is None or not chunk_ids:
            return set()
        if self._cache_chunk_ids and self._chunk_id_cache_complete and self._chunk_id_cache is not None:
            return {chunk_id for chunk_id in chunk_ids if chunk_id in self._chunk_id_cache}

        existing: set[str] = set()
        for id_batch in _iter_list_batches(chunk_ids, _MAX_IDS_PER_LANCEDB_FILTER):
            filter_expr = _id_filter_expression("chunk_id", id_batch)
            df = (
                self._table.search(None)
                .where(filter_expr)
                .select(["chunk_id"])
                .limit(len(id_batch))
                .to_pandas()
            )
            if not df.empty and "chunk_id" in df.columns:
                existing.update(df["chunk_id"].astype(str).tolist())
        return existing

    def _load_chunk_id_cache(self) -> None:
        if self._table is None or self._chunk_id_cache_complete:
            return

        row_count = _safe_count_rows(self._table)
        if not self._warned_full_scan_cache:
            logger.warning(
                "full_scan_cache=True: loading all chunk_id values from vector table '{}' "
                "into memory (rows: {}). This may consume significant RAM.",
                self._table_name,
                row_count if row_count is not None else "unknown",
            )
            self._warned_full_scan_cache = True

        df = self._table.search(None).select(["chunk_id"]).to_pandas()
        if "chunk_id" in df.columns:
            self._chunk_id_cache = set(df["chunk_id"].astype(str).tolist())
        else:
            self._chunk_id_cache = set()
        self._chunk_id_cache_complete = True


def _build_lancedb_rows(
    *,
    chunks: list[Chunk],
    embeddings: list[EmbeddingRecord],
    reserved_columns: set[str],
) -> tuple[list[dict[str, Any]], int, dict[str, str]]:
    vector_lengths = {len(emb.vector) for emb in embeddings}
    if len(vector_lengths) != 1:
        raise ValueError(f"All embeddings must have the same dimension; got {sorted(vector_lengths)}")
    vector_dim = vector_lengths.pop()
    if vector_dim <= 0:
        raise ValueError("Embedding vectors must be non-empty")

    rows: list[dict[str, Any]] = []
    metadata_keys: set[str] = set()

    for chunk, emb in zip(chunks, embeddings):
        if chunk.chunk_id != emb.chunk_id:
            raise ValueError("chunk_id mismatch between chunk and embedding")

        conflicts = reserved_columns.intersection(chunk.metadata.keys())
        if conflicts:
            raise ValueError(
                "Metadata fields conflict with reserved columns: "
                f"{sorted(conflicts)}"
            )

        vector = _coerce_vector(emb.vector)
        if len(vector) != vector_dim:
            raise ValueError(
                f"Embedding dimension mismatch for chunk {chunk.chunk_id}: "
                f"expected {vector_dim}, got {len(vector)}"
            )

        metadata_json = json.dumps(_json_safe(chunk.metadata), ensure_ascii=True, sort_keys=True)
        row = {
            "chunk_id": str(chunk.chunk_id),
            "document_id": str(chunk.document_id),
            "text": str(chunk.text),
            "chunk_index": _coerce_int(chunk.chunk_index),
            "start_char": _coerce_optional_int(chunk.start_char),
            "end_char": _coerce_optional_int(chunk.end_char),
            "embedding_model": str(getattr(emb, "embedding_model", "")),
            "metadata_json": metadata_json,
            "vector": vector,
        }

        for key, value in chunk.metadata.items():
            safe_key = str(key)
            row[safe_key] = _metadata_scalar(value)
            metadata_keys.add(safe_key)
        rows.append(row)

    metadata_type_map = _infer_metadata_types(rows, metadata_keys)
    for row in rows:
        for key in metadata_keys:
            row[key] = _coerce_metadata_value(row.get(key), metadata_type_map[key])

    return rows, vector_dim, metadata_type_map


def _build_pyarrow_schema(
    rows: list[dict[str, Any]],
    vector_dim: int,
    metadata_type_map: dict[str, str],
):
    try:
        import pyarrow as pa
    except ImportError:
        return None

    fields = [
        pa.field("chunk_id", pa.string()),
        pa.field("document_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("chunk_index", pa.int64()),
        pa.field("start_char", pa.int64()),
        pa.field("end_char", pa.int64()),
        pa.field("embedding_model", pa.string()),
        pa.field("metadata_json", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
    ]

    reserved = {field.name for field in fields}
    metadata_keys = sorted(key for key in metadata_type_map if key not in reserved)
    for key in metadata_keys:
        kind = metadata_type_map[key]
        if kind == "bool":
            arrow_type = pa.bool_()
        elif kind == "int":
            arrow_type = pa.int64()
        elif kind == "float":
            arrow_type = pa.float64()
        else:
            arrow_type = pa.string()
        fields.append(pa.field(key, arrow_type))

    return pa.schema(fields)


def _align_rows_to_existing_schema(table: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schema_names = _schema_field_names(table)
    if not schema_names:
        return rows

    extra_columns = sorted({key for row in rows for key in row if key not in schema_names})
    if extra_columns:
        logger.warning(
            "Dropping {} metadata columns not present in existing LanceDB table '{}'. "
            "Full metadata remains stored in metadata_json. Example columns: {}",
            len(extra_columns),
            getattr(table, "name", "<unknown>"),
            extra_columns[:10],
        )

    aligned_rows: list[dict[str, Any]] = []
    for row in rows:
        aligned_rows.append({name: row.get(name) for name in schema_names if name in row})
    return aligned_rows


def _metadata_scalar(value: Any) -> Any:
    if value is None:
        return None
    value = _maybe_numpy_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True)


def _infer_metadata_types(rows: list[dict[str, Any]], metadata_keys: set[str]) -> dict[str, str]:
    type_map: dict[str, str] = {}
    for key in metadata_keys:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        if not values:
            type_map[key] = "string"
            continue
        if all(isinstance(value, bool) for value in values):
            type_map[key] = "bool"
            continue
        if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            type_map[key] = "int"
            continue
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            type_map[key] = "float"
            continue
        type_map[key] = "string"
    return type_map


def _coerce_metadata_value(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if isinstance(value, str):
        return value
    return json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True)


def _coerce_vector(vector: list[float]) -> list[float]:
    coerced: list[float] = []
    for value in vector:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Embedding vector contains NaN or infinite values")
        coerced.append(numeric)
    return coerced


def _json_safe(value: Any) -> Any:
    value = _maybe_numpy_scalar(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _maybe_numpy_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value


def _import_lancedb():
    try:
        import lancedb
    except ImportError as exc:
        raise ImportError(
            "LanceDB is required for LanceDBVectorStore. "
            "Install with: pip install rag2riches[vector]"
        ) from exc
    return lancedb


def _list_table_names(db: Any) -> set[str]:
    try:
        result = db.list_tables()
    except Exception:
        return set()

    if isinstance(result, (list, tuple, set)):
        return {str(item) for item in result}

    tables_attr = getattr(result, "tables", None)
    if tables_attr is not None:
        if isinstance(tables_attr, (list, tuple, set)):
            return {str(item) for item in tables_attr}
        try:
            return {str(item) for item in list(tables_attr)}
        except Exception:
            pass

    names_attr = getattr(result, "names", None)
    if callable(names_attr):
        try:
            names_val = names_attr()
            if isinstance(names_val, (list, tuple, set)):
                return {str(item) for item in names_val}
        except Exception:
            pass
    elif isinstance(names_attr, (list, tuple, set)):
        return {str(item) for item in names_attr}

    for attr in ("to_list", "tolist"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                names_val = fn()
                if isinstance(names_val, (list, tuple, set)):
                    return {str(item) for item in names_val}
            except Exception:
                pass

    try:
        return {str(item) for item in result}
    except Exception:
        return set()


def _safe_count_rows(table: Any) -> int | None:
    count_fn = getattr(table, "count_rows", None)
    if callable(count_fn):
        try:
            return int(count_fn())
        except Exception:
            return None
    return None


def _schema_field_names(table: Any) -> list[str]:
    schema = getattr(table, "schema", None)
    if schema is None:
        return []
    names = getattr(schema, "names", None)
    if names is not None:
        return list(names)
    try:
        return [field.name for field in schema]
    except Exception:
        return []


def _build_filter_expression(filter_dict: dict[str, Any] | None) -> str | None:
    if not filter_dict:
        return None

    raw_expr = filter_dict.get("__filter_expr__")
    if raw_expr:
        return str(raw_expr)

    if "__clauses__" in filter_dict:
        clauses = filter_dict.get("__clauses__", [])
        logic = str(filter_dict.get("__logic__", "and")).strip().lower()
        clause_exprs = [_build_filter_clause(clause) for clause in clauses]
        clause_exprs = [expr for expr in clause_exprs if expr]
        if not clause_exprs:
            return None
        joiner = " OR " if logic == "or" else " AND "
        return joiner.join(clause_exprs)

    clauses = []
    for key, value in filter_dict.items():
        if str(key).startswith("__"):
            continue
        if value is None:
            continue
        if isinstance(value, dict) and value.get("op") is not None:
            clause_expr = _build_filter_clause({"field": key, **value})
        elif isinstance(value, (list, tuple, set)):
            clause_expr = _build_filter_clause({"field": key, "op": "in", "value": list(value)})
        else:
            clause_expr = _build_filter_clause({"field": key, "op": "=", "value": value})
        if clause_expr:
            clauses.append(clause_expr)

    return " AND ".join(clauses) if clauses else None


def _build_filter_clause(clause: dict[str, Any]) -> str | None:
    field = clause.get("field")
    if not field:
        return None

    operator = _normalize_operator(str(clause.get("op", "=")).strip())
    field_expr = _quote_identifier_if_needed(str(field))
    value = clause.get("value")

    if operator in {"is null", "is not null"}:
        suffix = "IS NULL" if operator == "is null" else "IS NOT NULL"
        return f"{field_expr} {suffix}"

    if operator in {"in", "not in"}:
        values_expr = _format_filter_sequence(value)
        if not values_expr:
            return None
        keyword = "IN" if operator == "in" else "NOT IN"
        return f"{field_expr} {keyword} ({values_expr})"

    if value is None:
        return None

    formatted_value = _format_filter_value(value)
    if formatted_value is None:
        return None

    return f"{field_expr} {operator} {formatted_value}"


def _normalize_operator(operator: str) -> str:
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


def _format_filter_sequence(values: Any) -> str:
    if values is None:
        return ""
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    formatted = [_format_filter_value(value) for value in values]
    formatted = [value for value in formatted if value is not None]
    return ", ".join(formatted)


def _format_filter_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _extract_score(row: dict[str, Any]) -> float:
    if "_score" in row and row["_score"] is not None:
        return float(row["_score"])
    if "_distance" in row and row["_distance"] is not None:
        return -float(row["_distance"])
    return 0.0


def _row_to_chunk(row: dict[str, Any], reserved: set[str]) -> Chunk:
    metadata: dict[str, Any] = {}
    raw_metadata = row.get("metadata_json")
    if isinstance(raw_metadata, str) and raw_metadata:
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata.update(parsed)
        except Exception:
            pass

    metadata.update(
        {k: v for k, v in row.items() if k not in reserved and not k.startswith("_")}
    )

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


def _build_chunk_id_filter(chunk_ids: list[str]) -> str:
    return _id_filter_expression("chunk_id", chunk_ids)


def _id_filter_expression(id_column: str, ids: list[str], use_in: bool = True) -> str:
    column_expr = _quote_identifier_if_needed(id_column)
    escaped_ids = [str(item).replace("'", "''") for item in ids]
    if not escaped_ids:
        return ""

    if use_in and len(escaped_ids) > 1:
        quoted_ids = ", ".join(f"'{item}'" for item in escaped_ids)
        return f"{column_expr} IN ({quoted_ids})"

    clauses = [f"{column_expr} = '{item}'" for item in escaped_ids]
    return " OR ".join(clauses)


def _quote_identifier_if_needed(column: str) -> str:
    if column and not column[0].isdigit() and column.replace("_", "").isalnum():
        return column
    escaped = column.replace('"', '""')
    return f'"{escaped}"'


def _iter_list_batches(items: list[str], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        if batch:
            yield batch


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
