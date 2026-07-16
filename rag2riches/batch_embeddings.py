"""
Batch embeddings workflow for RAG2riches.

Provides JSONL batch request creation, batch submission, status polling,
and ingestion of embeddings into LanceDB.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from .types import Chunk, EmbeddingRecord
from .vectorstores.lancedb_store import LanceDBVectorStore


# Keep LanceDB scalar predicates small and predictable. Large `IN (...)`
# clauses can become slow or fail to parse when candidate ID lists are huge.
# The public `chunk_lookup_batch_size` still controls external batching; this
# cap only protects individual LanceDB filter expressions.
_MAX_IDS_PER_LANCEDB_FILTER = 500

# OpenAI-compatible Batch API terminal statuses. `expired` is terminal but may
# still provide an output file containing completed requests, so treat it as
# ingestible when `output_file_id` is present.
_TERMINAL_BATCH_STATUSES = {"completed", "failed", "cancelled", "expired"}
_INGESTIBLE_BATCH_STATUSES = {"completed", "cancelled", "expired"}
_BATCH_JOB_COLUMNS = [
    "batch_id",
    "provider",
    "model",
    "status",
    "input_file_id",
    "output_file_id",
    "error_file_id",
    "jsonl_path",
    "created_at",
    "updated_at",
    "ingested_at",
]


def _import_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm


def _progress_iter(
    items: Iterable[Chunk],
    *,
    total: int | None,
    desc: str,
    enabled: bool,
) -> Iterable[Chunk]:
    if not enabled:
        return items
    tqdm = _import_tqdm()
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc, unit="chunk")





@dataclass(frozen=True)
class BatchEmbeddingState:
    """Persistent configuration for batch embedding workflows."""

    vector_store_path: Path
    table_name: str
    registry_table_name: str
    batch_table_name: str
    output_dir: Path
    provider: str
    model: str
    completion_window: str
    poll_completion_window: str | None
    max_lines_per_jsonl: int
    max_bytes_per_jsonl: int
    poll_seconds: int
    chunk_lookup_batch_size: int
    ingest_batch_size: int
    queue_limit: int | None
    stop_on_queue_limit: bool
    cache_chunk_ids: bool
    full_scan_cache: bool
    download_max_retries: int
    download_backoff_seconds: float
    download_backoff_max: float
    api_key: str | None = None
    api_base: str | None = None
    organization: str | None = None
    extra_body: dict[str, Any] | None = None


@dataclass
class BatchJobRecord:
    batch_id: str
    provider: str
    model: str
    status: str
    input_file_id: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    jsonl_path: str | None = None
    created_at: int = 0
    updated_at: int = 0
    ingested_at: int | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "input_file_id": self.input_file_id,
            "output_file_id": self.output_file_id,
            "error_file_id": self.error_file_id,
            "jsonl_path": self.jsonl_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ingested_at": self.ingested_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "BatchJobRecord":
        return cls(
            batch_id=str(row.get("batch_id", "")),
            provider=str(row.get("provider", "")),
            model=str(row.get("model", "")),
            status=str(row.get("status", "")),
            input_file_id=_optional_str(row.get("input_file_id")),
            output_file_id=_optional_str(row.get("output_file_id")),
            error_file_id=_optional_str(row.get("error_file_id")),
            jsonl_path=_optional_str(row.get("jsonl_path")),
            created_at=_coerce_int(row.get("created_at")),
            updated_at=_coerce_int(row.get("updated_at")),
            ingested_at=_coerce_optional_int(row.get("ingested_at")),
        )


@dataclass
class BatchIngestionResult:
    batch_id: str
    records_added: int
    records_skipped: int


@dataclass
class BatchInfo:
    batch_id: str
    status: str
    input_file_id: str | None
    output_file_id: str | None
    error_file_id: str | None


class BatchOutputDownloadError(RuntimeError):
    """Raised when a completed batch output file cannot be downloaded."""

class OpenAIBatchClient:
    """OpenAI-compatible batch client wrapper with a minimal interface."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        organization: str | None = None,
    ):
        self._client = _create_openai_client(api_key, api_base, organization)

    def upload_file(self, path: Path) -> str:
        with path.open("rb") as f:
            file_obj = self._client.files.create(file=f, purpose="batch")
        return _get_attr(file_obj, "id")

    def create_batch(
        self,
        input_file_id: str,
        completion_window: str,
        metadata: dict[str, Any] | None = None,
    ) -> BatchInfo:
        batch = self._client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/embeddings",
            completion_window=completion_window,
            metadata=metadata,
        )
        return _batch_info_from(batch)

    def retrieve_batch(self, batch_id: str) -> BatchInfo:
        batch = self._client.batches.retrieve(batch_id)
        return _batch_info_from(batch)

    def download_file(
        self,
        file_id: str,
        max_retries: int = 5,
        backoff_seconds: float = 5.0,
        backoff_max: float = 60.0,
    ) -> bytes:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_seconds <= 0:
            raise ValueError("backoff_seconds must be positive")
        if backoff_max <= 0:
            raise ValueError("backoff_max must be positive")

        for attempt in range(max_retries + 1):
            try:
                content = self._client.files.content(file_id)
                if hasattr(content, "read"):
                    return content.read()
                if isinstance(content, (bytes, bytearray)):
                    return bytes(content)
                return str(content).encode("utf-8")
            except Exception as exc:
                if attempt >= max_retries or not _is_retryable_download_error(exc):
                    raise BatchOutputDownloadError(
                        f"Failed to download batch output file {file_id!r} "
                        f"after {attempt + 1} attempt(s)."
                    ) from exc
                wait_seconds = min(backoff_max, backoff_seconds * (2**attempt))
                logger.warning(
                    "Download failed for {} (attempt {}/{}). Retrying in {:.1f}s. Error: {}",
                    file_id,
                    attempt + 1,
                    max_retries,
                    wait_seconds,
                    exc,
                )
                time.sleep(wait_seconds)


class LanceDBBatchRegistry:
    """Persist chunk metadata and batch job state in LanceDB."""

    def __init__(
        self,
        path: Path,
        chunk_table: str = "chunk_registry",
        batch_table: str = "batch_jobs",
        cache_chunk_ids: bool = True,
        full_scan_cache: bool = False,
    ):
        lancedb = _import_lancedb()
        self._db = lancedb.connect(str(path))
        self._chunk_table_name = chunk_table
        self._batch_table_name = batch_table
        self._chunk_table = None
        self._batch_table = None
        self._cache_chunk_ids = cache_chunk_ids
        self._full_scan_cache = full_scan_cache
        self._chunk_id_cache: set[str] | None = None
        self._chunk_id_cache_complete = False
        self._warned_full_scan = False
        self._warned_full_scan_cache = False

        if self._cache_chunk_ids and self._full_scan_cache:
            self._initialize_chunk_id_cache()

    def add_chunks(
        self,
        chunks: Iterable[Chunk],
        batch_size: int = 1000,
        skip_existing: bool = True,
    ) -> int:
        added = 0
        for batch in _iter_chunk_batches(chunks, batch_size):
            rows = [_chunk_to_row(chunk) for chunk in batch]
            if skip_existing:
                existing = self.get_existing_chunk_ids([row["chunk_id"] for row in rows])
                rows = [row for row in rows if row["chunk_id"] not in existing]
            if not rows:
                continue
            self._append_rows(self._chunk_table_name, rows)
            added += len(rows)
        return added

    def get_existing_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return the subset of ``chunk_ids`` already present in the registry.

        This method is deliberately implemented as a bounded, chunked scalar
        lookup rather than one large SQL predicate. That matters when callers
        pass hundreds of thousands or millions of candidate IDs: LanceDB should
        see many small `IN (...)` filters, not one unparseably large filter
        string, and we should only project the `chunk_id` column.

        The return type remains a set to preserve the existing public API. If
        millions of candidate IDs match, the resulting Python set will still be
        large; avoiding that would require changing the caller-facing contract
        to stream or yield results incrementally.
        """
        if not chunk_ids:
            return set()

        if (
            self._cache_chunk_ids
            and self._chunk_id_cache_complete
            and self._chunk_id_cache is not None
        ):
            return {chunk_id for chunk_id in chunk_ids if chunk_id in self._chunk_id_cache}

        table = self._get_table(self._chunk_table_name)
        if table is None:
            return set()

        existing: set[str] = set()
        for id_batch in _iter_list_batches(chunk_ids, _MAX_IDS_PER_LANCEDB_FILTER):
            existing.update(
                _query_existing_ids(
                    table=table,
                    ids=id_batch,
                    id_column="chunk_id",
                    warned_flag_name="registry chunk_id lookup",
                )
            )

        return existing

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Return registry rows for the requested chunk IDs using bounded filters."""
        if not chunk_ids:
            return {}

        table = self._get_table(self._chunk_table_name)
        if table is None:
            return {}

        chunk_map: dict[str, Chunk] = {}
        columns = [
            "chunk_id",
            "document_id",
            "text",
            "chunk_index",
            "start_char",
            "end_char",
            "metadata_json",
        ]

        for id_batch in _iter_list_batches(chunk_ids, _MAX_IDS_PER_LANCEDB_FILTER):
            df = _query_rows_by_ids(
                table=table,
                ids=id_batch,
                id_column="chunk_id",
                columns=columns,
                warned_flag_name="registry chunk row lookup",
                fallback_on_empty=True,
            )
            if df.empty:
                continue
            for _, row in df.iterrows():
                chunk = _row_to_chunk(row.to_dict())
                chunk_map[chunk.chunk_id] = chunk

        return chunk_map

    def append_batch_event(self, record: BatchJobRecord) -> None:
        self._append_rows(self._batch_table_name, [record.to_row()])

    def latest_batch_records(self) -> dict[str, BatchJobRecord]:
        """Return the latest persisted event for each batch job.

        The batch table is an append-only event log. We still read all event
        rows when all records are requested, but we use LanceDB's query-builder
        API with explicit projection rather than `table.to_pandas()`. This keeps
        the interaction consistent with the rest of the module and avoids
        accidental full-column exports.
        """
        table = self._get_table(self._batch_table_name)
        if table is None:
            return {}

        try:
            df = table.search(None).select(_BATCH_JOB_COLUMNS).to_pandas()
        except Exception as exc:
            raise RuntimeError(
                "Failed to query LanceDB batch_jobs table with projected "
                "query-builder API. This is intentionally not falling back "
                "to a full table export."
            ) from exc

        if df.empty:
            return {}
        df = df.sort_values("updated_at")
        latest = df.groupby("batch_id", as_index=False).tail(1)
        records: dict[str, BatchJobRecord] = {}
        for _, row in latest.iterrows():
            record = BatchJobRecord.from_row(row.to_dict())
            if record.batch_id:
                records[record.batch_id] = record
        return records

    def latest_batch_record(self, batch_id: str) -> BatchJobRecord | None:
        """Return the latest event for one batch ID without scanning unrelated IDs."""
        if not batch_id:
            return None
        table = self._get_table(self._batch_table_name)
        if table is None:
            return None

        df = _query_rows_by_ids(
            table=table,
            ids=[batch_id],
            id_column="batch_id",
            columns=_BATCH_JOB_COLUMNS,
            warned_flag_name="batch_jobs latest record lookup",
        )
        if df.empty:
            return None
        latest = df.sort_values("updated_at").tail(1)
        if latest.empty:
            return None
        return BatchJobRecord.from_row(latest.iloc[0].to_dict())

    def pending_batch_ids(self, batch_ids: list[str] | None = None) -> list[str]:
        if batch_ids is None:
            latest = self.latest_batch_records()
            candidates = list(latest.keys())
        else:
            candidates = batch_ids
            latest = {
                batch_id: record
                for batch_id in candidates
                if (record := self.latest_batch_record(batch_id)) is not None
            }

        pending: list[str] = []
        for batch_id in candidates:
            record = latest.get(batch_id)
            if record is None:
                pending.append(batch_id)
                continue

            if record.status not in _TERMINAL_BATCH_STATUSES:
                pending.append(batch_id)
                continue

            if record.status in _INGESTIBLE_BATCH_STATUSES:
                if record.ingested_at is None:
                    pending.append(batch_id)
        return pending

    def _get_table(self, name: str):
        if name == self._chunk_table_name:
            if self._chunk_table is None:
                self._chunk_table = self._open_table(name)
            return self._chunk_table

        if name == self._batch_table_name:
            if self._batch_table is None:
                self._batch_table = self._open_table(name)
            return self._batch_table
        return self._open_table(name)

    def _open_table(self, name: str):
        table_names = _list_table_names(self._db)
        if name in table_names:
            return self._db.open_table(name)
        return None

    def _initialize_chunk_id_cache(self) -> None:
        if self._chunk_id_cache_complete:
            return
        table = self._get_table(self._chunk_table_name)
        if table is None:
            self._chunk_id_cache = set()
            self._chunk_id_cache_complete = True
            return
        self._load_chunk_id_cache(table)

    def _load_chunk_id_cache(self, table: Any) -> None:
        if self._chunk_id_cache_complete:
            return

        row_count = _safe_count_rows(table)
        if not self._warned_full_scan_cache:
            logger.warning(
                "full_scan_cache=True: loading all chunk_id values from registry table '{}' "
                "into memory (rows: {}). This may consume significant RAM.",
                self._chunk_table_name,
                row_count if row_count is not None else "unknown",
            )
            self._warned_full_scan_cache = True

        df = table.search(None).select(["chunk_id"]).to_pandas()
        if "chunk_id" in df.columns:
            self._chunk_id_cache = set(df["chunk_id"].astype(str).tolist())
        else:
            self._chunk_id_cache = set()
        self._chunk_id_cache_complete = True

    def _append_rows(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = self._get_table(name)
        if table is None:
            schema = _batch_jobs_schema() if name == self._batch_table_name else None
            table = self._db.create_table(name, data=rows, schema=schema, mode="overwrite")
            if name == self._chunk_table_name:
                self._chunk_table = table
            if name == self._batch_table_name:
                self._batch_table = table
        else:
            try:
                table.add(rows)
            except ValueError as exc:
                if name != self._batch_table_name or not _is_null_cast_error(exc):
                    raise
                logger.warning(
                    "Rewriting batch_jobs table with explicit schema after cast error"
                )
                existing_rows = []
                try:
                    df = table.search(None).select(_BATCH_JOB_COLUMNS).to_pandas()
                    if not df.empty:
                        existing_rows = df.to_dict(orient="records")
                except Exception as read_exc:
                    logger.warning(f"Failed to read batch_jobs table for rewrite: {read_exc}")
                combined_rows = existing_rows + rows
                schema = _batch_jobs_schema()
                table = self._db.create_table(
                    name,
                    data=combined_rows,
                    schema=schema,
                    mode="overwrite",
                )
                self._batch_table = table

        if name == self._chunk_table_name and self._cache_chunk_ids and self._chunk_id_cache_complete:
            if self._chunk_id_cache is None:
                self._chunk_id_cache = set()
            for row in rows:
                chunk_id = row.get("chunk_id")
                if chunk_id is not None:
                    self._chunk_id_cache.add(str(chunk_id))


class BatchEmbeddingManager:
    """Orchestrates batch embedding workflows backed by LanceDB."""

    def __init__(
        self,
        vector_store_path: str | Path,
        table_name: str = "chunks",
        registry_table_name: str = "chunk_registry",
        batch_table_name: str = "batch_jobs",
        output_dir: str | Path = ".batch",
        provider: str = "openai",
        model: str = "openai/text-embedding-3-small",
        completion_window: str = "24h",
        poll_completion_window: str | None = None,
        max_lines_per_jsonl: int = 5_000, # technically can go up to 50k, but this creates large files that may cause server-side timeouts and other headaches
        max_bytes_per_jsonl: int = 20 * 1024 * 1024,
        poll_seconds: int = 20,
        chunk_lookup_batch_size: int = 500,
        ingest_batch_size: int = 500,
        queue_limit: int | None = None,
        stop_on_queue_limit: bool = True,
        cache_chunk_ids: bool = True,
        full_scan_cache: bool = False,
        download_max_retries: int = 5,
        download_backoff_seconds: float = 5.0,
        download_backoff_max: float = 60.0,
        api_key: str | None = None,
        api_base: str | None = None,
        organization: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.state = BatchEmbeddingState(
            vector_store_path=Path(vector_store_path),
            table_name=table_name,
            registry_table_name=registry_table_name,
            batch_table_name=batch_table_name,
            output_dir=Path(output_dir),
            provider=provider,
            model=model,
            completion_window=completion_window,
            poll_completion_window=poll_completion_window or completion_window,
            max_lines_per_jsonl=max_lines_per_jsonl,
            max_bytes_per_jsonl=max_bytes_per_jsonl,
            poll_seconds=poll_seconds,
            chunk_lookup_batch_size=chunk_lookup_batch_size,
            ingest_batch_size=ingest_batch_size,
            queue_limit=queue_limit,
            stop_on_queue_limit=stop_on_queue_limit,
            cache_chunk_ids=cache_chunk_ids,
            full_scan_cache=full_scan_cache,
            download_max_retries=download_max_retries,
            download_backoff_seconds=download_backoff_seconds,
            download_backoff_max=download_backoff_max,
            api_key=api_key,
            api_base=api_base,
            organization=organization,
            extra_body=extra_body,
        )

        self.vector_store = LanceDBVectorStore(
            path=self.state.vector_store_path,
            table_name=self.state.table_name,
            cache_chunk_ids=self.state.cache_chunk_ids,
            full_scan_cache=self.state.full_scan_cache,
        )
        self.vector_store.create_or_connect(
            path=str(self.state.vector_store_path),
            table_name=self.state.table_name,
        )

        self.registry = LanceDBBatchRegistry(
            path=self.state.vector_store_path,
            chunk_table=self.state.registry_table_name,
            batch_table=self.state.batch_table_name,
            cache_chunk_ids=self.state.cache_chunk_ids,
            full_scan_cache=self.state.full_scan_cache,
        )

        self.last_batch_ids: list[str] = []
        self.last_jsonl_paths: list[Path] = []

    @classmethod
    def from_state(cls, state: BatchEmbeddingState) -> "BatchEmbeddingManager":
        return cls(
            vector_store_path=state.vector_store_path,
            table_name=state.table_name,
            registry_table_name=state.registry_table_name,
            batch_table_name=state.batch_table_name,
            output_dir=state.output_dir,
            provider=state.provider,
            model=state.model,
            completion_window=state.completion_window,
            poll_completion_window=state.poll_completion_window,
            max_lines_per_jsonl=state.max_lines_per_jsonl,
            max_bytes_per_jsonl=state.max_bytes_per_jsonl,
            poll_seconds=state.poll_seconds,
            chunk_lookup_batch_size=state.chunk_lookup_batch_size,
            ingest_batch_size=state.ingest_batch_size,
            queue_limit=state.queue_limit,
            stop_on_queue_limit=state.stop_on_queue_limit,
            cache_chunk_ids=state.cache_chunk_ids,
            full_scan_cache=state.full_scan_cache,
            download_max_retries=state.download_max_retries,
            download_backoff_seconds=state.download_backoff_seconds,
            download_backoff_max=state.download_backoff_max,
            api_key=state.api_key,
            api_base=state.api_base,
            organization=state.organization,
            extra_body=state.extra_body,
        )

    def prepare_requests(
        self,
        chunks: list[Chunk],
        skip_existing: bool = True,
        show_progress: bool = True,
        queue_limit: int | None = None,
        stop_on_queue_limit: bool | None = None,
    ) -> list[Path]:
        if not chunks:
            return []

        if queue_limit is None:
            queue_limit = self.state.queue_limit
        if stop_on_queue_limit is None:
            stop_on_queue_limit = self.state.stop_on_queue_limit
        if queue_limit is not None and queue_limit <= 0:
            raise ValueError("queue_limit must be positive when set")
        if stop_on_queue_limit is None:
            stop_on_queue_limit = True

        self.state.output_dir.mkdir(parents=True, exist_ok=True)

        if queue_limit is not None and stop_on_queue_limit:
            writer = JsonlBatchWriter(
                output_dir=self.state.output_dir,
                max_lines=self.state.max_lines_per_jsonl,
                max_bytes=self.state.max_bytes_per_jsonl,
            )

            written = 0
            added = 0
            queue_limit_reached = False

            def process_batch(batch: list[Chunk]) -> None:
                nonlocal written, added, queue_limit_reached
                if not batch or queue_limit_reached:
                    return
                if queue_limit is not None and written >= queue_limit:
                    queue_limit_reached = True
                    return

                if skip_existing:
                    existing = self._get_existing_vector_chunk_ids(
                        [chunk.chunk_id for chunk in batch]
                    )
                    candidates = [chunk for chunk in batch if chunk.chunk_id not in existing]
                else:
                    candidates = batch

                if queue_limit is not None:
                    remaining = max(queue_limit - written, 0)
                    if remaining <= 0:
                        queue_limit_reached = True
                        return
                    if remaining < len(candidates):
                        candidates = candidates[:remaining]

                if not candidates:
                    return

                added += self.registry.add_chunks(
                    candidates,
                    batch_size=self.state.chunk_lookup_batch_size,
                    skip_existing=skip_existing,
                )

                for chunk in candidates:
                    if queue_limit is not None and written >= queue_limit:
                        queue_limit_reached = True
                        break
                    line_obj = build_batch_request_line(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        model=_normalize_batch_model(self.state.model, self.state.provider),
                        extra_body=self.state.extra_body,
                    )
                    if writer.write(line_obj):
                        written += 1
                    if queue_limit is not None and written >= queue_limit:
                        queue_limit_reached = True
                        break

            register_iter = _progress_iter(
                chunks,
                total=len(chunks),
                desc="Registering chunks",
                enabled=show_progress,
            )
            batch_buffer: list[Chunk] = []
            for chunk in register_iter:
                batch_buffer.append(chunk)
                if len(batch_buffer) >= self.state.chunk_lookup_batch_size:
                    process_batch(batch_buffer)
                    batch_buffer = []
                    if queue_limit_reached:
                        break

            if batch_buffer and not queue_limit_reached:
                process_batch(batch_buffer)

            jsonl_paths = writer.close()
            self.last_jsonl_paths = list(jsonl_paths)
            logger.info(
                "Registered {} new chunk rows in registry (skip_existing={}).",
                added,
                skip_existing,
            )
            logger.info(
                "Prepared {} batch requests across {} files",
                written,
                len(jsonl_paths),
            )
            reused = written - added if written >= added else 0
            if skip_existing and reused:
                logger.info(
                    "Reused existing registry rows for {} requests.",
                    reused,
                )
            if queue_limit_reached:
                logger.warning(
                    "Stopped early after reaching queue_limit={} new requests. "
                    "Remaining chunks were not registered or written.",
                    queue_limit,
                )
            return jsonl_paths

        register_iter = _progress_iter(
            chunks,
            total=len(chunks),
            desc="Registering chunks",
            enabled=show_progress,
        )
        added = self.registry.add_chunks(
            register_iter,
            batch_size=self.state.chunk_lookup_batch_size,
            skip_existing=skip_existing,
        )
        logger.info(
            "Registered {} new chunk rows in registry (skip_existing={}).",
            added,
            skip_existing,
        )

        writer = JsonlBatchWriter(
            output_dir=self.state.output_dir,
            max_lines=self.state.max_lines_per_jsonl,
            max_bytes=self.state.max_bytes_per_jsonl,
        )

        written = 0
        queue_limit_reached = False
        total = None if skip_existing else len(chunks)
        write_iter = _progress_iter(
            self._iter_chunks_to_embed(chunks, skip_existing=skip_existing),
            total=total,
            desc="Writing batch requests",
            enabled=show_progress,
        )
        for chunk in write_iter:
            if queue_limit is not None and written >= queue_limit:
                queue_limit_reached = True
                break
            line_obj = build_batch_request_line(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                model=_normalize_batch_model(self.state.model, self.state.provider),
                extra_body=self.state.extra_body,
            )
            if writer.write(line_obj):
                written += 1
            if queue_limit is not None and written >= queue_limit:
                queue_limit_reached = True
                break

        jsonl_paths = writer.close()
        self.last_jsonl_paths = list(jsonl_paths)
        logger.info(
            "Prepared {} batch requests across {} files",
            written,
            len(jsonl_paths),
        )
        reused = written - added if written >= added else 0
        if skip_existing and reused:
            logger.info(
                "Reused existing registry rows for {} requests.",
                reused,
            )
        if queue_limit_reached:
            logger.warning(
                "Stopped writing batch requests after reaching queue_limit={}.",
                queue_limit,
            )
        return jsonl_paths

    def submit_batches(
        self,
        jsonl_paths: list[Path],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        if not jsonl_paths:
            return []

        client = _resolve_batch_client(
            provider=self.state.provider,
            api_key=self.state.api_key,
            api_base=self.state.api_base,
            organization=self.state.organization,
        )

        batch_ids: list[str] = []
        for path in jsonl_paths:
            input_file_id = client.upload_file(path)
            batch_info = client.create_batch(
                input_file_id=input_file_id,
                completion_window=self.state.completion_window,
                metadata=metadata,
            )
            batch_ids.append(batch_info.batch_id)
            ts = _now_unix()
            self.registry.append_batch_event(
                BatchJobRecord(
                    batch_id=batch_info.batch_id,
                    provider=self.state.provider,
                    model=self.state.model,
                    status=batch_info.status,
                    input_file_id=batch_info.input_file_id,
                    output_file_id=batch_info.output_file_id,
                    error_file_id=batch_info.error_file_id,
                    jsonl_path=str(path),
                    created_at=ts,
                    updated_at=ts,
                )
            )
            logger.info(f"Created batch {batch_info.batch_id} from {path.name}")

        self.last_batch_ids = list(batch_ids)
        return batch_ids

    def poll_and_ingest(
        self,
        batch_ids: list[str] | None = None,
        poll_seconds: int | None = None,
        continue_on_download_error: bool = False,
    ) -> list[BatchIngestionResult]:
        client = _resolve_batch_client(
            provider=self.state.provider,
            api_key=self.state.api_key,
            api_base=self.state.api_base,
            organization=self.state.organization,
        )
        poll_seconds = poll_seconds or self.state.poll_seconds
        poll_window_seconds = _parse_completion_window_seconds(
            self.state.poll_completion_window or self.state.completion_window
        )

        total_batches = len(self.registry.pending_batch_ids(batch_ids))

        results: list[BatchIngestionResult] = []
        skipped_downloads_this_run: set[str] = set()
        skipped_due_to_window: set[str] = set()
        ingested_batches = 0
        while True:
            pending = [
                batch_id
                for batch_id in self.registry.pending_batch_ids(batch_ids)
                if batch_id not in skipped_downloads_this_run
                and batch_id not in skipped_due_to_window
            ]
            if not pending:
                break

            progressed = False
            now_ts = _now_unix()
            for batch_id in pending:
                latest = self.registry.latest_batch_record(batch_id)
                if (
                    latest
                    and poll_window_seconds is not None
                    and latest.created_at
                    and now_ts > latest.created_at + poll_window_seconds
                ):
                    skipped_due_to_window.add(batch_id)
                    logger.warning(
                        "Stopping polling for batch {} because completion window ({}) elapsed. "
                        "created_at={}, deadline={}",
                        batch_id,
                        self.state.poll_completion_window or self.state.completion_window,
                        latest.created_at,
                        latest.created_at + poll_window_seconds,
                    )
                    continue
                if (
                    latest
                    and latest.status in _INGESTIBLE_BATCH_STATUSES
                    and latest.ingested_at is not None
                ):
                    continue

                batch_info = client.retrieve_batch(batch_id)
                ts = _now_unix()
                self.registry.append_batch_event(
                    BatchJobRecord(
                        batch_id=batch_info.batch_id,
                        provider=self.state.provider,
                        model=self.state.model,
                        status=batch_info.status,
                        input_file_id=batch_info.input_file_id,
                        output_file_id=batch_info.output_file_id,
                        error_file_id=batch_info.error_file_id,
                        created_at=latest.created_at if latest else ts,
                        updated_at=ts,
                        jsonl_path=latest.jsonl_path if latest else None,
                        ingested_at=latest.ingested_at if latest else None,
                    )
                )

                if batch_info.status in _INGESTIBLE_BATCH_STATUSES and batch_info.output_file_id:
                    if latest and latest.ingested_at is not None:
                        continue

                    try:
                        result = self._ingest_batch_output(batch_id, batch_info.output_file_id)
                    except BatchOutputDownloadError as exc:
                        if not continue_on_download_error:
                            raise

                        skipped_downloads_this_run.add(batch_id)
                        logger.warning(
                            "Skipping batch {} for this poll_and_ingest() run because its "
                            "output file could not be downloaded after retries. The batch is "
                            "not marked as ingested and will be retried on a later run. Error: {}",
                            batch_id,
                            exc,
                        )
                        continue

                    results.append(result)
                    progressed = True
                    ingested_batches += 1

                    ingested_ts = _now_unix()
                    self.registry.append_batch_event(
                        BatchJobRecord(
                            batch_id=batch_info.batch_id,
                            provider=self.state.provider,
                            model=self.state.model,
                            status=batch_info.status,
                            input_file_id=batch_info.input_file_id,
                            output_file_id=batch_info.output_file_id,
                            error_file_id=batch_info.error_file_id,
                            created_at=latest.created_at if latest else ingested_ts,
                            updated_at=ingested_ts,
                            jsonl_path=latest.jsonl_path if latest else None,
                            ingested_at=ingested_ts,
                        )
                    )
                    logger.info(
                        "Processed batch {} ({}/{} available)",
                        batch_id,
                        ingested_batches,
                        total_batches,
                    )
                elif batch_info.status in _TERMINAL_BATCH_STATUSES:
                    logger.warning(
                        f"Batch {batch_id} ended with terminal status "
                        f"{batch_info.status} and no ingestible output file"
                    )

            if not progressed:
                time.sleep(poll_seconds)

        return results

    def _iter_chunks_to_embed(self, chunks: list[Chunk], skip_existing: bool) -> Iterable[Chunk]:
        if not skip_existing:
            return iter(chunks)

        if not hasattr(self.vector_store, "get_existing_chunk_ids"):
            return iter(chunks)

        def generator():
            for batch in _iter_chunk_batches(chunks, self.state.chunk_lookup_batch_size):
                chunk_ids = [chunk.chunk_id for chunk in batch]
                existing = self._get_existing_vector_chunk_ids(chunk_ids)
                for chunk in batch:
                    if chunk.chunk_id in existing:
                        continue
                    yield chunk

        return generator()

    def _get_existing_vector_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return vector-store IDs using bounded LanceDB scalar lookups.

        This intentionally does *not* call ``self.vector_store.get_existing_chunk_ids``
        first, because older implementations of that method may silently full-scan
        the table after unsupported ``to_pandas(filter=..., columns=...)`` calls.
        Querying the LanceDB table directly here keeps this batch workflow
        scalable without changing its public inputs or outputs.
        """
        if not chunk_ids:
            return set()

        if self.state.full_scan_cache:
            cache = getattr(self.vector_store, "_chunk_id_cache", None)
            complete = getattr(self.vector_store, "_chunk_id_cache_complete", False)
            if complete and cache is not None:
                return {chunk_id for chunk_id in chunk_ids if chunk_id in cache}

        table = self.registry._get_table(self.state.table_name)
        if table is not None:
            existing: set[str] = set()
            for id_batch in _iter_list_batches(chunk_ids, _MAX_IDS_PER_LANCEDB_FILTER):
                existing.update(
                    _query_existing_ids(
                        table=table,
                        ids=id_batch,
                        id_column="chunk_id",
                        warned_flag_name="vector chunk_id lookup",
                    )
                )
            return existing

        # Last-resort compatibility path. This should rarely be needed because
        # the vector table is created during __init__, but it preserves behavior
        # for custom vector-store implementations.
        if hasattr(self.vector_store, "get_existing_chunk_ids"):
            try:
                return set(self.vector_store.get_existing_chunk_ids(chunk_ids))
            except Exception as exc:
                logger.warning(f"Vector store get_existing_chunk_ids failed: {exc}")

        return set()

    def _ingest_batch_output(self, batch_id: str, output_file_id: str) -> BatchIngestionResult:
        client = _resolve_batch_client(
            provider=self.state.provider,
            api_key=self.state.api_key,
            api_base=self.state.api_base,
            organization=self.state.organization,
        )
        content_bytes = client.download_file(
            output_file_id,
            max_retries=self.state.download_max_retries,
            backoff_seconds=self.state.download_backoff_seconds,
            backoff_max=self.state.download_backoff_max,
        )

        buffer: list[tuple[str, list[float]]] = []
        records_added = 0
        records_skipped = 0
        output_lines_seen = 0
        output_lines_parsed = 0
        output_lines_unparseable = 0

        for raw_line in content_bytes.splitlines():
            output_lines_seen += 1
            line = raw_line.decode("utf-8")
            parsed = parse_openai_batch_output_line(line)
            if not parsed:
                output_lines_unparseable += 1
                records_skipped += 1
                continue

            output_lines_parsed += 1
            buffer.append(parsed)

            if len(buffer) >= self.state.ingest_batch_size:
                added, skipped = self._ingest_vectors(buffer)
                records_added += added
                records_skipped += skipped
                buffer = []

        if buffer:
            added, skipped = self._ingest_vectors(buffer)
            records_added += added
            records_skipped += skipped

        logger.info(
            "Ingested batch {}: {} added, {} skipped. "
            "Output lines: {} seen, {} parsed, {} unparseable/error.",
            batch_id,
            records_added,
            records_skipped,
            output_lines_seen,
            output_lines_parsed,
            output_lines_unparseable,
        )

        return BatchIngestionResult(
            batch_id=batch_id,
            records_added=records_added,
            records_skipped=records_skipped,
        )

    def _ingest_vectors(self, items: list[tuple[str, list[float]]]) -> tuple[int, int]:
        if not items:
            return 0, 0

        chunk_ids = [item[0] for item in items]

        chunk_map = self.registry.get_chunks_by_ids(chunk_ids)
        existing = self._get_existing_vector_chunk_ids(chunk_ids)

        chunks: list[Chunk] = []
        embeddings: list[EmbeddingRecord] = []

        skipped_existing = 0
        skipped_missing_registry = 0

        for chunk_id, vector in items:
            if chunk_id in existing:
                skipped_existing += 1
                continue

            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                skipped_missing_registry += 1
                continue

            chunks.append(chunk)
            embeddings.append(
                EmbeddingRecord(
                    chunk_id=chunk_id,
                    vector=vector,
                    embedding_model=self.state.model,
                )
            )

        if skipped_missing_registry:
            logger.warning(
                "Skipped {} embedding records because their chunk IDs were not found "
                "in the LanceDB registry table '{}'. Example missing IDs: {}",
                skipped_missing_registry,
                self.state.registry_table_name,
                [
                    chunk_id
                    for chunk_id, _ in items
                    if chunk_id not in existing and chunk_id not in chunk_map
                ][:5],
            )

        if skipped_existing:
            logger.info(
                "Skipped {} embedding records because their chunk IDs already exist "
                "in vector table '{}'",
                skipped_existing,
                self.state.table_name,
            )

        if not chunks and skipped_missing_registry and not skipped_existing:
            raise RuntimeError(
                f"All {len(items)} parsed embedding records were missing from the "
                f"LanceDB registry table '{self.state.registry_table_name}'. "
                f"This usually means poll_and_ingest() is being run against a different "
                f"vector_store_path/registry_table_name than prepare_requests(), or the "
                f"batch output custom_id values do not match registered chunk_id values."
            )

        if chunks:
            self.vector_store.add_chunks(chunks, embeddings)

        return len(chunks), skipped_existing + skipped_missing_registry


class JsonlBatchWriter:
    def __init__(self, output_dir: Path, max_lines: int, max_bytes: int):
        self.output_dir = output_dir
        self.max_lines = max_lines
        self.max_bytes = max_bytes
        self.file_index = 0
        self.lines_written = 0
        self.bytes_written = 0
        self.created_files: list[Path] = []
        self._fh = None
        self._seen_ids: set[str] = set()
        self._open_new_file()

    def write(self, obj: dict[str, Any]) -> bool:
        if self._fh is None:
            self._open_new_file()
        line = json.dumps(obj, ensure_ascii=True) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if self.lines_written >= self.max_lines or self.bytes_written + line_bytes > self.max_bytes:
            self._rollover()

        custom_id = str(obj.get("custom_id", ""))
        if custom_id and custom_id in self._seen_ids:
            return False

        self._fh.write(line)
        if custom_id:
            self._seen_ids.add(custom_id)
        self.lines_written += 1
        self.bytes_written += line_bytes
        return True

    def close(self) -> list[Path]:
        self._finalize_file()
        return self.created_files

    def _open_new_file(self) -> None:
        path = self.output_dir / f"embeddings_requests_{self.file_index:06d}.jsonl"
        self._fh = path.open("w", encoding="utf-8", newline="\n")
        self.lines_written = 0
        self.bytes_written = 0
        self._seen_ids = set()
        self._current_path = path

    def _rollover(self) -> None:
        self._finalize_file()
        self.file_index += 1
        self._open_new_file()

    def _finalize_file(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        if self.lines_written > 0:
            self.created_files.append(self._current_path)
        self._fh = None


def build_batch_request_line(
    chunk_id: str,
    text: str,
    model: str,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {"model": model, "input": text}
    if extra_body:
        body.update(extra_body)
    return {
        "custom_id": chunk_id,
        "method": "POST",
        "url": "/v1/embeddings",
        "body": body,
    }


def parse_openai_batch_output_line(line: str) -> tuple[str, list[float]] | None:
    obj = json.loads(line)
    doc_id = obj.get("custom_id")
    resp = obj.get("response", {})
    status = resp.get("status_code")
    if status != 200:
        return None
    body = resp.get("body", {})
    data = body.get("data", [])
    if not data:
        return None
    embedding = data[0].get("embedding")
    if embedding is None:
        return None
    return str(doc_id), list(embedding)


def _iter_chunk_batches(items: Iterable[Chunk], batch_size: int) -> Iterable[list[Chunk]]:
    batch: list[Chunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _chunk_to_row(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "metadata_json": json.dumps(chunk.metadata, ensure_ascii=True),
        "created_at": _now_unix(),
    }


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    metadata_raw = row.get("metadata_json") or "{}"
    metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else {}
    return Chunk(
        chunk_id=str(row.get("chunk_id", "")),
        document_id=str(row.get("document_id", "")),
        text=str(row.get("text", "")),
        metadata=metadata,
        chunk_index=_coerce_int(row.get("chunk_index")),
        start_char=_coerce_optional_int(row.get("start_char")),
        end_char=_coerce_optional_int(row.get("end_char")),
    )


def _iter_list_batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    """Yield non-empty slices from a list without materializing another list of lists."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        if batch:
            yield batch


def _query_existing_ids(
    table: Any,
    ids: list[str],
    id_column: str,
    warned_flag_name: str,
) -> set[str]:
    """Query one bounded batch of IDs and project only the ID column."""
    if not ids:
        return set()

    df = _query_rows_by_ids(
        table=table,
        ids=ids,
        id_column=id_column,
        columns=[id_column],
        warned_flag_name=warned_flag_name,
    )
    if df.empty or id_column not in df.columns:
        return set()
    return set(df[id_column].astype(str).tolist())


def _query_rows_by_ids(
    table: Any,
    ids: list[str],
    id_column: str,
    columns: list[str] | None,
    warned_flag_name: str,
    fallback_on_empty: bool = False,
):
    """Query LanceDB rows by ID using the query-builder API.

    This function intentionally fails loudly if LanceDB cannot execute the
    filtered query. Earlier versions fell back to `table.to_pandas()` and then
    filtered in pandas, but that can silently full-scan very large vector
    tables. A hard failure is safer because it exposes version/backend/filter
    problems immediately instead of looking like a hang or memory leak.
    """
    import pandas as pd

    if not ids:
        return pd.DataFrame(columns=columns or [])

    def run_query(filter_expr: str):
        query = table.search(None).where(filter_expr)
        if columns:
            query = query.select(columns)
        query = query.limit(len(ids))
        return query.to_pandas()

    filter_expr = _id_filter_expression(id_column, ids, use_in=True)
    try:
        df = run_query(filter_expr)
        if df.empty and fallback_on_empty and len(ids) > 1:
            # Some older Lance/DataFusion parser combinations have been more
            # reliable with explicit OR chains than IN lists. This is still a
            # bounded, filtered LanceDB query, not a pandas-side full scan.
            fallback_expr = _id_filter_expression(id_column, ids, use_in=False)
            df = run_query(fallback_expr)
        return df
    except Exception as exc:
        raise RuntimeError(
            f"LanceDB {warned_flag_name} failed for a bounded batch of "
            f"{len(ids)} IDs using the query-builder API. Refusing to fall "
            f"back to table.to_pandas() because that can full-scan large "
            f"tables. Filter expression prefix: {filter_expr[:500]!r}"
        ) from exc


def _quote_identifier_if_needed(column: str) -> str:
    # LanceDB/DataFusion filters can behave unexpectedly with double-quoted
    # identifiers in some versions. Use bare identifiers for simple column names.
    if column.replace("_", "").isalnum() and not column[0].isdigit():
        return column
    return f'"{column.replace(chr(34), chr(34) + chr(34))}"'


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


def _chunk_filter_expression(chunk_ids: list[str]) -> str:
    # Kept for backward compatibility with any external code that imports this
    # helper from the module. New internal code uses _id_filter_expression.
    return _id_filter_expression("chunk_id", chunk_ids)


def _normalize_batch_model(model: str, provider: str) -> str:
    provider = provider.lower()
    if provider in {"openai", "litellm"} and "/" in model:
        prefix, remainder = model.split("/", 1)
        if prefix == "openai" and remainder:
            return remainder
    return model


def _create_openai_client(api_key: str | None, api_base: str | None, organization: str | None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI client is required for batch embeddings. "
            "Install with: pip install rag2riches[batch]"
        ) from exc

    params: dict[str, Any] = {}
    if api_key:
        params["api_key"] = api_key
    if api_base:
        params["base_url"] = api_base
    if organization:
        params["organization"] = organization
    return OpenAI(**params)


def _resolve_batch_client(
    provider: str,
    api_key: str | None,
    api_base: str | None,
    organization: str | None,
) -> OpenAIBatchClient:
    provider = provider.lower()
    if provider in {"openai", "litellm"}:
        if provider == "litellm":
            try:
                _import_litellm()
                logger.info("LiteLLM detected; using OpenAI-compatible batch client")
            except ImportError:
                logger.warning("LiteLLM not installed; using OpenAI client for batch")
        return OpenAIBatchClient(api_key=api_key, api_base=api_base, organization=organization)

    # TODO: Add provider-specific batch clients for Gemini or Anthropic when available.
    raise NotImplementedError(
        "Batch embeddings are only implemented for OpenAI-compatible endpoints. "
        "Add a provider-specific batch client to extend Gemini or Anthropic support."
    )


def _batch_info_from(batch: Any) -> BatchInfo:
    return BatchInfo(
        batch_id=_get_attr(batch, "id"),
        status=_get_attr(batch, "status"),
        input_file_id=_get_attr(batch, "input_file_id"),
        output_file_id=_get_attr(batch, "output_file_id"),
        error_file_id=_get_attr(batch, "error_file_id"),
    )


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        if value != value:  # NaN
            return 0
    except Exception:
        pass
    return int(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except Exception:
        pass
    return int(value)


def _now_unix() -> int:
    return int(time.time())


def _parse_completion_window_seconds(value: str) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+)\s*([smhdSMHD])\s*", value)
    if not match:
        logger.warning(
            "Unsupported completion window format: {!r}. Expected like '24h' or '90m'.",
            value,
        )
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 60 * 60
    if unit == "d":
        return amount * 24 * 60 * 60
    return None


def _import_lancedb():
    try:
        import lancedb
    except ImportError as exc:
        raise ImportError(
            "LanceDB is required for batch embeddings. "
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


def _batch_jobs_schema():
    try:
        import pyarrow as pa
    except ImportError:
        return None

    return pa.schema(
        [
            pa.field("batch_id", pa.string()),
            pa.field("provider", pa.string()),
            pa.field("model", pa.string()),
            pa.field("status", pa.string()),
            pa.field("input_file_id", pa.string()),
            pa.field("output_file_id", pa.string()),
            pa.field("error_file_id", pa.string()),
            pa.field("jsonl_path", pa.string()),
            pa.field("created_at", pa.int64()),
            pa.field("updated_at", pa.int64()),
            pa.field("ingested_at", pa.int64()),
        ]
    )


def _is_null_cast_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cannot cast field" in message and "to null" in message


def _is_retryable_download_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is not None:
        try:
            status_code = int(status)
        except Exception:
            status_code = None
        if status_code is not None:
            return 500 <= status_code < 600

    message = str(exc).lower()
    retry_markers = [
        "timeout",
        "timed out",
        "gateway time-out",
        "502",
        "503",
        "504",
        "server error",
    ]
    return any(marker in message for marker in retry_markers)


def _import_litellm():
    try:
        import litellm
    except ImportError as exc:
        raise ImportError(
            "LiteLLM is required to use provider='litellm'. "
            "Install with: pip install rag2riches[llm]"
        ) from exc
    return litellm
