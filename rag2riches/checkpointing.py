"""
Checkpointing utilities for RAG2riches.

Provides JSONL-based persistence and resume helpers for long-running runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from loguru import logger

from .types import ResponseRecord


def _response_to_dict(record: ResponseRecord) -> dict:
    data = record.model_dump()
    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat()
    return data


def append_response_records(records: Iterable[ResponseRecord], path: str | Path) -> int:
    """Append response records to a JSONL file.

    Args:
        records: Iterable of ResponseRecord objects.
        path: Output JSONL file path.

    Returns:
        Number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_response_to_dict(record)) + "\n")
            count += 1

    logger.info(f"Appended {count} records to {path}")
    return count


def load_response_records(path: str | Path) -> list[ResponseRecord]:
    """Load response records from a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of ResponseRecord objects.
    """
    path = Path(path)
    if not path.exists():
        return []

    records: list[ResponseRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(ResponseRecord(**data))

    return records


def read_processed_keys(
    path: str | Path,
    key_fields: tuple[str, str] = ("query_id", "cell_id"),
) -> set[tuple[str, str]]:
    """Read processed key tuples from a JSONL checkpoint file.

    Args:
        path: Path to JSONL file.
        key_fields: Fields used to identify a unique response.

    Returns:
        Set of key tuples.
    """
    path = Path(path)
    if not path.exists():
        return set()

    keys: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            key = tuple(str(data.get(field, "")) for field in key_fields)
            keys.add(key)  # type: ignore[arg-type]

    return keys


@dataclass
class CheckpointManager:
    """Checkpoint manager for response records."""

    path: Path
    key_fields: tuple[str, str] = ("query_id", "cell_id")

    def processed_keys(self) -> set[tuple[str, str]]:
        return read_processed_keys(self.path, key_fields=self.key_fields)

    def filter_unprocessed(self, records: Iterable[ResponseRecord]) -> list[ResponseRecord]:
        seen = self.processed_keys()
        remaining: list[ResponseRecord] = []
        for record in records:
            key = tuple(getattr(record, field, "") for field in self.key_fields)
            if key not in seen:
                remaining.append(record)
        return remaining

    def append(self, records: Iterable[ResponseRecord]) -> int:
        return append_response_records(records, self.path)

