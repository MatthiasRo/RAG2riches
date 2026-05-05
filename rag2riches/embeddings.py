"""
Embedding interfaces and implementations for RAG2riches.

This module defines a minimal Embedder interface, a deterministic MockEmbedder,
and a LiteLLM-backed embedder with batching, caching, and retries.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from .types import EmbeddingRecord


class Embedder(ABC):
    """Abstract embedder interface.

    Implementations should return dense vectors for text inputs.
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: List of input strings.

        Returns:
            List of vectors (one per text).
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a query string into a vector.

        Args:
            query: Input query string.

        Returns:
            Vector representing the query.
        """


class MockEmbedder(Embedder):
    """Deterministic mock embedder for tests and local runs.

    This embedder uses a stable hash to generate a pseudo-embedding with a
    fixed dimension. It is deterministic across runs for the same input.
    """

    def __init__(self, dim: int = 12, model_name: str = "mock-embedder"):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_to_vector(query)

    def embed_records(self, chunk_ids: list[str], texts: list[str]) -> list[EmbeddingRecord]:
        """Create EmbeddingRecord objects for chunks.

        Args:
            chunk_ids: Chunk IDs corresponding to texts.
            texts: Texts to embed.

        Returns:
            List of EmbeddingRecord objects.
        """
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")

        vectors = self.embed_texts(texts)
        records = [
            EmbeddingRecord(chunk_id=cid, vector=vec, embedding_model=self.model_name)
            for cid, vec in zip(chunk_ids, vectors)
        ]
        return records

    def _hash_to_vector(self, text: str) -> list[float]:
        # Use SHA256 to create a deterministic pseudo-embedding.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        bytes_data = bytes.fromhex(digest)

        # Convert bytes to floats in [-1, 1]
        vector = []
        for i in range(self.dim):
            byte_val = bytes_data[i % len(bytes_data)]
            scaled = (byte_val / 255.0) * 2.0 - 1.0
            vector.append(scaled)
        return vector


class LiteLLMEmbedder(Embedder):
    """LiteLLM-backed embedder with batching, caching, and retries."""

    def __init__(
        self,
        model: str,
        batch_size: int = 32,
        cache_path: str | Path | None = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 20.0,
        request_timeout: float | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.model = model
        self.model_name = model
        self.batch_size = batch_size
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.request_timeout = request_timeout
        self.api_base = api_base
        self.api_key = api_key
        self.organization = organization
        self.extra_params = extra_params or {}

        self._cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = _load_embedding_cache(self.cache_path, model=self.model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        hashes = [_hash_text(text) for text in texts]
        vectors: list[list[float] | None] = [None] * len(texts)

        missing: list[tuple[int, str, str]] = []
        for idx, (text, text_hash) in enumerate(zip(texts, hashes)):
            cached = self._cache.get(text_hash)
            if cached is not None:
                vectors[idx] = cached
            else:
                missing.append((idx, text, text_hash))

        if missing:
            new_cache_entries: list[tuple[str, list[float]]] = []
            for batch in _iter_batches(missing, self.batch_size):
                batch_texts = [item[1] for item in batch]
                batch_vectors = self._embed_batch(batch_texts)

                for (idx, _, text_hash), vector in zip(batch, batch_vectors):
                    vectors[idx] = vector
                    self._cache[text_hash] = vector
                    new_cache_entries.append((text_hash, vector))

            if self.cache_path and new_cache_entries:
                _append_embedding_cache(self.cache_path, new_cache_entries, model=self.model)

        if any(vector is None for vector in vectors):
            raise RuntimeError("Embedding failed for one or more inputs")

        return [vector for vector in vectors]  # type: ignore[return-value]

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed_texts([query])
        return vectors[0]

    def embed_records(self, chunk_ids: list[str], texts: list[str]) -> list[EmbeddingRecord]:
        """Create EmbeddingRecord objects for chunks."""
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")

        vectors = self.embed_texts(texts)
        return [
            EmbeddingRecord(chunk_id=chunk_id, vector=vector, embedding_model=self.model)
            for chunk_id, vector in zip(chunk_ids, vectors)
        ]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        litellm = _import_litellm()

        params: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key
        if self.organization:
            params["organization"] = self.organization
        if self.request_timeout is not None:
            params["request_timeout"] = self.request_timeout
        params.update(self.extra_params)

        response = _with_retries(
            lambda: litellm.embedding(**params),
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
            backoff_max=self.backoff_max,
        )

        data = response.get("data", []) if isinstance(response, dict) else response.data
        if data and isinstance(data, list) and "index" in data[0]:
            data = sorted(data, key=lambda item: item.get("index", 0))

        vectors = [item.get("embedding") for item in data]
        if any(vec is None for vec in vectors):
            raise ValueError("Embedding response missing vectors")

        logger.debug(f"Embedded {len(vectors)} texts with {self.model}")
        return vectors  # type: ignore[return-value]


def _import_litellm():
    try:
        import litellm
    except ImportError as exc:
        raise ImportError(
            "LiteLLM is required for LiteLLMEmbedder. "
            "Install with: pip install rag2riches[llm]"
        ) from exc
    return litellm


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_batches(items: list[tuple[int, str, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _with_retries(
    func: Callable[[], Any],
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
):
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - external failures vary
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_for = min(backoff_max, backoff_base * (2**attempt))
            time.sleep(sleep_for)
    if last_exc:
        raise last_exc
    raise RuntimeError("Embedding request failed")


def _load_embedding_cache(path: Path, model: str) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("model") != model:
                continue
            text_hash = data.get("text_hash")
            vector = data.get("vector")
            if text_hash and vector is not None:
                cache[text_hash] = vector
    return cache


def _append_embedding_cache(
    path: Path,
    entries: list[tuple[str, list[float]]],
    model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for text_hash, vector in entries:
            record = {
                "text_hash": text_hash,
                "model": model,
                "vector": vector,
            }
            f.write(json.dumps(record) + "\n")

