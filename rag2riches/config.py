"""
Configuration schemas for RAG2riches pipeline stages.

These Pydantic models define configuration for ingestion, cleaning, chunking,
embedding, and other pipeline stages.
"""

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""

    format: Literal["csv", "txt", "pdf"] = "csv"
    text_column: Optional[str] = None  # For CSV only
    metadata_columns: list[str] = Field(default_factory=list)  # For CSV only
    directory_path: Optional[str] = None  # For TXT directory ingestion
    document_id_column: Optional[str] = None  # Use existing column if present
    auto_generate_ids: bool = True
    encoding: str = "utf-8"  # For CSV only
    low_memory: bool = False  # For CSV only


class CleaningConfig(BaseModel):
    """Configuration for text cleaning."""

    normalize_whitespace: bool = True
    remove_extra_newlines: bool = True
    remove_headers_footers: bool = False
    header_footer_patterns: list[str] = Field(default_factory=list)  # Regex patterns
    strip_urls: bool = False
    strip_emails: bool = False
    lowercase: bool = False


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = 750
    chunk_overlap: int = 100
    method: Literal["character", "sentence"] = "character"
    preserve_paragraph_breaks: bool = True


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    model: str = "openai/text-embedding-3-small"
    batch_size: int = 32
    cache_embeddings: bool = True
    cache_dir: Path = Field(default_factory=lambda: Path(".cache/embeddings"))
    resume: bool = True
    max_retries: int = 3


class BatchEmbeddingConfig(BaseModel):
    """Configuration for batch embedding workflows."""

    provider: Literal["openai", "litellm"] = "openai"
    completion_window: str = "24h"
    output_dir: Path = Field(default_factory=lambda: Path(".batch"))
    table_name: str = "chunks"
    registry_table_name: str = "chunk_registry"
    batch_table_name: str = "batch_jobs"
    max_lines_per_jsonl: int = 50_000
    max_bytes_per_jsonl: int = 180 * 1024 * 1024
    poll_seconds: int = 20
    chunk_lookup_batch_size: int = 500
    ingest_batch_size: int = 500


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    store_type: Literal["lancedb"] = "lancedb"
    path: Path = Field(default_factory=lambda: Path("./rag2riches_store"))
    table_name: str = "chunks"
    create_if_missing: bool = True


class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""

    k: int = 5
    method: Literal["similarity"] = "similarity"
    min_score: Optional[float] = None


class GenerationConfig(BaseModel):
    """Configuration for response generation."""

    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None


class ExportConfig(BaseModel):
    """Configuration for export."""

    format: Literal["csv", "json", "parquet"] = "csv"
    output_path: Path
    include_retrieved_text: bool = True
    include_metadata: bool = True


class PipelineConfig(BaseModel):
    """Top-level configuration for the entire pipeline."""

    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    batch_embedding: BatchEmbeddingConfig = Field(default_factory=BatchEmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    verbose: bool = True
    log_file: Optional[Path] = None

