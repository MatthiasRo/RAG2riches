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
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    verbose: bool = True
    log_file: Optional[Path] = None

