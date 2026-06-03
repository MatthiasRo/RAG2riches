"""
RAG2riches: Comparative Retrieval-Augmented Generation for Social Science Research.

A Python library for comparing LLM-generated responses across metadata-defined
subsets of a text corpus. Unlike generic RAG, RAG2riches retrieves and generates
responses separately for each cell (e.g., party-year, firm-year) to ensure
grounding in the relevant subcorpus.

Basic usage:

    from rag2riches import RAG2richesPipeline

    pipeline = RAG2richesPipeline.from_csv(
        path="speeches.csv",
        text_column="speech_text",
        metadata_columns=["party", "year"]
    )

    pipeline.clean()
    pipeline.chunk(chunk_size=750, chunk_overlap=100)
    pipeline.export_chunks("output/chunks.csv")
"""

__version__ = "0.1.0a1"
__author__ = "Matthias Roesti"
__license__ = "MIT"

from .batch_embeddings import BatchEmbeddingManager, BatchEmbeddingState, BatchIngestionResult
from .cleaning import clean_documents, default_clean_text
from .chunking import chunk_text, chunks_from_documents
from .config import (
    BatchEmbeddingConfig,
    ChunkingConfig,
    CleaningConfig,
    EmbeddingConfig,
    ExportConfig,
    GenerationConfig,
    IngestionConfig,
    PipelineConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from .embeddings import Embedder, LiteLLMEmbedder, MockEmbedder
from .export import (
    export_chunks,
    export_chunks_csv,
    export_chunks_json,
    export_responses,
    export_responses_csv,
    export_responses_json,
)
from .compare import ComparativeRunner
from .checkpointing import CheckpointManager, append_response_records, load_response_records
from .generation import Generator, LiteLLMGenerator, MockGenerator
from .ingestion import CSVIngester, PDFIngester, TXTIngester, ingest_documents
from .logging_utils import get_logger, setup_logging
from .metadata import (
    chunks_dataframe,
    chunks_for_cell,
    construct_cells,
    construct_cell_id,
    get_unique_metadata_values,
)
from .pipeline import RAG2richesPipeline
from .prompts import DEFAULT_SYSTEM_PROMPT, PROMPT_VERSION, build_user_prompt, format_context
from .retrieval import Retriever
from .types import Cell, Chunk, Document, EmbeddingRecord, QuerySpec, ResponseRecord
from .vectorstores import LanceDBVectorStore, InMemoryVectorStore, SearchResult, VectorStore

__all__ = [
    # Types
    "Document",
    "Chunk",
    "Cell",
    "EmbeddingRecord",
    "BatchEmbeddingManager",
    "BatchEmbeddingState",
    "BatchIngestionResult",
    "QuerySpec",
    "ResponseRecord",
    "Embedder",
    "MockEmbedder",
    "LiteLLMEmbedder",
    "Generator",
    "MockGenerator",
    "LiteLLMGenerator",
    "ComparativeRunner",
    "CheckpointManager",
    "append_response_records",
    "load_response_records",
    "VectorStore",
    "SearchResult",
    "InMemoryVectorStore",
    "LanceDBVectorStore",
    "Retriever",
    "RAG2richesPipeline",
    "DEFAULT_SYSTEM_PROMPT",
    "PROMPT_VERSION",
    "build_user_prompt",
    "format_context",
    # Config
    "IngestionConfig",
    "CleaningConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "BatchEmbeddingConfig",
    "VectorStoreConfig",
    "RetrievalConfig",
    "GenerationConfig",
    "ExportConfig",
    "PipelineConfig",
    # Ingestion
    "ingest_documents",
    "CSVIngester",
    "TXTIngester",
    "PDFIngester",
    # Cleaning
    "clean_documents",
    "default_clean_text",
    # Chunking
    "chunk_text",
    "chunks_from_documents",
    # Metadata
    "construct_cells",
    "construct_cell_id",
    "chunks_for_cell",
    "chunks_dataframe",
    "get_unique_metadata_values",
    # Export
    "export_chunks",
    "export_chunks_csv",
    "export_chunks_json",
    "export_responses",
    "export_responses_csv",
    "export_responses_json",
    # Logging
    "setup_logging",
    "get_logger",
]

