"""
Core data types for RAG2riches.

This module defines the fundamental data structures used throughout the pipeline:
- Document: A source document with metadata
- Chunk: A text chunk derived from a document
- EmbeddingRecord: A chunk's vector embedding
- Cell: A metadata-defined comparison group
- ResponseRecord: A generated response to a query within a cell
"""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A source document with text and metadata.

    Attributes:
        document_id: Unique identifier for the document
        source_path: Path or source identifier where the document came from
        text: The full document text
        metadata: Dictionary of document-level metadata (e.g., party, year)
        raw: Optional raw content before cleaning (for debugging)
    """

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    source_path: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_001",
                "source_path": "speeches.csv:row_42",
                "text": "We must address climate change...",
                "metadata": {"party": "Democratic", "year": 2020, "speaker": "Alice"},
            }
        }
    )


class Chunk(BaseModel):
    """A text chunk derived from a document.

    Attributes:
        chunk_id: Unique identifier for the chunk
        document_id: Reference to parent document
        text: The chunk text
        metadata: Dictionary of metadata (merged from document + chunk-level)
        chunk_index: Position of chunk within the document
        start_char: Character offset where chunk starts in document text
        end_char: Character offset where chunk ends in document text
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "chunk_001",
                "document_id": "doc_001",
                "text": "We must address climate change with urgent action.",
                "metadata": {"party": "Democratic", "year": 2020},
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 45,
            }
        }
    )


class EmbeddingRecord(BaseModel):
    """Record of an embedded chunk.

    Attributes:
        chunk_id: Reference to the chunk
        vector: The embedding vector
        embedding_model: Name of the embedding model used
        timestamp: When the embedding was created
    """

    chunk_id: str
    vector: list[float]
    embedding_model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "chunk_001",
                "vector": [0.1, 0.2, 0.3, -0.1],
                "embedding_model": "openai/text-embedding-3-small",
            }
        }
    )


class Cell(BaseModel):
    """A metadata-defined comparison cell.

    A cell is a unique combination of metadata values that defines a subcorpus.
    For example, party=Democrat|year=2020 defines all documents from Democrats in 2020.

    Attributes:
        cell_id: Unique identifier (e.g., "party=D|year=2020")
        fields: Dictionary of field values that define this cell
        filter_expression: Metadata filter for retrieval (vector-store specific)
    """

    cell_id: str
    fields: dict[str, Any]
    filter_expression: Optional[Any] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cell_id": "party=Democratic|year=2020",
                "fields": {"party": "Democratic", "year": 2020},
                "filter_expression": {"party": "Democratic", "year": 2020},
            }
        }
    )


class QuerySpec(BaseModel):
    """Specification for a comparative query.

    Attributes:
        query_id: Unique query identifier
        query_text: The question to ask
        cell_fields: Which metadata fields define cells for comparison
        retrieval_k: Number of chunks to retrieve per cell
        additional_instructions: Extra instructions for generation
        generation_model: Which LLM model to use
        generation_params: Additional parameters for generation (temperature, etc.)
    """

    query_id: str = Field(default_factory=lambda: str(uuid4()))
    query_text: str
    cell_fields: list[str]
    retrieval_k: int = 5
    additional_instructions: Optional[str] = None
    generation_model: str = "gpt-3.5-turbo"
    generation_params: dict[str, Any] = Field(default_factory=dict)


class ResponseRecord(BaseModel):
    """A generated response to a query within a cell.

    Attributes:
        response_id: Unique identifier for this response
        query_id: Reference to the query
        query_text: The question asked
        cell_id: Which cell this response was generated for
        cell_filter: The metadata filter used for retrieval
        retrieved_chunk_ids: IDs of chunks that were retrieved
        retrieved_context: The concatenated retrieved text
        response_text: The generated response
        model_name: The LLM model that generated the response
        embedding_model_name: The embedding model used for retrieval
        timestamp: When the response was generated
        metadata: Additional metadata (error messages, retry counts, etc.)
    """

    response_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    query_text: str
    cell_id: str
    cell_filter: dict[str, Any] = Field(default_factory=dict)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_context: str = ""
    response_text: str = ""
    model_name: str = ""
    embedding_model_name: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response_id": "resp_001",
                "query_id": "q_001",
                "query_text": "How is climate change discussed?",
                "cell_id": "party=Democratic|year=2020",
                "cell_filter": {"party": "Democratic", "year": 2020},
                "retrieved_chunk_ids": ["chunk_001", "chunk_042"],
                "retrieved_context": "We must address climate...",
                "response_text": "The corpus emphasizes urgent action on climate...",
                "model_name": "gpt-4o-mini",
                "embedding_model_name": "text-embedding-3-small",
            }
        }
    )

