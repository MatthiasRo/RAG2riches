# Architecture

## Overview

RAG2riches implements a modular pipeline for comparative retrieval-augmented generation. The distinctive feature is **metadata-filtered pre-retrieval**: retrieval is constrained to user-defined cells before vector search, ensuring responses are grounded in the relevant subcorpus only.

For most users, the high-level `RAG2richesPipeline` in `rag2riches/pipeline.py` provides an end-to-end interface that wraps ingestion, cleaning, chunking, embedding, indexing, and comparative runs.

A Streamlit UI in `rag2riches/ui/streamlit_app.py` sits on top of the same pipeline and is intended for exploratory use.

## Pipeline Stages

### 1. Ingestion (`rag2riches/ingestion.py`)

**Purpose**: Load documents from various formats.

**Supported Formats**:
- **CSV**: Tabular data with configurable text and metadata columns
- **TXT**: Directory of plain-text files
- **PDF**: Single or multiple PDF files (current extraction is full text; page-level metadata planned)

**Key Classes**:
- `CSVIngester`: Loads CSV files, maps columns to text and metadata
- `TXTIngester`: Loads all `.txt` files from a directory
- `PDFIngester`: Extracts text from PDF files

**Output**: List of `Document` objects with metadata

### 2. Cleaning (`rag2riches/cleaning.py`)

**Purpose**: Normalize and clean raw text before chunking.

**Default Operations**:
- Normalize multiple spaces to single space
- Normalize multiple newlines to single newline
- Strip leading/trailing whitespace
- Preserve original raw text for debugging

**Customization**: Users can provide custom cleaning functions

**Output**: List of cleaned `Document` objects

### 3. Chunking (`rag2riches/chunking.py`)

**Purpose**: Split documents into overlapping chunks for embedding.

**Default Strategy**: Character-based chunking with configurable size and overlap

**Parameters**:
- `chunk_size`: Maximum characters per chunk (default: 750)
- `chunk_overlap`: Characters to overlap between chunks (default: 100)

**Custom Chunking**: Users can provide functions that split text by sentences, paragraphs, or custom rules

**Key Feature**: Each chunk inherits all document-level metadata and adds:
- `chunk_id`: Unique identifier
- `chunk_index`: Position in document
- `start_char`, `end_char`: Character offsets

**Output**: List of `Chunk` objects with inherited and enriched metadata

### 4. Metadata Management (`rag2riches/metadata.py`)

**Purpose**: Construct comparison cells and filter chunks by cell.

**Key Functions**:
- `construct_cell_id()`: Create unique cell ID from metadata values (e.g., "party=D|year=2020")
- `construct_cells()`: Identify all unique cells from chunk metadata
- `chunks_for_cell()`: Filter chunks to those matching a cell's metadata
- `get_unique_metadata_values()`: Extract unique values for a field

**Cell Definition**: User specifies `cell_fields = ["party", "year"]`, and the system identifies all unique combinations

**Output**: List of `Cell` objects and filtered chunk subsets

### 5. Export (`rag2riches/export.py`)

**Purpose**: Write chunks and responses to disk for inspection and downstream analysis.

**Supported Formats**:
- **CSV**: Flat, tabular format suitable for Excel or R
- **JSON/JSONL**: Nested format preserving all metadata and structure

**Export Functions**:
- `export_chunks_csv()`: Write chunks to CSV
- `export_chunks_json()`: Write chunks to JSONL (one per line)
- `export_responses_csv()`: Write responses to CSV
- `export_responses_json()`: Write responses to JSONL

**Output**: Files ready for analysis, visualization, or further processing

### 6. Embedding

**Purpose**: Convert text to vector embeddings.

**Architecture**:
- Abstract `Embedder` interface
- `MockEmbedder`: Deterministic local embeddings for tests
- `LiteLLMEmbedder`: Uses LiteLLM for unified provider support (implemented)
- Supports OpenAI, Gemini, Claude, Ollama, and other endpoints
- Batching for efficiency
- Caching to avoid recomputing embeddings
- Checkpointing for resumable runs

**Output**: `EmbeddingRecord` objects with vector, chunk_id, and metadata

### 7. Vector Store

**Purpose**: Store embeddings and metadata for fast retrieval.

**Architecture**:
- Abstract `VectorStore` interface
- `InMemoryVectorStore`: Local, testable backend (implemented)
- `LanceDBVectorStore`: Production implementation using LanceDB (implemented)
- Supports metadata pre-filtering before similarity search

**Key Feature**: Metadata filters are applied **before** vector search, not after

**Methods**:
- `add_chunks()`: Insert embeddings
- `similarity_search()`: Query with optional metadata filter
- `get_chunk()`: Retrieve by ID
- `persist()`: Save to disk

**Output**: Persistent vector database with indexed metadata

### 8. Retrieval

**Purpose**: Find relevant chunks for a query within a specific cell.

**Workflow**:
1. Embed the query text
2. Construct metadata filter for the cell
3. Search vector store with filter applied
4. Return top-k chunks with similarity scores

**Key Feature**: Retrieval is constrained to the cell; global retrieval is never performed

**Output**: List of retrieved `Chunk` objects with scores

### 9. Generation

**Purpose**: Generate LLM responses grounded in retrieved context.

**Architecture**:
- Abstract `Generator` interface
- `MockGenerator`: Deterministic local responses for tests
- `LiteLLMGenerator`: Uses LiteLLM for unified provider support (implemented)
- Supports OpenAI, Gemini, Claude, Ollama, and other endpoints

**System Prompt**: Instructs the model to:
- Answer only from provided context
- Say "I don't know" if answer cannot be inferred
- Not use outside knowledge

**Additional Instructions**: User can provide custom instructions appended to the generation prompt

**Prompt Templates**: Defined in `rag2riches/prompts.py` (`DEFAULT_SYSTEM_PROMPT`, `build_user_prompt`).

**Output**: `ResponseRecord` with generated text, metadata, and provenance

### 10. Comparative Runner

**Purpose**: Execute a query across all cells in the dataset.

**Workflow**:
1. Identify all unique cells from metadata
2. For each cell:
   a. Construct metadata filter
   b. Retrieve chunks for that cell only
   c. Generate response from those chunks
   d. Store `ResponseRecord`
3. Continue even if individual cells fail
4. Support resuming interrupted runs via checkpoint files

**Resilience**: Errors in one cell do not stop the pipeline; failures are logged

**Output**: List of `ResponseRecord` objects, one per cell per query

### 11. Checkpointing

**Purpose**: Resume long-running comparative runs without repeating completed work.

**Architecture**:
- JSONL append-only output for responses
- `CheckpointManager` to track processed `(query_id, cell_id)` pairs

**Output**: JSONL files and filtered response lists for resuming

### 12. Streamlit UI (`rag2riches/ui/streamlit_app.py`)

**Purpose**: Provide a browser-based interface for comparative runs.

**Key Features**:
- CSV upload or existing LanceDB reuse
- Metadata column selection
- Chunk size and overlap controls
- Embedding model selection with progress feedback
- Cell field selection
- Retrieval and generation controls
- Live table updates as cells complete
- CSV or JSON export

## Data Flow

```
CSV/TXT/PDF
    ↓
Ingestion: Document[]
    ↓
Cleaning: Document[] (cleaned)
    ↓
Chunking: Chunk[] (metadata inherited)
    ↓
Export/Analysis: chunks.csv, chunks.json
    ↓
[Optional] Embedding: EmbeddingRecord[]
    ↓
[Optional] Vector Store: Persistent LanceDB
    ↓
[Optional] Retrieval: Chunk[] (filtered by cell)
    ↓
[Optional] Generation: ResponseRecord[]
    ↓
Export: responses.csv, responses.json
```

## Core Data Types

All implemented with Pydantic for type safety and validation:

### Document
```python
Document(
    document_id: str,           # Unique ID
    source_path: str,           # File source
    text: str,                  # Full text
    metadata: dict[str, Any],   # E.g., {"party": "D", "year": 2020}
    raw: Optional[str],         # Original text before cleaning
)
```

### Chunk
```python
Chunk(
    chunk_id: str,              # Unique ID
    document_id: str,           # Parent document
    text: str,                  # Chunk text
    metadata: dict[str, Any],   # Inherited from document
    chunk_index: int,           # Position in document
    start_char: Optional[int],  # Character offset in original
    end_char: Optional[int],
)
```

### Cell
```python
Cell(
    cell_id: str,                   # E.g., "party=D|year=2020"
    fields: dict[str, Any],         # Field values defining cell
    filter_expression: Optional[Any], # Vector store filter (backend-specific)
)
```

### ResponseRecord
```python
ResponseRecord(
    response_id: str,
    query_id: str,
    query_text: str,
    cell_id: str,                    # Which cell this response is for
    cell_filter: dict[str, Any],
    retrieved_chunk_ids: list[str],  # IDs of chunks used
    retrieved_context: str,          # Full text of retrieved chunks
    response_text: str,              # Generated response
    model_name: str,                 # Which LLM model
    embedding_model_name: str,       # Which embedding model
    timestamp: datetime,
    metadata: dict[str, Any],        # Error messages, retries, etc.
)
```

## Extension Points

### Custom Cleaners
```python
def my_clean_fn(text: str) -> str:
    # Custom logic
    return cleaned_text

cleaned_docs = clean_documents(docs, clean_fn=my_clean_fn)
```

### Custom Chunkers
```python
def sentence_chunker(text: str) -> list[str]:
    # Split by sentences
    return sentences

chunks = chunks_from_documents(docs, chunk_fn=sentence_chunker)
```

### Custom Embedders
```python
class MyEmbedder(Embedder):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Custom embedding logic
        pass
```

### Custom Vector Stores
```python
class MyVectorStore(VectorStore):
    def add_chunks(self, chunks: list[Chunk], embeddings: list[EmbeddingRecord]) -> None:
        # Custom storage logic
        pass
```

### Custom Generators
```python
class MyGenerator(Generator):
    def generate(self, query_text: str, chunks: list[Chunk]) -> str:
        # Custom generation logic
        pass
```

## Configuration

All major components use Pydantic configuration schemas:

```python
from rag2riches import PipelineConfig

config = PipelineConfig(
    ingestion=IngestionConfig(format="csv"),
    cleaning=CleaningConfig(normalize_whitespace=True),
    chunking=ChunkingConfig(chunk_size=750, chunk_overlap=100),
    embedding=EmbeddingConfig(model="openai/text-embedding-3-small"),
    generation=GenerationConfig(model="gpt-4o-mini"),
)
```

## Performance Considerations

### Chunking
- Character-based chunking is O(n) where n = document length
- Overlap is necessary to prevent information loss at boundaries

### Retrieval
- **Metadata pre-filtering** is critical for efficiency
- Without pre-filtering, similarity search over all chunks is slow
- With pre-filtering (cell-constrained), search is fast
- LanceDB's built-in filtering makes pre-filtered retrieval native

### Embeddings
- Batching (default batch_size=32) reduces API calls and cost
- Caching prevents recomputing embeddings for identical text
- Checkpointing allows resuming interrupted embedding jobs

### Generation
- Parallel cell processing is safe (non-overlapping cells)
- Batch API support (v0.3+) reduces latency for many cells

## Testing Strategy

- **Unit tests**: Each module tested independently with mocked data
- **Integration tests**: Data flows through complete pipeline
- **No live API calls**: Mocks used for LLM and embedding services
- **Fixtures**: Small CSV and TXT files in `tests/fixtures/`

## Logging

Every major stage logs:
- Documents ingested
- Chunks created
- Cells constructed
- Embeddings completed
- Retrieval queries
- Generation requests
- Errors and retries

Use `setup_logging()` to configure log level and output file.

## Security

- API keys stored in environment variables, never in code
- `.env` file excluded from version control
- No automatic data uploads
- User controls what gets sent to external services

## Future Extensions

1. **Async/Await**: Support for async embedding and generation
2. **Distributed Processing**: Multi-GPU or multi-machine embeddings
3. **Advanced Retrieval**: BM25 + vector hybrid search, cross-encoders for reranking
4. **Streaming**: Stream responses as they're generated
5. **Caching**: Cache retrieved contexts to avoid duplicate searches
6. **Versioning**: Track changes to chunks and metadata over time

