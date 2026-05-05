# RAG2riches Development Plan

## Vision

RAG2riches is a comparative retrieval-augmented generation library designed for social science researchers. It enables researchers to ask the same question across metadata-defined subsets of a text corpus (e.g., party-year, firm-year, outlet-month cells), with retrieval restricted to the relevant subcorpus before response generation.

## Core Distinctive Feature

Unlike generic RAG systems, RAG2riches implements **metadata-filtered pre-retrieval** rather than global retrieval followed by post-filtering. This ensures:
- Responses are grounded only in the relevant subcorpus
- Efficient batch comparisons across many cells
- Natural fit with social science research workflows

## Development Philosophy

1. **Start simple, build modular**: Implement a minimal working version first, leaving room for extensions
2. **Prioritize usability for non-experts**: Default workflows should be concise and robust
3. **Build on proven tools**: Use LiteLLM, LanceDB, pydantic, and other open-source packages rather than reimplementing
4. **Logging and checkpointing**: Every major stage should be resumable and observable
5. **Type-safe, documented code**: Full type hints and docstrings throughout

## Architecture Overview

```
Data Input
    ↓
[Ingestion] → CSV, TXT, PDF files
    ↓
[Cleaning] → Normalize whitespace, remove boilerplate
    ↓
[Chunking] → Character-length chunking with overlap
    ↓
[Metadata Attachment] → Document & chunk-level metadata
    ↓
[Embedding] → LiteLLM embeddings with caching
    ↓
[Vector Storage] → LanceDB with metadata indexing
    ↓
[Metadata-Filtered Retrieval] ← Per-cell filtering BEFORE retrieval
    ↓
[Response Generation] → LiteLLM-based generation
    ↓
[Comparative Execution] → Repeat across all cells
    ↓
[Export] → CSV, JSON output for analysis
```

## Implementation Roadmap

### Phase 1: Foundation (Steps 1-5)
**Goal**: Ingest data, transform it, manage metadata, and export results

- Step 1: Package scaffold, pyproject.toml, README, .gitignore, basic docs
- Step 2: Core data types (Document, Chunk, EmbeddingRecord, QuerySpec, ResponseRecord)
- Step 3: CSV and TXT ingestion
- Step 4: Cleaning and chunking
- Step 5: Metadata utilities and cell construction
- **Checkpoint**: Validate data flows through ingestion → cleaning → chunking → export

### Phase 2: Embedding & Retrieval (Steps 6-8)
**Goal**: Embed and retrieve text with metadata filtering

- Step 6: Abstract Embedder interface with mock implementation for testing
- Step 7: LanceDB vector store with metadata-pre-filtering
- Step 8: Retriever class for metadata-filtered similarity search
- **Checkpoint**: Retrieve chunks only from specific cells, not globally

### Phase 3: Generation & Comparison (Steps 9-10)
**Goal**: Generate responses and compare across cells

- Step 9: Generator interface with mock and LiteLLM implementations
- Step 10: ComparativeRunner orchestrating cross-cell queries
- **Checkpoint**: Run a single query across multiple cells

### Phase 4: Robustness & Testing (Steps 11-14)
**Goal**: Make the system resilient and testable

- Step 11: Export to CSV and JSON
- Step 12: Checkpointing and resume logic
- Step 13: Streamlit exploratory UI
- Step 14: Unit tests for all major components
- **Checkpoint**: Interrupted runs resume without reprocessing

### Phase 5: Documentation & Release (Step 15)
**Goal**: Make it accessible to researchers

- Step 15: Complete README, architecture docs, credits, examples
- **Deliverable**: Working package with clear quickstart

## First Session Deliverables

This session will produce:

1. ✅ **Development plan** (this document)
2. ✅ **Package scaffold** with correct folder structure
3. ✅ **pyproject.toml** with dependencies
4. ✅ **.gitignore** and **LICENSE** stub
5. ✅ **Core data types** (pydantic models)
6. ✅ **Ingestion module** (CSV and TXT)
7. ✅ **Cleaning and chunking** with defaults
8. ✅ **Metadata propagation** and cell construction
9. ✅ **Export utilities** (CSV, JSON)
10. ✅ **Unit tests** (mocked, no API calls)
11. ✅ **README** with minimal working example

After this session:
- A researcher should be able to ingest a CSV file, chunk it, attach metadata, construct cells, and export metadata-enriched chunks in ~10 lines
- All major data flows should be tested
- Future steps (embeddings, retrieval, generation) should be straightforward

## Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Config | Pydantic | Type-safe, validates at creation, integrates with Streamlit |
| Data Storage | Pandas DataFrames (initially) + eventual Parquet | Familiar to social scientists, efficient storage |
| Vector Store | LanceDB (default) | Native metadata filtering, persistent storage, Apache license |
| Embedder Abstraction | Interface + LiteLLM implementation | Supports OpenAI, Gemini, Claude, Ollama |
| Generator Abstraction | Interface + LiteLLM implementation | Same provider diversity as embeddings |
| Progress Display | tqdm/rich | Familiar to Python users, minimal overhead |
| Logging | Python logging + custom module | Standard library, easy to redirect to files or Streamlit |
| Testing | pytest with mocks | Standard, mocks avoid API costs |
| Async Support | Deferred | Design interfaces so async can be added later (LiteLLM batch) |

## Data Model Overview

```python
Document(
    document_id: str,
    source_path: str,
    text: str,
    metadata: dict[str, Any],
)

Chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    metadata: dict[str, Any],  # merged from document + chunk-level
    chunk_index: int,
    start_char: Optional[int],
    end_char: Optional[int],
)

EmbeddingRecord(
    chunk_id: str,
    vector: list[float],
    embedding_model: str,
)

Cell(
    cell_id: str,  # e.g., "party=D|year=2020"
    fields: dict[str, Any],  # e.g., {"party": "D", "year": 2020}
    filter_expression: Any,  # For LanceDB or other vector store
)

ResponseRecord(
    response_id: str,
    query_id: str,
    query_text: str,
    cell_id: str,
    retrieved_chunk_ids: list[str],
    retrieved_context: str,
    response_text: str,
    model_name: str,
    embedding_model_name: str,
    timestamp: datetime,
    metadata: dict[str, Any],
)
```

## Dependencies (Minimal First Version)

**Core**:
- pydantic >= 2.0
- pandas >= 1.5
- pyarrow >= 10.0

**Data**:
- pypdf (PDF text extraction)

**Progress & Logging**:
- tqdm
- loguru

**Testing**:
- pytest
- pytest-cov

**Future** (not in v0.1):
- litellm (added in Phase 2)
- lancedb (added in Phase 2)
- streamlit (added in Phase 4)

## API Examples (Target)

### Minimal usage (first session achievable):
```python
from rag2riches import RAG2richesPipeline

pipeline = RAG2richesPipeline.from_csv(
    path="speeches.csv",
    text_column="speech_text",
    metadata_columns=["party", "year"]
)

pipeline.clean()
pipeline.chunk(chunk_size=750, chunk_overlap=100)

# Export enriched chunks with metadata
pipeline.export_chunks("output/chunks.csv")
```

### Full pipeline (Phases 2-3):
```python
pipeline.embed(model="openai/text-embedding-3-small")
pipeline.index(vector_store="lancedb", path="./rag2riches_store")

responses = pipeline.compare(
    queries=["How does climate regulation appear?"],
    cell_fields=["party", "year"],
    generation_model="gpt-4o-mini",
)

responses.to_csv("output/responses.csv")
```

## Testing Strategy

- **Unit tests**: Each module tested independently with mocked data
- **Integration tests**: Data flows through full pipeline
- **No live API calls** in tests (use mocks for LLM and embedding services)
- **Fixture-based test data**: Small CSV and text files in `tests/fixtures/`

## Documentation Structure

- **README.md**: What, why, quickstart, setup, minimal example
- **docs/architecture.md**: System design, module interactions, extensibility points
- **docs/quickstart.md**: Detailed walkthrough with examples
- **docs/api_reference.md**: Auto-generated from docstrings
- **docs/credits.md**: Acknowledgments of open-source dependencies
- **examples/**: Runnable Python scripts demonstrating common workflows
- **CHANGELOG.md**: Version history (placeholder in v0.1)

## Success Criteria for First Session

- [x] All code has type hints
- [x] All public functions have docstrings
- [x] Ingestion works for CSV and TXT
- [x] Metadata propagates correctly through pipeline
- [x] Cells are constructed and usable for filtering
- [x] Export produces valid CSV and JSON
- [x] Tests pass without external API calls
- [x] README includes working minimal example
- [x] Package can be installed in editable mode (`pip install -e .`)

## Known Gaps for Future Work

1. **Sentence-aware chunking**: Initial chunking is character-based; sentence tokenization can be added later
2. **Async batch API support**: LiteLLM batching is deferred; current implementation is synchronous
3. **Advanced retrieval**: Only similarity search in v0.1; BM25, reranking, hybrid search deferred
4. **Provider-specific optimizations**: OpenAI batch API, Anthropic caching, etc. are future work
5. **PDF page-level metadata**: Current PDF ingestion extracts full text; page tracking is deferred
6. **Streamlit caching**: Streamlit app in Phase 4 will add caching decorators
7. **Distributed embeddings**: Multi-GPU or multi-node embeddings are future work

## Repository Structure

```
rag2riches/                          # Main package
  __init__.py
  config.py                      # Pydantic config schemas
  types.py                       # Core data types (Chunk, Document, etc.)
  ingestion.py                   # CSV, TXT, PDF loading
  cleaning.py                    # Text cleaning
  chunking.py                    # Text chunking
  metadata.py                    # Metadata utilities, cell construction
  embeddings.py                  # Embedder interface (mock in v0.1)
  vectorstores/
    __init__.py
    base.py                      # Abstract VectorStore
    lancedb_store.py             # LanceDB implementation (v0.2)
  retrieval.py                   # Retriever interface (v0.2)
  prompts.py                     # Prompt templates
  generation.py                  # Generator interface (v0.2)
  compare.py                     # ComparativeRunner (v0.2)
  export.py                      # Export to CSV, JSON
  checkpointing.py               # Caching, resumable runs (v0.2)
  logging_utils.py               # Logging setup
  ui/
    streamlit_app.py             # Streamlit explorer (v0.4)

tests/
  __init__.py
  fixtures/                      # Test data (CSV, TXT files)
    speeches_sample.csv
    sample_documents.txt
  test_ingestion.py
  test_chunking.py
  test_metadata.py
  test_export.py
  test_cleaning.py

docs/
  architecture.md                # System design
  quickstart.md                  # Detailed walkthrough
  credits.md                     # Acknowledgments

examples/
  minimal_csv_example.py         # Simplest usage
  party_year_comparison_example.py # Social science example (v0.2)

.gitignore                       # Python + RAG-specific ignores
.env.example                     # Template for environment variables
LICENSE                          # Apache 2.0
README.md                        # Main entry point
pyproject.toml                   # Build config, dependencies
CHANGELOG.md                     # Version history
```

## Next Steps After First Session

1. Test the package locally with a real CSV file from the RAG project
2. Validate that metadata flows correctly through all stages
3. Gather feedback from intended users (social science researchers)
4. Plan Phase 2 with embedding and retrieval
5. Consider if any abstractions should be adjusted based on early use

