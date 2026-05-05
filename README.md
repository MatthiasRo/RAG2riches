# RAG2riches

**Comparative Retrieval-Augmented Generation for Social Science Research**

RAG2riches is a Python library that makes it easy to compare LLM-generated responses across metadata-defined subsets of a text corpus. Unlike generic RAG systems, RAG2riches implements **metadata-filtered pre-retrieval**, ensuring responses are grounded only in the relevant subcorpus.

## What is Comparative RAG?

In traditional RAG, you retrieve globally and then filter. RAG2riches does the opposite:

1. Define comparison cells using metadata (e.g., `party-year`, `firm-year`, `outlet-month`)
2. Ask a single question
3. For each cell, retrieve text **only from that cell**
4. Generate a response grounded **only** in that cell's text
5. Export responses for downstream analysis

This approach is natural for social science researchers who want to compare how different groups, time periods, or sources discuss the same issue.

## Example: Party-Year Comparison

Imagine you have speeches from Democrats and Republicans across multiple years. You want to ask: *"How does this corpus discuss climate regulation?"*

With RAG2riches, you can ask this question once and get separate, grounded responses for each `(party, year)` combination:

| Party | Year | Response |
|-------|------|----------|
| Democratic | 2020 | "The corpus emphasizes urgent federal action on climate..." |
| Democratic | 2022 | "Responses focus on investment in clean energy infrastructure..." |
| Republican | 2020 | "The corpus frames regulation as economically costly..." |
| Republican | 2022 | "Discourse emphasizes market-based solutions and innovation..." |

Each response is grounded only in speeches from that specific party and year.

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda

### Alpha Installation (GitHub Only)

RAG2riches is not yet published on PyPI. Install it directly from the GitHub repository.

#### Option A: Install via Git URL

```bash
pip install git+https://github.com/MatthiasRo/RAG2riches.git
```

#### Option B: Clone and Install (Recommended for Development)

```bash
git clone https://github.com/MatthiasRo/RAG2riches.git
cd rag2riches
pip install -e .
```

To install optional features used in this repository:

```bash
pip install -e ".[llm,vector,ui]"
```

To install the full development stack:

```bash
pip install -e ".[dev,llm,vector,ui]"
```

## Quick Start

### 1. Ingest Documents

```python
from rag2riches import RAG2richesPipeline

# Load from CSV
pipeline = RAG2richesPipeline.from_csv(
    path="speeches.csv",
    text_column="speech_text",
    metadata_columns=["party", "year", "speaker"]
)
```

### 2. Clean and Chunk

```python
# Clean text (normalize whitespace, etc.)
pipeline.clean()

# Chunk into manageable pieces
pipeline.chunk(chunk_size=750, chunk_overlap=100)
```

### 3. Explore Chunks and Metadata

```python
# Export chunks with metadata for inspection
pipeline.export_chunks("output/chunks.csv")

# See unique values for metadata fields
print("Parties:", pipeline.get_metadata_values("party"))
print("Years:", pipeline.get_metadata_values("year"))
```

### Complete Minimal Example

```python
from rag2riches import RAG2richesPipeline

# Step 1: Ingest
pipeline = RAG2richesPipeline.from_csv(
    path="speeches.csv",
    text_column="speech_text",
    metadata_columns=["party", "year"]
)

# Step 2: Process
pipeline.clean()
pipeline.chunk(chunk_size=750, chunk_overlap=100)

# Step 3: Export chunks with metadata
pipeline.export_chunks("output/chunks.csv", format="csv")

# Step 4: View results
import pandas as pd
chunks_df = pd.read_csv("output/chunks.csv")
print(f"Total chunks: {len(chunks_df)}")
print(f"Unique parties: {chunks_df['party'].nunique()}")
print(f"Unique years: {chunks_df['year'].nunique()}")
```

## Embedding and Comparison (Current)

RAG2riches includes a testable comparative pipeline with both mock and LiteLLM-backed
components:

1. **Mock or LiteLLM embeddings** via `MockEmbedder` or `LiteLLMEmbedder`
2. **In-memory or LanceDB vector store** via `InMemoryVectorStore` or `LanceDBVectorStore`
3. **Retriever** with metadata pre-filtering
4. **Mock or LiteLLM generation** via `MockGenerator` or `LiteLLMGenerator`
5. **ComparativeRunner** to run queries across cells
6. **Checkpointing** to append results to JSONL and resume

LiteLLM and LanceDB support are available as optional extras:
`pip install -e ".[llm,vector]"`.

You can run an end-to-end mock comparative workflow locally:

```python
from rag2riches import (
    MockEmbedder,
    InMemoryVectorStore,
    Retriever,
    MockGenerator,
    ComparativeRunner,
    construct_cells,
    chunks_from_documents,
    clean_documents,
    ingest_documents,
)

docs = ingest_documents(
    "tests/fixtures/speeches_sample.csv",
    format="csv",
    text_column="speech_text",
    metadata_columns=["party", "year"],
)
docs = clean_documents(docs)
chunks = chunks_from_documents(docs, chunk_size=250, chunk_overlap=50)

embedder = MockEmbedder(dim=8)
store = InMemoryVectorStore()
embeddings = embedder.embed_records([c.chunk_id for c in chunks], [c.text for c in chunks])
store.add_chunks(chunks, embeddings)

retriever = Retriever(embedder=embedder, vector_store=store, default_k=2)
generator = MockGenerator()
runner = ComparativeRunner(retriever=retriever, generator=generator)

responses = runner.run_query_across_cells(
    query_text="How does the corpus discuss climate?",
    chunks=chunks,
    cell_fields=["party", "year"],
    k=2,
)

print(responses[0].response_text)
```

## Streamlit UI

RAG2riches includes a Streamlit app for interactive comparative querying.

Launch it with:

```bash
streamlit run rag2riches/ui/streamlit_app.py
```

Note: The Streamlit app will automatically load a `.env` file from the project root if present. To test OpenAI embeddings/generation, copy `.env.example` to `.env` and add your `OPENAI_API_KEY=sk-...` before launching the app.

The UI lets you:

- Load a CSV file or connect to an existing LanceDB database
- Choose the text column and metadata columns for CSV input
- Set chunk size and overlap, with defaults of 750 characters and 100 overlap
- Choose an embedding model, with `openai/text-embedding-3-small` as the default
- Optionally reuse an existing vector database instead of embedding from scratch
- Select retrieval method and number of passages to retrieve
- Choose comparison cell fields such as `party`, `year`, or `firm`, `month`
- Choose the generation model, with `gpt-4o-mini` as the default
- Run a query across cells and watch cell-level results appear as the run proceeds in a table
- Export the full responses table to CSV or JSON

For a guided walkthrough, see [examples/streamlit_demo_instructions.md](examples/streamlit_demo_instructions.md).

## Supported File Formats

| Format | Usage | Example |
|--------|-------|---------|
| CSV | Tabular data with text column | `pipeline.from_csv(path, text_column="text", metadata_columns=[...])` |
| TXT | Directory of text files | `pipeline.from_txt_directory(path)` |
| PDF | Single or multiple PDF files | `pipeline.from_pdf(path)` |

## Configuration

### Chunking Options

```python
pipeline.chunk(
    chunk_size=750,      # Characters per chunk
    chunk_overlap=100,   # Overlap between chunks
    method="character"   # "character" or "sentence" (v0.2+)
)
```

### Metadata-Defined Cells

```python
# Define which fields create comparison cells
cell_fields = ["party", "year"]  # Results in cells like "party=D|year=2020"
# or
cell_fields = ["firm", "year"]   # For firm-level analysis
# or
cell_fields = ["outlet", "month", "topic"]  # Multi-dimensional comparisons
```

## Environment Variables

For LLM and embedding provider credentials:

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

Never commit `.env` files. See `.env.example` for templates.

## Data Privacy

- RAG2riches does **not** upload your documents by default
- Local processing: Chunking, metadata handling, and local embeddings are all done client-side
- Optional remote services: Only if you explicitly configure external LLM/embedding providers
- Store credentials in environment variables, never in code

## Design Principles

1. **Modular**: Use individual components or the full pipeline
2. **Accessible**: Simple APIs for non-experts, powerful APIs for advanced users
3. **Built on Open Source**: Uses pandas, pydantic, LanceDB, LiteLLM, and other proven tools
4. **Efficient**: Batching, caching, and resumable runs reduce redundant work
5. **Observable**: Logging and progress bars at every stage

## Architecture

```
Data Input (CSV, TXT, PDF)
    ↓
[Ingestion] → Document objects with metadata
    ↓
[Cleaning] → Normalize text
    ↓
[Chunking] → Text chunks with inherited metadata
    ↓
[Export/Analysis] ← Chunks ready for embedding
    ↓
[Embedding] → Vector embeddings (optional)
    ↓
[Vector Store] → LanceDB with metadata indices (optional)
    ↓
[Metadata-Filtered Retrieval] ← Retrieve only from relevant cell
    ↓
[Generation] → LLM generates response (optional)
    ↓
[Export] → CSV/JSON for analysis
```

## Roadmap

### ✅ v0.1 (Current)
- CSV, TXT ingestion
- Text cleaning and chunking
- Metadata propagation and cell construction
- Export to CSV/JSON
- LiteLLM and mock embeddings
- LanceDB and in-memory vector stores
- Metadata-pre-filtered retrieval
- ComparativeRunner for cross-cell queries
- Checkpointing and resume logic
- Comprehensive tests
- Streamlit UI for interactive comparative runs

### 📋 v0.2 (Next)
- PDF ingestion with page-level metadata
- Sentence-aware chunking
- Async batch API support
- Advanced retrieval strategies (BM25, hybrid search, reranking)

### 🔮 v0.3+
- PDF ingestion with page-level metadata
- Sentence-aware chunking
- Advanced retrieval (BM25, hybrid search, reranking)
- Streamlit exploratory UI
- Async batch API support
- Multi-language support

## Documentation

- **[Getting Started](docs/quickstart.md)**: Detailed walkthrough with examples
- **[Architecture](docs/architecture.md)**: System design and extension points
- **[API Reference](docs/api_reference.md)** (auto-generated from docstrings)
- **[Credits](docs/credits.md)**: Acknowledgments of open-source dependencies
- **[Examples](examples/)**: Runnable Python scripts for common workflows
- **[Streamlit demo instructions](examples/streamlit_demo_instructions.md)**: How to launch the interactive UI

## Testing

Run tests locally:

```bash
pip install -e ".[dev]"
pytest
```

Tests use mocked data and do not require API keys or external services.

## Contributing

I welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Run `pytest` and `ruff` to verify
5. Submit a pull request

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use RAG2riches in research, please cite:

Roesti, Matthias, From RAGs to (feature) Riches - An Efficient Pipeline for Exploratory Text Analysis (June 29, 2025). Available at SSRN: https://ssrn.com/abstract=5331899 or http://dx.doi.org/10.2139/ssrn.5331899

```bibtex
@software{roesti2024rag2riches,
  title = {RAG2riches: Comparative Retrieval-Augmented Generation for Social Science},
  author = {Roesti, Matthias},
  year = {2026},
    url = {https://github.com/MatthiasRo/RAG2riches}
}

@article{roesti2025rags,
    title = {From RAGs to (feature) Riches - An Efficient Pipeline for Exploratory Text Analysis},
    author = {Roesti, Matthias},
    year = {2025},
    month = jun,
    url = {https://ssrn.com/abstract=5331899},
    doi = {10.2139/ssrn.5331899},
    note = {Available at SSRN}
}
```

## Important Disclaimer

**Generated responses should be validated and are not substitutes for substantive interpretation.**

RAG2riches helps researchers explore large text corpora efficiently, but like all LLM-based tools:
- Responses may hallucinate or misinterpret context
- Embeddings capture statistical similarity, not semantic truth
- Filtering to metadata cells does not guarantee relevance
- Always review retrieved context and validate findings

Use RAG2riches as part of a rigorous research workflow, not as a replacement for careful reading and analysis.

## Support

- **Issues**: [GitHub Issues](https://github.com/MatthiasRo/RAG2riches/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MatthiasRo/RAG2riches/discussions)
- **Email**: matthias_roesti@brown.edu

## Acknowledgments

RAG2riches builds on excellent open-source projects:
- **pandas** and **pyarrow** for data handling
- **pydantic** for configuration validation
- **LiteLLM** for unified LLM/embedding interfaces
- **LanceDB** for vector storage with metadata filtering
- **tqdm** and **loguru** for progress and logging
- **pytest** for testing

See [docs/credits.md](docs/credits.md) for full attribution.
