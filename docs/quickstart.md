# Getting Started with RAG2riches

This guide walks you through setting up and testing the current RAG2riches release.

## Prerequisites

- **Python**: 3.10 or higher
- **pip**: Latest version recommended
- **bash/PowerShell**: For running commands

## Installation
RAG2riches is not yet published on PyPI. Install it from the GitHub repository.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag2riches.git
```

### 2. Navigate to Project Directory

```bash
cd "path/to/rag2riches"
```

### 3. Create Virtual Environment (Recommended)

```bash
# Create a new virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install RAG2riches

```bash
# Install in development mode with core dependencies
pip install -e .
```

This installs:
- rag2riches package (editable, so code changes take effect immediately)
- Core dependencies: pydantic, pandas, pyarrow, tqdm, loguru, pypdf

### 5. Verify Installation

```bash
python -c "import rag2riches; print(f'RAG2riches {rag2riches.__version__} installed')"
```

Expected output:
```
RAG2riches 0.1.0a1 installed
```

## Running Tests

### Install Testing Dependencies

```bash
pip install -e ".[dev]"
```

This adds pytest, pytest-cov, ruff, and other development tools.

### Run All Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_ingestion.py::test_csv_basic_ingestion PASSED
tests/test_ingestion.py::test_csv_document_id_generation PASSED
...
============ 40+ passed in X.XXs ==============
```

### Run Specific Test Module

```bash
# Test ingestion
pytest tests/test_ingestion.py -v

# Test chunking
pytest tests/test_chunking.py -v

# Test export
pytest tests/test_export.py -v
```

### Run with Coverage Report

```bash
pytest tests/ --cov=rag2riches --cov-report=html
```

This creates an `htmlcov/index.html` file showing code coverage (should be near 100%).

### Run Tests with Output

```bash
# Show print statements and logs
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Run only failing tests (if any)
pytest tests/ --lf
```

## Running the Example

### Run Minimal Example

```bash
python examples/minimal_csv_example.py
```

Expected output:
```
============================================================
RAG2riches Minimal Example
============================================================

1. Ingesting from CSV...
   ✓ Ingested 6 documents
   ✓ Metadata fields: ['party', 'year', 'speaker']

2. Cleaning text...
   ✓ Cleaned 6 documents
   ✓ Sample cleaned text length: XXX chars

3. Chunking documents...
   ✓ Created 25-30 chunks
   ✓ Chunk size range: 123 - 250 chars

4. Verifying metadata...
   ✓ Sample chunk ID: <uuid>
   ✓ Sample chunk metadata: {'party': 'Democrat', 'year': 2020, 'speaker': 'Alice'}

5. Constructing metadata cells...
   ✓ Identified 4 unique (party, year) cells
   ✓ Parties: ['Democrat', 'Republican']
   ✓ Years: [2020, 2022]

6. Exporting chunks...
   ✓ Exported to output/chunks.csv
   ✓ Exported to output/chunks.json

============================================================
Summary
============================================================
Input documents:  6
Output chunks:    ~25-30
Unique cells:     4
Output files:
  - output/chunks.csv
  - output/chunks.json

Next steps:
  1. Open chunks.csv in Excel or a data analysis tool
  2. Review metadata distribution across cells
    3. Embed chunks and run comparative queries
============================================================
```

### Inspect Generated Output

```bash
# View CSV output
head output/chunks.csv

# View JSON output
head output/chunks.json

# Analyze in Python
python -c "
import pandas as pd
df = pd.read_csv('output/chunks.csv')
print(f'Total chunks: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Unique parties: {df[\"party\"].nunique()}')
print(f'Unique years: {df[\"year\"].nunique()}')
print(f'\\nFirst few rows:')
print(df.head())
"
```

## Using RAG2riches Programmatically

### Minimal Script

```python
from rag2riches import (
    ingest_documents,
    clean_documents,
    chunks_from_documents,
    construct_cells,
    export_chunks,
)

# Step 1: Ingest
docs = ingest_documents(
    "tests/fixtures/speeches_sample.csv",
    format="csv",
    text_column="speech_text",
    metadata_columns=["party", "year"]
)

# Step 2: Clean
docs = clean_documents(docs)

# Step 3: Chunk
chunks = chunks_from_documents(docs, chunk_size=250, chunk_overlap=50)

# Step 4: Construct cells
cells = construct_cells(chunks, cell_fields=["party", "year"])

# Step 5: Export
export_chunks(chunks, "my_chunks.csv", format="csv")

print(f"Processed {len(docs)} documents into {len(chunks)} chunks")
print(f"Created {len(cells)} comparison cells")
```

### Pipeline Comparative Run (Mock)

```python
from rag2riches import (
    InMemoryVectorStore,
    MockEmbedder,
    MockGenerator,
    QuerySpec,
    RAG2richesPipeline,
)

pipeline = RAG2richesPipeline.from_csv(
    path="tests/fixtures/speeches_sample.csv",
    text_column="speech_text",
    metadata_columns=["party", "year"],
)

pipeline.clean()
pipeline.chunk(chunk_size=250, chunk_overlap=50)
pipeline.embed(embedder=MockEmbedder(dim=8))
pipeline.index(vector_store=InMemoryVectorStore())

responses = pipeline.compare(
    queries=[
        QuerySpec(
            query_text="How does the corpus discuss climate?",
            cell_fields=["party", "year"],
            retrieval_k=2,
        )
    ],
    generator=MockGenerator(),
)

print(responses[0].response_text)
```

## Streamlit UI

RAG2riches also ships with a Streamlit app for interactive comparative retrieval.

Launch it with:

```bash
streamlit run rag2riches/ui/streamlit_app.py
```

The UI supports:

- CSV upload or an existing LanceDB store
- Text and metadata column selection for CSV input
- Chunk size and overlap controls, defaulting to 750 and 100
- Embedding model selection with `openai/text-embedding-3-small` as the default
- Retrieval method selection and passage count selection
- Cell field selection for comparison runs
- Generation model selection with `gpt-4o-mini` as the default
- Real-time cell-by-cell results and CSV/JSON export

For a guided walkthrough, see [examples/streamlit_demo_instructions.md](../examples/streamlit_demo_instructions.md).

### Using Configuration Objects

```python
from rag2riches import (
    IngestionConfig,
    CleaningConfig,
    ChunkingConfig,
    ingest_documents,
    clean_documents,
    chunks_from_documents,
)

# Define configurations
ingest_cfg = IngestionConfig(
    format="csv",
    text_column="speech_text",
    metadata_columns=["party", "year", "speaker"]
)

clean_cfg = CleaningConfig(
    normalize_whitespace=True,
    remove_extra_newlines=True
)

chunk_cfg = ChunkingConfig(
    chunk_size=500,
    chunk_overlap=75
)

# Use configurations
docs = ingest_documents("data.csv", **ingest_cfg.dict())
docs = clean_documents(docs)
chunks = chunks_from_documents(
    docs,
    chunk_size=chunk_cfg.chunk_size,
    chunk_overlap=chunk_cfg.chunk_overlap
)
```

### Custom Processing

```python
from rag2riches import (
    ingest_documents,
    clean_documents,
    chunks_from_documents,
    export_chunks,
)

# Custom cleaner (e.g., remove numbers)
def remove_numbers(text):
    import re
    return re.sub(r'\d+', '', text)

# Custom chunker (e.g., by sentences)
def sentence_chunker(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s + '.' if not s.endswith(('.', '!', '?')) else s for s in sentences]

# Apply custom functions
docs = ingest_documents("data.csv", format="csv", text_column="text")
docs = clean_documents(docs, clean_fn=remove_numbers)
chunks = chunks_from_documents(docs, chunk_fn=sentence_chunker)
export_chunks(chunks, "output.csv")
```

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'rag2riches'
```

**Solution**: Make sure you installed the package in editable mode:
```bash
pip install -e .
```

### Missing Dependencies

```
ImportError: No module named 'pandas'
```

**Solution**: Install dependencies:
```bash
pip install -e .
```

### Tests Fail

```
FAILED tests/test_ingestion.py::test_csv_basic_ingestion
```

**Solution**:
1. Make sure you're in the correct directory:
   ```bash
    cd "path/to/rag2riches"
   ```
2. Make sure dependencies are installed:
   ```bash
   pip install -e ".[dev]"
   ```
3. Check Python version (need 3.10+):
   ```bash
   python --version
   ```

### Permission Errors on Output

```
PermissionError: [Errno 13] Permission denied: 'output/chunks.csv'
```

**Solution**: Make sure the output directory is writable:
```bash
mkdir -p output
chmod 755 output  # On macOS/Linux
```

## Next Steps

1. ✅ **Verify Installation**: Run `python -c "import rag2riches; print('OK')"`
2. ✅ **Run Tests**: Run `pytest tests/ -v` (should see 35 PASSED)
3. ✅ **Run Example**: Run `python examples/minimal_csv_example.py`
4. ✅ **Inspect Output**: Check `output/chunks.csv` in Excel or Python
5. 📖 **Read Documentation**: Read [README.md](../README.md) for overview
6. 📖 **Read Architecture**: Read [docs/architecture.md](../docs/architecture.md) for deep dive
7. 💻 **Try Your Data**: Replace test CSV with your own data
8. 🔮 **Plan Next Steps**: When ready for advanced retrieval, see DEVELOPMENT_PLAN.md

## Environment Variables (Optional)

For embedding and LLM support, create a `.env` file:

```bash
cp .env.example .env
```

Then add your API keys. See `.env.example` for detailed instructions.

## Getting Help

### Documentation Files

- **[README.md](../README.md)** — Overview, installation, quick start
- **[DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md)** — Roadmap and architecture decisions
- **[docs/architecture.md](../docs/architecture.md)** — System design and extension points
- **[docs/credits.md](../docs/credits.md)** — Acknowledgments and citations
- **[IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)** — What's implemented and what's next

### Code Documentation

All modules have comprehensive docstrings:

```python
from rag2riches import ingest_documents
help(ingest_documents)  # View docstring
```

### Common Tasks

#### Load a CSV File
```python
from rag2riches import ingest_documents

docs = ingest_documents(
    "data.csv",
    format="csv",
    text_column="text",
    metadata_columns=["party", "year"]
)
```

#### Clean and Chunk
```python
from rag2riches import clean_documents, chunks_from_documents

docs = clean_documents(docs)
chunks = chunks_from_documents(docs, chunk_size=750, chunk_overlap=100)
```

#### Construct Cells and Analyze
```python
from rag2riches import construct_cells, chunks_for_cell, chunks_dataframe

cells = construct_cells(chunks, cell_fields=["party", "year"])
for cell in cells:
    cell_chunks = chunks_for_cell(chunks, cell)
    print(f"{cell.cell_id}: {len(cell_chunks)} chunks")

# Export to DataFrame
df = chunks_dataframe(chunks)
print(df.groupby("party").size())
```

#### Export Results
```python
from rag2riches import export_chunks

# Export as CSV
export_chunks(chunks, "chunks.csv", format="csv")

# Export as JSON
export_chunks(chunks, "chunks.json", format="json")
```

## Contributing Feedback

As you use RAG2riches, please consider:

1. **Trying with your own data** — Does it work? Are there issues?
2. **Reporting bugs** — Create an issue on GitHub
3. **Suggesting improvements** — What would make it easier to use?
4. **Sharing examples** — Show how you're using RAG2riches

Your feedback shapes the roadmap for upcoming releases!

---

**Happy coding!** 🚀

For questions or issues, see the [README.md](../README.md#support) for support options.

