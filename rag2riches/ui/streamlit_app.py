"""
RAG2riches Streamlit UI.

Run with:
    streamlit run rag2riches/ui/streamlit_app.py
"""

from __future__ import annotations

import html
import io
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag2riches.chunking import chunks_from_documents
from rag2riches.cleaning import clean_documents
from rag2riches.embeddings import LiteLLMEmbedder
from rag2riches.export import export_responses
from rag2riches.generation import LiteLLMGenerator
from rag2riches.ingestion import CSVIngester
from rag2riches.metadata import construct_cells
from rag2riches.retrieval import Retriever
from rag2riches.types import Chunk, EmbeddingRecord, ResponseRecord
from rag2riches.vectorstores import LanceDBVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CUSTOM_EMBEDDING_MODEL = "__custom__"
EMBEDDING_MODEL_PRESETS: list[tuple[str, list[str]]] = [
    (
        "OpenAI",
        [
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
            "openai/text-embedding-ada-002",
        ],
    ),
    (
        "Google",
        [
            "google/text-embedding-004",
        ],
    ),
    (
        "Anthropic",
        [
            "anthropic/claude-3-haiku-20240307",
        ],
    ),
    (
        "Voyage AI",
        [
            "voyage/voyage-3",
            "voyage/voyage-3-lite",
            "voyage/voyage-2",
        ],
    ),
    (
        "Open-weight",
        [
            "bge-small-en-v1.5",
            "bge-base-en-v1.5",
            "bge-large-en-v1.5",
            "e5-small-v2",
            "e5-base-v2",
            "e5-large-v2",
            "gte-base",
            "gte-large",
            "nomic-embed-text-v1.5",
            "jina-embeddings-v2-base-en",
            "snowflake-arctic-embed-m",
        ],
    ),
]
FILTER_OPERATORS: list[tuple[str, str]] = [
    ("Equals", "="),
    ("Not equals", "!="),
    ("Greater than", ">"),
    ("Greater or equal", ">="),
    ("Less than", "<"),
    ("Less or equal", "<="),
    ("Is any of (comma-separated)", "in"),
    ("Is none of (comma-separated)", "not in"),
    ("Is empty (null)", "is null"),
    ("Is not empty", "is not null"),
]
FILTER_NULL_OPERATORS = {"is null", "is not null"}
MAX_FILTER_CONDITIONS = 8


def main() -> None:
    st.set_page_config(
        page_title="RAG2riches Studio",
        page_icon="🔎",
        layout="wide",
    )
    _apply_style()

    st.markdown(
        """
        <div class="rag2riches-hero">
            <div>
                <h1>RAG2riches Studio</h1>
                <p>Comparative retrieval-augmented generation across metadata-defined cells.</p>
            </div>
            <div class="rag2riches-pill">Beta UI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Load a corpus, define metadata cells, and compare LLM responses across groups. "
        "Embedding and indexing can be built on the fly or loaded from an existing LanceDB store."
    )

    subset_enabled = False
    subset_conditions: list[dict[str, Any]] = []
    subset_invalid_conditions = 0

    with st.sidebar:
        st.header("Data Source")
        data_mode = st.radio(
            "Choose input",
            ["CSV file", "Existing LanceDB"],
            horizontal=True,
        )

        csv_path = ""
        uploaded_file = None
        csv_df: pd.DataFrame | None = None
        text_column = None
        metadata_columns: list[str] = []

        if data_mode == "CSV file":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            csv_path = st.text_input(
                "CSV path (optional)",
                placeholder="C:/data/speeches.csv",
            )
            csv_df = _load_csv_preview(uploaded_file, csv_path)
            if csv_df is not None:
                text_column = st.selectbox("Text column", csv_df.columns)
                metadata_columns = st.multiselect(
                    "Metadata columns",
                    [col for col in csv_df.columns if col != text_column],
                )

        st.header("Chunking")
        chunk_size = st.number_input("Chunk size (chars)", min_value=100, max_value=5000, value=750)
        chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=100)

        st.header("Embedding + Index")
        if data_mode == "CSV file":
            embed_model = _select_embedding_model(
                default_model="openai/text-embedding-3-small",
                key_suffix="csv",
                help_text=(
                    "Presets are LiteLLM model identifiers. Some open-weight models require provider-specific "
                    "prefixes (ollama/, huggingface/, together/). Choose Custom to enter a model name."
                ),
            )
            embed_batch_size = st.number_input(
                "Embedding batch size",
                min_value=1,
                max_value=512,
                value=32,
            )
            cache_path = st.text_input(
                "Embedding cache path",
                value=".cache/embeddings/streamlit_embeddings.jsonl",
            )
            lancedb_path = st.text_input(
                "LanceDB path",
                value="./rag2riches_store",
            )
            table_name = st.text_input(
                "LanceDB table name",
                value="chunks",
                help="This is the table name inside the LanceDB database path.",
            )
        else:
            lancedb_path = st.text_input(
                "LanceDB path",
                value="./rag2riches_store",
            )
            table_name = st.text_input(
                "LanceDB table name",
                value="chunks",
                help="This is the table name inside the LanceDB database path.",
            )
            embed_model = _select_embedding_model(
                default_model="openai/text-embedding-3-small",
                key_suffix="db",
                help_text=(
                    "Use the same embedding model used to build the LanceDB table. Presets are LiteLLM "
                    "model identifiers; choose Custom to enter a provider-specific name."
                ),
            )
            embed_batch_size = 1
            cache_path = ".cache/embeddings/streamlit_embeddings.jsonl"

        st.header("Retrieval + Generation")
        st.selectbox("Retrieval method", ["similarity"], index=0)
        retrieval_k = st.number_input("Passages to retrieve", min_value=1, max_value=50, value=5)
        generation_model = st.text_input("LLM model", value="gpt-4o-mini")

        st.header("Cells")
        available_fields: list[str] = []
        if data_mode == "CSV file":
            available_fields = list(metadata_columns)
        else:
            if st.button("Load metadata fields"):
                st.session_state["db_fields"] = _load_metadata_fields_from_lancedb(
                    lancedb_path,
                    table_name,
                )
            available_fields = st.session_state.get("db_fields", [])

        cell_fields = st.multiselect(
            "Cell fields",
            options=available_fields,
            default=available_fields[:2] if len(available_fields) >= 2 else available_fields,
        )

        st.header("Subset filter")
        subset_enabled = st.checkbox(
            "Limit corpus to matching metadata",
            value=False,
            help=(
                "All conditions are combined with AND. Use 'Is any of' for OR values. "
                "Values auto-parse (2016 -> int, 3.5 -> float, true/false -> bool); "
                "wrap in quotes to force text."
            ),
        )
        if subset_enabled:
            if not available_fields:
                st.info("Select metadata columns or load metadata fields to build filters.")
            else:
                condition_count = st.number_input(
                    "Number of conditions",
                    min_value=1,
                    max_value=MAX_FILTER_CONDITIONS,
                    value=1,
                    step=1,
                    help="Use fewer conditions for faster filtering.",
                )
                operator_labels = [label for label, _ in FILTER_OPERATORS]
                operator_map = {label: op for label, op in FILTER_OPERATORS}
                for idx in range(int(condition_count)):
                    columns = st.columns([2, 2, 3])
                    field = columns[0].selectbox(
                        "Field",
                        options=available_fields,
                        key=f"subset_field_{idx}",
                        help="Metadata field to filter on.",
                    )
                    operator_label = columns[1].selectbox(
                        "Operator",
                        options=operator_labels,
                        key=f"subset_operator_{idx}",
                        help="Choose a comparison; 'Is any of' maps to an efficient IN filter.",
                    )
                    operator = operator_map[operator_label]
                    value_placeholder = _filter_value_placeholder(operator)
                    value_disabled = operator in FILTER_NULL_OPERATORS
                    value_input = columns[2].text_input(
                        "Value",
                        key=f"subset_value_{idx}",
                        placeholder=value_placeholder,
                        disabled=value_disabled,
                        help=(
                            "Comma-separate values for 'Is any of'. "
                            "Quote text if it looks numeric (e.g., '00123')."
                        ),
                    )
                    clause = _build_filter_clause(field, operator, value_input)
                    if clause:
                        subset_conditions.append(clause)
                    elif not value_disabled:
                        subset_invalid_conditions += 1

                subset_summary = _format_filter_summary(subset_conditions)
                if subset_summary:
                    st.caption(f"Subset filter: {subset_summary}")

                with st.expander("Filter tips", expanded=False):
                    st.markdown(
                        "- Example: year >= 2016\n"
                        "- Example: company is any of Coca Cola, Pepsi\n"
                        "- Example: region is not empty\n"
                        "- Wrap text with quotes to force strings: '00123'"
                    )

        st.header("Output")
        output_format = st.selectbox("Output format", ["json", "csv"], index=0)
        output_path = st.text_input(
            "Output path",
            value=f"output/streamlit_responses.{output_format}",
        )

    st.subheader("Corpus Preview")
    if data_mode == "CSV file":
        if csv_df is None:
            st.info("Upload a CSV file or provide a path to preview the data.")
        else:
            st.dataframe(csv_df.head(12), use_container_width=True)
    else:
        st.info("Using an existing LanceDB store. Metadata will be read from the table.")

    st.subheader("Query")
    query_text = st.text_area(
        "Ask a comparative question",
        placeholder="How does the corpus discuss climate regulation?",
    )
    additional_instructions = st.text_area(
        "Optional generation instructions",
        placeholder="Answer in 2-3 sentences and avoid outside knowledge.",
    )

    run_button = st.button("Run Comparative Query", type="primary")

    if run_button:
        if not query_text.strip():
            st.error("Please enter a query before running.")
            return

        if subset_enabled:
            if not available_fields:
                st.error("Load metadata fields to configure subset filters.")
                return
            if subset_invalid_conditions:
                st.error("Fill in all subset filter values or reduce the number of conditions.")
                return
            if not subset_conditions:
                st.error("Add at least one subset filter condition or disable filtering.")
                return

        subset_filter_spec = _build_filter_spec(subset_conditions)
        subset_filter_expr = _build_lancedb_filter_expression(subset_conditions)

        if data_mode == "CSV file":
            if csv_df is None:
                st.error("Please provide a CSV file or path.")
                return
            if not text_column:
                st.error("Please select a text column.")
                return
            if not metadata_columns:
                st.error("Please select at least one metadata column.")
                return

            csv_source = _materialize_csv(uploaded_file, csv_path)
            ingester = CSVIngester(
                text_column=text_column,
                metadata_columns=metadata_columns,
            )
            documents = ingester.ingest(csv_source)
            documents = clean_documents(documents)
            chunks = chunks_from_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if subset_conditions:
                filtered_chunks = _filter_chunks_by_conditions(chunks, subset_conditions)
                st.write(
                    f"Subset filter kept {len(filtered_chunks)} of {len(chunks)} chunks."
                )
                chunks = filtered_chunks
                if not chunks:
                    st.error("Subset filter removed all chunks.")
                    return

            st.write(f"Created {len(chunks)} chunks from {len(documents)} documents.")

            embedder = LiteLLMEmbedder(
                model=embed_model,
                batch_size=embed_batch_size,
                cache_path=cache_path,
            )
            embeddings = _embed_chunks_with_progress(
                embedder,
                chunks,
                batch_size=embed_batch_size,
            )

            store = LanceDBVectorStore(path=lancedb_path, table_name=table_name)
            store.create_or_connect(path=lancedb_path, table_name=table_name)
            store.add_chunks(chunks, embeddings)
        else:
            chunks = _load_chunks_from_lancedb(
                lancedb_path,
                table_name,
                filter_expression=subset_filter_expr,
            )
            if not chunks:
                st.error("No chunks found in the specified LanceDB table.")
                return

            embedder = LiteLLMEmbedder(
                model=embed_model,
                batch_size=embed_batch_size,
                cache_path=cache_path,
            )
            store = LanceDBVectorStore(path=lancedb_path, table_name=table_name)
            store.create_or_connect(path=lancedb_path, table_name=table_name)

        if not cell_fields:
            st.error("Please select at least one cell field.")
            return

        retriever = Retriever(embedder=embedder, vector_store=store, default_k=retrieval_k)
        generator = LiteLLMGenerator(model=generation_model)

        cells = construct_cells(chunks, cell_fields)
        st.write(f"Identified {len(cells)} unique cells.")

        responses = _run_comparative_with_progress(
            retriever=retriever,
            generator=generator,
            cells=cells,
            query_text=query_text,
            additional_instructions=additional_instructions,
            retrieval_k=retrieval_k,
            base_filter_spec=subset_filter_spec,
        )

        export_responses(responses, output_path, format=output_format)
        st.success(f"Saved {len(responses)} responses to {output_path}")
        st.subheader("Responses")
        _render_response_cards(responses)


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }
        code, pre, textarea {
            font-family: 'IBM Plex Mono', monospace;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 10% 20%, rgba(246, 244, 255, 0.6), rgba(255, 255, 255, 0.9)),
                        linear-gradient(120deg, #f8fafc 0%, #f0f4ff 50%, #fef6f0 100%);
        }

        .rag2riches-hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem 1.8rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            color: #f9fafb;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.25);
            margin-bottom: 1.2rem;
        }

        .rag2riches-hero h1 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: 0.5px;
        }

        .rag2riches-hero p {
            margin: 0.25rem 0 0 0;
            color: #d1d5db;
        }

        .rag2riches-pill {
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            background: #fbbf24;
            color: #111827;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08rem;
        }

        div.stButton > button {
            background: linear-gradient(90deg, #111827, #1f2937);
            color: #f9fafb;
            border-radius: 12px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }

        div.stButton > button:hover {
            background: linear-gradient(90deg, #1f2937, #111827);
            transform: translateY(-1px);
        }

        .stProgress > div > div > div {
            background-image: linear-gradient(90deg, #f59e0b, #fbbf24);
        }

        .rag2riches-response-card {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin: 1rem 0 0.75rem 0;
            box-shadow: 0 16px 35px rgba(15, 23, 42, 0.08);
        }

        .rag2riches-response-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.6rem;
        }

        .rag2riches-response-title {
            font-weight: 700;
            font-size: 1rem;
            color: #111827;
        }

        .rag2riches-response-meta {
            color: #475569;
            font-size: 0.82rem;
            margin-top: 0.15rem;
        }

        .rag2riches-response-pill {
            background: #111827;
            color: #f9fafb;
            padding: 0.3rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .rag2riches-response-text {
            color: #1f2937;
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .rag2riches-chunk-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.7rem;
        }

        .rag2riches-chunk-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.4rem;
        }

        .rag2riches-chunk-label {
            font-weight: 600;
            font-size: 0.8rem;
            color: #0f172a;
        }

        .rag2riches-chunk-id {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: #475569;
        }

        .rag2riches-chunk-text {
            color: #1f2937;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_csv_preview(uploaded_file, csv_path: str) -> pd.DataFrame | None:
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        if not data:
            return None
        return pd.read_csv(io.BytesIO(data))

    if csv_path:
        path = Path(csv_path)
        if path.exists():
            return pd.read_csv(path)
    return None


def _materialize_csv(uploaded_file, csv_path: str) -> str:
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        temp_path = Path(tempfile.gettempdir()) / f"rag2riches_upload_{uuid4().hex}.csv"
        temp_path.write_bytes(data)
        return str(temp_path)

    return csv_path


def _embed_chunks_with_progress(
    embedder: LiteLLMEmbedder,
    chunks: list[Chunk],
    batch_size: int,
) -> list[EmbeddingRecord]:
    total = len(chunks)
    progress = st.progress(0.0)
    status = st.empty()
    records: list[EmbeddingRecord] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        status.info(f"Embedding {start + 1}-{end} of {total} chunks")
        vectors = embedder.embed_texts([c.text for c in batch])
        for chunk, vector in zip(batch, vectors):
            records.append(
                EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    vector=vector,
                    embedding_model=embedder.model_name,
                )
            )
        progress.progress(end / total)

    status.success("Embedding complete")
    return records


def _load_chunks_from_lancedb(
    path: str | Path,
    table_name: str,
    filter_expression: str | None = None,
) -> list[Chunk]:
    try:
        import lancedb
    except ImportError:
        st.error("LanceDB is not installed. Install with: pip install rag2riches[vector]")
        return []

    db = lancedb.connect(str(path))
    table = db.open_table(table_name)
    if filter_expression:
        df = table.search(None).where(filter_expression).to_pandas()
    else:
        df = table.to_pandas()

    reserved = {
        "chunk_id",
        "document_id",
        "text",
        "vector",
        "chunk_index",
        "start_char",
        "end_char",
    }

    chunks: list[Chunk] = []
    for row in df.to_dict(orient="records"):
        metadata = {k: v for k, v in row.items() if k not in reserved and not k.startswith("_")}
        chunks.append(
            Chunk(
                chunk_id=str(row.get("chunk_id")),
                document_id=str(row.get("document_id")),
                text=str(row.get("text", "")),
                metadata=metadata,
                chunk_index=int(row.get("chunk_index", 0)),
                start_char=_coerce_optional_int(row.get("start_char")),
                end_char=_coerce_optional_int(row.get("end_char")),
            )
        )

    return chunks


def _load_metadata_fields_from_lancedb(path: str | Path, table_name: str) -> list[str]:
    chunks = _load_chunks_from_lancedb(path, table_name)
    return _metadata_fields_from_chunks(chunks)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    return int(value)


def _metadata_fields_from_chunks(chunks: list[Chunk]) -> list[str]:
    fields = set()
    for chunk in chunks:
        fields.update(chunk.metadata.keys())
    return sorted(fields, key=str)


def _embedding_model_options() -> tuple[list[str], dict[str, str]]:
    options: list[str] = []
    labels: dict[str, str] = {}
    for group, models in EMBEDDING_MODEL_PRESETS:
        for model in models:
            options.append(model)
            labels[model] = f"{group} | {model}"
    options.append(CUSTOM_EMBEDDING_MODEL)
    labels[CUSTOM_EMBEDDING_MODEL] = "Custom..."
    return options, labels


def _select_embedding_model(
    default_model: str,
    key_suffix: str,
    help_text: str,
) -> str:
    options, labels = _embedding_model_options()
    default_index = options.index(default_model) if default_model in options else 0
    selected = st.selectbox(
        "Embedding model",
        options=options,
        index=default_index,
        format_func=lambda model: labels.get(model, model),
        help=help_text,
        key=f"embedding_model_select_{key_suffix}",
    )
    if selected == CUSTOM_EMBEDDING_MODEL:
        return st.text_input(
            "Custom embedding model",
            value=default_model,
            help="Example: openai/text-embedding-3-small",
            key=f"embedding_model_custom_{key_suffix}",
        )
    return selected


def _filter_value_placeholder(operator: str) -> str:
    if operator in {"in", "not in"}:
        return "Coca Cola, Pepsi"
    if operator in {">", ">=", "<", "<="}:
        return "2016"
    if operator in FILTER_NULL_OPERATORS:
        return ""
    return "value"


def _build_filter_clause(
    field: str,
    operator: str,
    raw_value: str,
) -> dict[str, Any] | None:
    if not field or not operator:
        return None

    normalized_op = _normalize_filter_operator(operator)
    if normalized_op in FILTER_NULL_OPERATORS:
        return {"field": field, "op": normalized_op}

    if not raw_value or not raw_value.strip():
        return None

    if normalized_op in {"in", "not in"}:
        values = _parse_filter_list(raw_value)
        if not values:
            return None
        return {"field": field, "op": normalized_op, "value": values}

    value = _parse_filter_scalar(raw_value)
    return {"field": field, "op": normalized_op, "value": value}


def _parse_filter_list(raw_value: str) -> list[Any]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    return [_parse_filter_scalar(part) for part in parts]


def _parse_filter_scalar(raw_value: str) -> Any:
    cleaned = raw_value.strip()
    if not cleaned:
        return ""
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def _normalize_filter_operator(operator: str) -> str:
    normalized = operator.strip().lower()
    mapping = {
        "==": "=",
        "eq": "=",
        "neq": "!=",
        "ne": "!=",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "notin": "not in",
        "isnull": "is null",
        "isnotnull": "is not null",
    }
    return mapping.get(normalized, normalized)


def _format_filter_summary(clauses: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for clause in clauses:
        field = clause.get("field")
        if not field:
            continue
        operator = _normalize_filter_operator(str(clause.get("op", "=")))
        if operator in FILTER_NULL_OPERATORS:
            label = "is null" if operator == "is null" else "is not null"
            parts.append(f"{field} {label}")
            continue
        value = clause.get("value")
        if operator in {"in", "not in"}:
            values = value if isinstance(value, (list, tuple, set)) else [value]
            values_text = ", ".join(str(item) for item in values if item is not None)
            parts.append(f"{field} {operator} ({values_text})")
        else:
            parts.append(f"{field} {operator} {value}")
    return " AND ".join(parts)


def _build_filter_spec(clauses: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not clauses:
        return None
    return {"__logic__": "and", "__clauses__": [dict(clause) for clause in clauses]}


def _combine_filter_specs(
    base_filter: dict[str, Any] | None,
    cell_filter: dict[str, Any] | None,
) -> dict[str, Any] | None:
    clauses = _filter_spec_to_clauses(base_filter)
    clauses.extend(_dict_filter_to_clauses(cell_filter))
    if not clauses:
        return None
    return {"__logic__": "and", "__clauses__": clauses}


def _filter_spec_to_clauses(filter_spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not filter_spec:
        return []
    if "__clauses__" in filter_spec:
        return [dict(clause) for clause in filter_spec.get("__clauses__", [])]
    return _dict_filter_to_clauses(filter_spec)


def _dict_filter_to_clauses(filter_dict: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not filter_dict:
        return []
    clauses = []
    for key, value in filter_dict.items():
        if str(key).startswith("__"):
            continue
        clauses.append({"field": key, "op": "=", "value": value})
    return clauses


def _filter_chunks_by_conditions(
    chunks: list[Chunk],
    clauses: list[dict[str, Any]],
) -> list[Chunk]:
    if not clauses:
        return chunks
    return [chunk for chunk in chunks if _matches_chunk_filter(chunk, clauses)]


def _matches_chunk_filter(chunk: Chunk, clauses: list[dict[str, Any]]) -> bool:
    for clause in clauses:
        field = clause.get("field")
        if not field:
            continue
        if not _evaluate_filter_clause(chunk.metadata.get(field), clause):
            return False
    return True


def _evaluate_filter_clause(actual: Any, clause: dict[str, Any]) -> bool:
    operator = _normalize_filter_operator(str(clause.get("op", "=")))
    expected = clause.get("value")

    if operator == "is null":
        return actual is None
    if operator == "is not null":
        return actual is not None
    if actual is None:
        return False
    if operator == "in":
        return isinstance(expected, (list, tuple, set)) and actual in expected
    if operator == "not in":
        return isinstance(expected, (list, tuple, set)) and actual not in expected
    if operator == "=":
        return actual == expected
    if operator == "!=":
        return actual != expected
    try:
        if operator == ">":
            return actual > expected
        if operator == ">=":
            return actual >= expected
        if operator == "<":
            return actual < expected
        if operator == "<=":
            return actual <= expected
    except Exception:
        return False
    return False


def _build_lancedb_filter_expression(clauses: list[dict[str, Any]]) -> str | None:
    if not clauses:
        return None
    expressions: list[str] = []
    for clause in clauses:
        field = clause.get("field")
        if not field:
            continue
        operator = _normalize_filter_operator(str(clause.get("op", "=")))
        field_expr = _quote_identifier_if_needed(str(field))
        if operator in FILTER_NULL_OPERATORS:
            suffix = "IS NULL" if operator == "is null" else "IS NOT NULL"
            expressions.append(f"{field_expr} {suffix}")
            continue
        if operator in {"in", "not in"}:
            values_expr = _format_filter_list(clause.get("value"))
            if not values_expr:
                continue
            keyword = "IN" if operator == "in" else "NOT IN"
            expressions.append(f"{field_expr} {keyword} ({values_expr})")
            continue
        value_expr = _format_filter_value(clause.get("value"))
        if value_expr is None:
            continue
        expressions.append(f"{field_expr} {operator} {value_expr}")
    return " AND ".join(expressions) if expressions else None


def _format_filter_list(values: Any) -> str:
    if values is None:
        return ""
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    formatted = [_format_filter_value(value) for value in values]
    formatted = [value for value in formatted if value is not None]
    return ", ".join(formatted)


def _format_filter_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _quote_identifier_if_needed(column: str) -> str:
    if column and not column[0].isdigit() and column.replace("_", "").isalnum():
        return column
    escaped = column.replace('"', '""')
    return f"\"{escaped}\""


def _run_comparative_with_progress(
    retriever: Retriever,
    generator: LiteLLMGenerator,
    cells: list[Any],
    query_text: str,
    additional_instructions: str | None,
    retrieval_k: int,
    base_filter_spec: dict[str, Any] | None = None,
) -> list[ResponseRecord]:
    total = len(cells)
    progress = st.progress(0.0)
    table_placeholder = st.empty()
    responses: list[ResponseRecord] = []
    query_id = str(uuid4())

    for idx, cell in enumerate(cells, start=1):
        try:
            combined_filter = _combine_filter_specs(base_filter_spec, cell.filter_expression)
            results = retriever.retrieve(
                query_text=query_text,
                cell_filter=combined_filter,
                k=retrieval_k,
            )
            retrieved_chunks = [r.chunk for r in results]
            retrieved_ids = [c.chunk_id for c in retrieved_chunks]
            retrieved_context = "\n\n".join(c.text for c in retrieved_chunks)

            response_text = generator.generate(
                query_text=query_text,
                retrieved_chunks=retrieved_chunks,
                additional_instructions=additional_instructions,
            )

            generator_metadata = generator.last_metadata or {}
            responses.append(
                ResponseRecord(
                    query_id=query_id,
                    query_text=query_text,
                    cell_id=cell.cell_id,
                    cell_filter=cell.fields,
                    retrieved_chunk_ids=retrieved_ids,
                    retrieved_context=retrieved_context,
                    response_text=response_text,
                    model_name=generator.model_name,
                    embedding_model_name=getattr(retriever.embedder, "model_name", ""),
                    metadata=generator_metadata,
                )
            )
        except Exception as exc:
            responses.append(
                ResponseRecord(
                    query_id=query_id,
                    query_text=query_text,
                    cell_id=cell.cell_id,
                    cell_filter=cell.fields,
                    retrieved_chunk_ids=[],
                    retrieved_context="",
                    response_text="",
                    model_name=generator.model_name,
                    embedding_model_name=getattr(retriever.embedder, "model_name", ""),
                    metadata={"error": str(exc)},
                )
            )

        progress.progress(idx / total)
        table_placeholder.dataframe(
            _responses_to_table(responses),
            use_container_width=True,
        )

    return responses


def _responses_to_table(responses: list[ResponseRecord]) -> pd.DataFrame:
    rows = []
    for response in responses:
        row = {**response.cell_filter}
        row["Response"] = _truncate_text(response.response_text)
        rows.append(row)
    return pd.DataFrame(rows)


def _truncate_text(text: str, max_len: int = 140) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _render_response_cards(responses: list[ResponseRecord]) -> None:
    if not responses:
        st.info("No responses to display yet.")
        return

    for idx, response in enumerate(responses, start=1):
        cell_label = _format_cell_fields(response.cell_filter)
        response_text = response.response_text or "No response generated."
        model_label = response.model_name or "Unknown model"
        embedding_label = response.embedding_model_name or "Unknown embedder"
        error_text = response.metadata.get("error") if response.metadata else None

        response_html = f"""
        <div class="rag2riches-response-card">
            <div class="rag2riches-response-header">
                <div>
                    <div class="rag2riches-response-title">Response {idx}</div>
                    <div class="rag2riches-response-meta">Cell: {_escape_html(cell_label)}</div>
                    <div class="rag2riches-response-meta">LLM: {_escape_html(model_label)} | Embeddings: {_escape_html(embedding_label)}</div>
                </div>
                <div class="rag2riches-response-pill">{len(response.retrieved_chunk_ids)} chunks</div>
            </div>
            <div class="rag2riches-response-text">{_escape_html(response_text)}</div>
        </div>
        """
        st.markdown(response_html, unsafe_allow_html=True)

        if error_text:
            st.error(f"Generation error: {error_text}")

        with st.expander(
            f"Show retrieved chunks ({len(response.retrieved_chunk_ids)})",
            expanded=False,
        ):
            chunk_texts = _split_retrieved_context(response.retrieved_context)
            if not chunk_texts:
                st.caption("No retrieved context available for this response.")
                continue

            for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
                chunk_id = (
                    response.retrieved_chunk_ids[chunk_index - 1]
                    if chunk_index - 1 < len(response.retrieved_chunk_ids)
                    else f"chunk-{chunk_index}"
                )
                chunk_html = f"""
                <div class="rag2riches-chunk-card">
                    <div class="rag2riches-chunk-header">
                        <div class="rag2riches-chunk-label">Chunk {chunk_index}</div>
                        <div class="rag2riches-chunk-id">{_escape_html(str(chunk_id))}</div>
                    </div>
                    <div class="rag2riches-chunk-text">{_escape_html(chunk_text)}</div>
                </div>
                """
                st.markdown(chunk_html, unsafe_allow_html=True)


def _format_cell_fields(cell_filter: dict[str, Any]) -> str:
    if not cell_filter:
        return "All data"
    parts = [f"{key}={value}" for key, value in sorted(cell_filter.items(), key=lambda item: str(item[0]))]
    return " | ".join(parts)


def _split_retrieved_context(context: str) -> list[str]:
    if not context:
        return []
    return [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]


def _escape_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br />")


if __name__ == "__main__":
    main()

