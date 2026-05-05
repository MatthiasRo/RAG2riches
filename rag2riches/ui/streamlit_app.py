"""
RAG2riches Streamlit UI.

Run with:
    streamlit run rag2riches/ui/streamlit_app.py
"""

from __future__ import annotations

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
            embed_model = st.text_input(
                "Embedding model",
                value="openai/text-embedding-3-small",
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
            embed_model = st.text_input(
                "Embedding model used for DB",
                value="openai/text-embedding-3-small",
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
            chunks = _load_chunks_from_lancedb(lancedb_path, table_name)
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
        )

        export_responses(responses, output_path, format=output_format)
        st.success(f"Saved {len(responses)} responses to {output_path}")


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


def _load_chunks_from_lancedb(path: str | Path, table_name: str) -> list[Chunk]:
    try:
        import lancedb
    except ImportError:
        st.error("LanceDB is not installed. Install with: pip install rag2riches[vector]")
        return []

    db = lancedb.connect(str(path))
    table = db.open_table(table_name)
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


def _run_comparative_with_progress(
    retriever: Retriever,
    generator: LiteLLMGenerator,
    cells: list[Any],
    query_text: str,
    additional_instructions: str | None,
    retrieval_k: int,
) -> list[ResponseRecord]:
    total = len(cells)
    progress = st.progress(0.0)
    table_placeholder = st.empty()
    responses: list[ResponseRecord] = []
    query_id = str(uuid4())

    for idx, cell in enumerate(cells, start=1):
        try:
            results = retriever.retrieve(
                query_text=query_text,
                cell_filter=cell.filter_expression,
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


if __name__ == "__main__":
    main()

