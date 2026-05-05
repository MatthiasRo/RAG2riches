"""
High-level pipeline API for RAG2riches.

RAG2richesPipeline provides a simple, end-to-end interface for ingestion, cleaning,
chunking, embedding, indexing, and comparative querying.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .chunking import chunks_from_documents
from .cleaning import clean_documents
from .compare import ComparativeRunner
from .embeddings import Embedder, LiteLLMEmbedder
from .export import export_chunks, export_responses
from .generation import Generator, LiteLLMGenerator
from .ingestion import CSVIngester, PDFIngester, TXTIngester
from .metadata import construct_cells, get_unique_metadata_values
from .retrieval import Retriever
from .types import Chunk, Document, EmbeddingRecord, QuerySpec, ResponseRecord
from .vectorstores import LanceDBVectorStore, VectorStore


class RAG2richesPipeline:
    """High-level pipeline for comparative RAG workflows."""

    def __init__(self, documents: list[Document] | None = None):
        self.documents: list[Document] = documents or []
        self.chunks: list[Chunk] = []
        self.embeddings: list[EmbeddingRecord] = []
        self.vector_store: VectorStore | None = None
        self.embedder: Embedder | None = None
        self.retriever: Retriever | None = None
        self.generator: Generator | None = None
        self.responses: list[ResponseRecord] = []

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        text_column: str,
        metadata_columns: list[str] | None = None,
        document_id_column: str | None = None,
        auto_generate_ids: bool = True,
    ) -> "RAG2richesPipeline":
        """Create a pipeline from a CSV file."""
        ingester = CSVIngester(
            text_column=text_column,
            metadata_columns=metadata_columns,
            document_id_column=document_id_column,
            auto_generate_ids=auto_generate_ids,
        )
        documents = ingester.ingest(path)
        return cls(documents=documents)

    @classmethod
    def from_txt_directory(
        cls,
        path: str | Path,
        auto_generate_ids: bool = True,
    ) -> "RAG2richesPipeline":
        """Create a pipeline from a directory of .txt files."""
        ingester = TXTIngester(auto_generate_ids=auto_generate_ids)
        documents = ingester.ingest(path)
        return cls(documents=documents)

    @classmethod
    def from_pdf(
        cls,
        path: str | Path,
        auto_generate_ids: bool = True,
    ) -> "RAG2richesPipeline":
        """Create a pipeline from a PDF file."""
        ingester = PDFIngester(auto_generate_ids=auto_generate_ids)
        documents = ingester.ingest(path)
        return cls(documents=documents)

    def clean(self, preserve_raw: bool = True, clean_fn: Any | None = None) -> "RAG2richesPipeline":
        """Clean documents in place."""
        self.documents = clean_documents(self.documents, clean_fn=clean_fn, preserve_raw=preserve_raw)
        return self

    def chunk(
        self,
        chunk_size: int = 750,
        chunk_overlap: int = 100,
        chunk_fn: Any | None = None,
    ) -> "RAG2richesPipeline":
        """Chunk documents and store results in the pipeline."""
        self.chunks = chunks_from_documents(
            self.documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_fn=chunk_fn,
        )
        return self

    def embed(
        self,
        model: str | None = None,
        embedder: Embedder | None = None,
        batch_size: int = 32,
        cache_path: str | Path | None = None,
        resume: bool = True,
        **kwargs: Any,
    ) -> "RAG2richesPipeline":
        """Embed chunks using the provided embedder or LiteLLM model."""
        if not self.chunks:
            raise ValueError("No chunks available. Run chunk() first.")

        if embedder is None:
            if not model:
                raise ValueError("Either embedder or model must be provided")
            embedder = LiteLLMEmbedder(
                model=model,
                batch_size=batch_size,
                cache_path=cache_path,
                **kwargs,
            )

        self.embedder = embedder

        if resume and self.embeddings and len(self.embeddings) == len(self.chunks):
            logger.info("Embeddings already present; skipping re-embed")
            return self

        chunk_ids = [c.chunk_id for c in self.chunks]
        texts = [c.text for c in self.chunks]
        if hasattr(embedder, "embed_records"):
            self.embeddings = embedder.embed_records(chunk_ids, texts)  # type: ignore[call-arg]
        else:
            vectors = embedder.embed_texts(texts)
            model_name = getattr(embedder, "model_name", "")
            self.embeddings = [
                EmbeddingRecord(
                    chunk_id=chunk_id,
                    vector=vector,
                    embedding_model=model_name,
                )
                for chunk_id, vector in zip(chunk_ids, vectors)
            ]
        return self

    def index(
        self,
        vector_store: VectorStore | None = None,
        store_type: str = "lancedb",
        path: str | Path | None = None,
        table_name: str = "chunks",
    ) -> "RAG2richesPipeline":
        """Index embeddings into a vector store."""
        if not self.chunks or not self.embeddings:
            raise ValueError("Chunks and embeddings are required before indexing")
        if self.embedder is None:
            raise ValueError("Embedder is required before indexing")

        if vector_store is None:
            if store_type == "lancedb":
                vector_store = LanceDBVectorStore(path=path, table_name=table_name)
            else:
                raise ValueError("Unsupported vector store type")

        vector_store.create_or_connect(path=path, table_name=table_name)
        vector_store.add_chunks(self.chunks, self.embeddings)

        self.vector_store = vector_store
        self.retriever = Retriever(embedder=self.embedder, vector_store=vector_store)
        return self

    def compare(
        self,
        queries: str | list[str] | list[QuerySpec],
        cell_fields: list[str] | None = None,
        retrieval_k: int = 5,
        additional_instructions: str | None = None,
        generation_model: str | None = None,
        generator: Generator | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> list[ResponseRecord]:
        """Run comparative queries across metadata-defined cells."""
        if not self.chunks:
            raise ValueError("No chunks available. Run chunk() first.")
        if self.retriever is None:
            if self.embedder is None or self.vector_store is None:
                raise ValueError("Embedder and vector store are required for comparison")
            self.retriever = Retriever(embedder=self.embedder, vector_store=self.vector_store)

        if generator is None:
            if generation_model is None:
                generation_model = _infer_generation_model(queries)
            if generation_model is None:
                raise ValueError("Either generator or generation_model must be provided")
            generator = LiteLLMGenerator(model=generation_model)
        self.generator = generator

        if resume and checkpoint_path is None:
            raise ValueError("checkpoint_path is required when resume=True")

        runner = ComparativeRunner(retriever=self.retriever, generator=self.generator)

        query_specs = _normalize_queries(queries, cell_fields, retrieval_k, additional_instructions, generation_model)

        responses = runner.run_queries_across_cells(
            queries=query_specs,
            chunks=self.chunks,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )

        self.responses = responses
        return responses

    def export_chunks(self, output_path: str | Path, format: str = "csv") -> None:
        """Export chunks to disk."""
        export_chunks(self.chunks, output_path, format=format)

    def export_responses(self, output_path: str | Path, format: str = "csv") -> None:
        """Export responses to disk."""
        export_responses(self.responses, output_path, format=format)

    def get_metadata_values(self, field: str) -> list[Any]:
        """Return unique values for a metadata field across chunks."""
        if not self.chunks:
            return []
        return get_unique_metadata_values(self.chunks, field)

    def construct_cells(self, cell_fields: list[str]):
        """Construct comparison cells from current chunks."""
        return construct_cells(self.chunks, cell_fields)


def _normalize_queries(
    queries: str | list[str] | list[QuerySpec],
    cell_fields: list[str] | None,
    retrieval_k: int,
    additional_instructions: str | None,
    generation_model: str | None,
) -> list[QuerySpec]:
    if isinstance(queries, str):
        queries_list: Iterable[str] = [queries]
        if not cell_fields:
            raise ValueError("cell_fields is required when queries is a string")
        return [
            QuerySpec(
                query_id=str(uuid4()),
                query_text=query,
                cell_fields=cell_fields,
                retrieval_k=retrieval_k,
                additional_instructions=additional_instructions,
                generation_model=generation_model or "",
            )
            for query in queries_list
        ]

    if queries and isinstance(queries[0], QuerySpec):
        return list(queries)  # type: ignore[return-value]

    if not cell_fields:
        raise ValueError("cell_fields is required when queries is a list of strings")

    return [
        QuerySpec(
            query_id=str(uuid4()),
            query_text=query,
            cell_fields=cell_fields,
            retrieval_k=retrieval_k,
            additional_instructions=additional_instructions,
            generation_model=generation_model or "",
        )
        for query in queries  # type: ignore[arg-type]
    ]


def _infer_generation_model(queries: str | list[str] | list[QuerySpec]) -> str | None:
    if isinstance(queries, list) and queries and isinstance(queries[0], QuerySpec):
        return queries[0].generation_model
    return None

