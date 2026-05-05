"""Tests for retrieval module."""

from rag2riches.embeddings import MockEmbedder
from rag2riches.retrieval import Retriever
from rag2riches.types import Chunk
from rag2riches.vectorstores.in_memory import InMemoryVectorStore


class TestRetriever:
    """Test Retriever behavior."""

    def test_retrieve_with_filter(self):
        embedder = MockEmbedder(dim=8)
        store = InMemoryVectorStore()

        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="alpha climate text",
                metadata={"party": "D", "year": 2020},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d2",
                text="beta market text",
                metadata={"party": "R", "year": 2020},
                chunk_index=0,
            ),
        ]

        embeddings = embedder.embed_records(
            [c.chunk_id for c in chunks],
            [c.text for c in chunks],
        )
        store.add_chunks(chunks, embeddings)

        retriever = Retriever(embedder=embedder, vector_store=store, default_k=1)

        results = retriever.retrieve(
            query_text="climate",
            cell_filter={"party": "D"},
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"

    def test_retrieve_respects_k(self):
        embedder = MockEmbedder(dim=8)
        store = InMemoryVectorStore()

        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="alpha text",
                metadata={"party": "D"},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d2",
                text="alpha text two",
                metadata={"party": "D"},
                chunk_index=0,
            ),
        ]

        embeddings = embedder.embed_records(
            [c.chunk_id for c in chunks],
            [c.text for c in chunks],
        )
        store.add_chunks(chunks, embeddings)

        retriever = Retriever(embedder=embedder, vector_store=store, default_k=2)

        results = retriever.retrieve(
            query_text="alpha",
            cell_filter={"party": "D"},
            k=2,
        )

        assert len(results) == 2

