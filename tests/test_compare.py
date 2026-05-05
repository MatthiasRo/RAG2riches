"""Tests for comparative runner."""

from rag2riches.compare import ComparativeRunner
from rag2riches.embeddings import MockEmbedder
from rag2riches.generation import MockGenerator
from rag2riches.retrieval import Retriever
from rag2riches.types import Chunk, QuerySpec
from rag2riches.vectorstores.in_memory import InMemoryVectorStore


class TestComparativeRunner:
    """Test ComparativeRunner behavior."""

    def _make_runner(self):
        embedder = MockEmbedder(dim=8)
        store = InMemoryVectorStore()
        retriever = Retriever(embedder=embedder, vector_store=store, default_k=1)
        generator = MockGenerator()
        return embedder, store, ComparativeRunner(retriever=retriever, generator=generator)

    def test_run_query_across_cells(self):
        embedder, store, runner = self._make_runner()

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

        responses = runner.run_query_across_cells(
            query_text="climate",
            chunks=chunks,
            cell_fields=["party", "year"],
            k=1,
        )

        assert len(responses) == 2
        cell_ids = {r.cell_id for r in responses}
        assert "party=D|year=2020" in cell_ids
        assert "party=R|year=2020" in cell_ids

    def test_run_queries_across_cells(self):
        embedder, store, runner = self._make_runner()

        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="alpha climate text",
                metadata={"party": "D", "year": 2020},
                chunk_index=0,
            ),
        ]

        embeddings = embedder.embed_records(
            [c.chunk_id for c in chunks],
            [c.text for c in chunks],
        )
        store.add_chunks(chunks, embeddings)

        queries = [
            QuerySpec(query_text="climate", cell_fields=["party", "year"], retrieval_k=1),
            QuerySpec(query_text="policy", cell_fields=["party", "year"], retrieval_k=1),
        ]

        responses = runner.run_queries_across_cells(queries=queries, chunks=chunks)
        assert len(responses) == 2

