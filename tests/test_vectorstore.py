"""Tests for vector stores."""

import pytest

from rag2riches.embeddings import MockEmbedder
from rag2riches.types import Chunk
from rag2riches.vectorstores.in_memory import InMemoryVectorStore


class TestInMemoryVectorStore:
    """Test in-memory vector store behavior."""

    def test_add_and_get_chunk(self):
        store = InMemoryVectorStore()
        embedder = MockEmbedder(dim=6)

        chunk = Chunk(
            chunk_id="c1",
            document_id="d1",
            text="alpha text",
            metadata={"party": "D", "year": 2020},
            chunk_index=0,
        )

        emb = embedder.embed_records([chunk.chunk_id], [chunk.text])[0]
        store.add_chunks([chunk], [emb])

        retrieved = store.get_chunk("c1")
        assert retrieved is not None
        assert retrieved.text == "alpha text"

    def test_similarity_search_with_filter(self):
        store = InMemoryVectorStore()
        embedder = MockEmbedder(dim=8)

        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="alpha text",
                metadata={"party": "D", "year": 2020},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d2",
                text="beta text",
                metadata={"party": "R", "year": 2020},
                chunk_index=0,
            ),
        ]

        embeddings = embedder.embed_records(
            [c.chunk_id for c in chunks],
            [c.text for c in chunks],
        )
        store.add_chunks(chunks, embeddings)

        query_vector = embedder.embed_query("alpha text")
        results = store.similarity_search(query_vector, k=1, filter={"party": "D"})

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"

    def test_list_metadata_values(self):
        store = InMemoryVectorStore()
        embedder = MockEmbedder(dim=6)

        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="alpha",
                metadata={"party": "D", "year": 2020},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d2",
                text="beta",
                metadata={"party": "R", "year": 2021},
                chunk_index=0,
            ),
        ]

        embeddings = embedder.embed_records(
            [c.chunk_id for c in chunks],
            [c.text for c in chunks],
        )
        store.add_chunks(chunks, embeddings)

        values = store.list_metadata_values(["party", "year"])
        assert values["party"] == ["D", "R"]
        assert values["year"] == [2020, 2021]


def test_lancedb_vectorstore_basic(tmp_path):
    pytest.importorskip("lancedb")

    from rag2riches.vectorstores.lancedb_store import LanceDBVectorStore

    store = LanceDBVectorStore(path=tmp_path / "lancedb", table_name="chunks")
    store.create_or_connect()

    embedder = MockEmbedder(dim=6)
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            text="alpha text",
            metadata={"party": "D", "year": 2020},
            chunk_index=0,
        ),
        Chunk(
            chunk_id="c2",
            document_id="d2",
            text="beta text",
            metadata={"party": "R", "year": 2020},
            chunk_index=0,
        ),
    ]

    embeddings = embedder.embed_records(
        [c.chunk_id for c in chunks],
        [c.text for c in chunks],
    )
    store.add_chunks(chunks, embeddings)

    query_vector = embedder.embed_query("alpha text")
    results = store.similarity_search(query_vector, k=1, filter={"party": "D"})

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"

