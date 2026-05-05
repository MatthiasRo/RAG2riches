"""Tests for RAG2richesPipeline."""

from pathlib import Path

from rag2riches import (
    InMemoryVectorStore,
    MockEmbedder,
    MockGenerator,
    QuerySpec,
    RAG2richesPipeline,
)


def test_pipeline_basic_compare():
    csv_path = Path(__file__).parent / "fixtures" / "speeches_sample.csv"

    pipeline = RAG2richesPipeline.from_csv(
        path=csv_path,
        text_column="speech_text",
        metadata_columns=["party", "year"],
    )
    pipeline.clean()
    pipeline.chunk(chunk_size=200, chunk_overlap=50)
    pipeline.embed(embedder=MockEmbedder(dim=8))
    pipeline.index(vector_store=InMemoryVectorStore())

    cells = pipeline.construct_cells(["party", "year"])
    query = QuerySpec(
        query_text="How does the corpus discuss climate?",
        cell_fields=["party", "year"],
        retrieval_k=1,
    )

    responses = pipeline.compare(
        queries=[query],
        generator=MockGenerator(),
    )

    cell_ids = {cell.cell_id for cell in cells}
    assert len(responses) == len(cells)
    assert all(response.cell_id in cell_ids for response in responses)
    assert pipeline.responses == responses

