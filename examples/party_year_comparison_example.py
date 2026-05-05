"""
Party-Year Comparison Example (Mock)

This example runs a comparative query across party-year cells using mock
embeddings and generation, so it works without API keys.
"""

from pathlib import Path

from rag2riches import (
    InMemoryVectorStore,
    MockEmbedder,
    MockGenerator,
    QuerySpec,
    RAG2richesPipeline,
)


def main() -> None:
    csv_path = Path(__file__).parent.parent / "tests" / "fixtures" / "speeches_sample.csv"

    pipeline = RAG2richesPipeline.from_csv(
        path=csv_path,
        text_column="speech_text",
        metadata_columns=["party", "year", "speaker"],
    )

    pipeline.clean()
    pipeline.chunk(chunk_size=300, chunk_overlap=60)
    pipeline.embed(embedder=MockEmbedder(dim=8))
    pipeline.index(vector_store=InMemoryVectorStore())

    queries = [
        QuerySpec(
            query_text="How does the corpus discuss climate regulation?",
            cell_fields=["party", "year"],
            retrieval_k=2,
        )
    ]

    responses = pipeline.compare(
        queries=queries,
        generator=MockGenerator(),
    )

    for response in responses:
        print(f"{response.cell_id}: {response.response_text}")


if __name__ == "__main__":
    main()

