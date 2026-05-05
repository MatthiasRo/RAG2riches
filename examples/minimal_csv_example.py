"""
Minimal Example: Ingest, Clean, Chunk, and Export

This script demonstrates the simplest RAG2riches workflow:
1. Ingest a CSV file with text and metadata
2. Clean the text
3. Chunk the documents
4. Export chunks with metadata for inspection

This example uses only built-in functionality with no external API calls.

Usage:
    python examples/minimal_csv_example.py
"""

from pathlib import Path

from rag2riches import RAG2richesPipeline


def main():
    """Run minimal RAG2riches pipeline."""
    
    # For this example, we'll use the test fixture CSV
    csv_path = Path(__file__).parent.parent / "tests" / "fixtures" / "speeches_sample.csv"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("RAG2riches Minimal Example")
    print("=" * 60)

    # Step 1: Ingest from CSV
    print("\n1. Ingesting from CSV...")
    pipeline = RAG2richesPipeline.from_csv(
        path=csv_path,
        text_column="speech_text",
        metadata_columns=["party", "year", "speaker"],
    )
    print(f"   ✓ Ingested {len(pipeline.documents)} documents")
    print(f"   ✓ Metadata fields: {list(pipeline.documents[0].metadata.keys())}")

    # Step 2: Clean documents
    print("\n2. Cleaning text...")
    pipeline.clean()
    print(f"   ✓ Cleaned {len(pipeline.documents)} documents")
    print(f"   ✓ Sample cleaned text length: {len(pipeline.documents[0].text)} chars")

    # Step 3: Chunk documents
    print("\n3. Chunking documents...")
    pipeline.chunk(
        chunk_size=250,
        chunk_overlap=50,
    )
    chunks = pipeline.chunks
    print(f"   ✓ Created {len(chunks)} chunks")
    print(f"   ✓ Chunk size range: {len(chunks[0].text)} - {len(chunks[-1].text)} chars")

    # Step 4: Verify metadata inheritance
    print("\n4. Verifying metadata...")
    sample_chunk = chunks[0]
    print(f"   ✓ Sample chunk ID: {sample_chunk.chunk_id}")
    print(f"   ✓ Sample chunk metadata: {sample_chunk.metadata}")

    # Step 5: Construct cells (optional analysis)
    print("\n5. Constructing metadata cells...")
    from rag2riches import get_unique_metadata_values
    
    cells = pipeline.construct_cells(cell_fields=["party", "year"])
    print(f"   ✓ Identified {len(cells)} unique (party, year) cells")
    
    parties = get_unique_metadata_values(chunks, "party")
    years = get_unique_metadata_values(chunks, "year")
    print(f"   ✓ Parties: {parties}")
    print(f"   ✓ Years: {years}")

    # Step 6: Export chunks
    print("\n6. Exporting chunks...")
    csv_output = output_dir / "chunks.csv"
    json_output = output_dir / "chunks.json"
    
    pipeline.export_chunks(csv_output, format="csv")
    pipeline.export_chunks(json_output, format="json")
    
    print(f"   ✓ Exported to {csv_output}")
    print(f"   ✓ Exported to {json_output}")

    # Step 7: Display summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Input documents:  {len(pipeline.documents)}")
    print(f"Output chunks:    {len(chunks)}")
    print(f"Unique cells:     {len(cells)}")
    print(f"Output files:")
    print(f"  - {csv_output}")
    print(f"  - {json_output}")
    print("\nNext steps:")
    print("  1. Open chunks.csv in Excel or a data analysis tool")
    print("  2. Review metadata distribution across cells")
    print("  3. Embed chunks and run comparative queries")
    print("=" * 60)


if __name__ == "__main__":
    main()

