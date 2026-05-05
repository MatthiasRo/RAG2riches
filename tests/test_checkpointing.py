"""Tests for checkpointing utilities."""

from pathlib import Path

from rag2riches.checkpointing import CheckpointManager, append_response_records, load_response_records
from rag2riches.types import ResponseRecord


def _make_response(query_id: str, cell_id: str) -> ResponseRecord:
    return ResponseRecord(
        query_id=query_id,
        query_text="test query",
        cell_id=cell_id,
        cell_filter={"party": "D"},
        retrieved_chunk_ids=["c1"],
        retrieved_context="context",
        response_text="response",
        model_name="mock",
        embedding_model_name="mock-embed",
        metadata={},
    )


def test_append_and_load(tmp_path: Path):
    output_path = tmp_path / "responses.jsonl"
    records = [_make_response("q1", "cell1"), _make_response("q1", "cell2")]

    count = append_response_records(records, output_path)
    assert count == 2

    loaded = load_response_records(output_path)
    assert len(loaded) == 2
    assert loaded[0].cell_id == "cell1"


def test_checkpoint_manager_filters(tmp_path: Path):
    output_path = tmp_path / "responses.jsonl"
    manager = CheckpointManager(path=output_path)

    existing = [_make_response("q1", "cell1")]
    manager.append(existing)

    incoming = [_make_response("q1", "cell1"), _make_response("q1", "cell2")]
    remaining = manager.filter_unprocessed(incoming)

    assert len(remaining) == 1
    assert remaining[0].cell_id == "cell2"

