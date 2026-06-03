"""Tests for batch embedding helpers."""

import json

from rag2riches.batch_embeddings import (
    JsonlBatchWriter,
    build_batch_request_line,
    parse_openai_batch_output_line,
)


def test_build_batch_request_line_includes_extra_body():
    line = build_batch_request_line(
        chunk_id="c1",
        text="hello",
        model="text-embedding-3-small",
        extra_body={"encoding_format": "float"},
    )

    assert line["custom_id"] == "c1"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/embeddings"
    assert line["body"]["input"] == "hello"
    assert line["body"]["model"] == "text-embedding-3-small"
    assert line["body"]["encoding_format"] == "float"


def test_parse_openai_batch_output_line():
    payload = {
        "custom_id": "c1",
        "response": {
            "status_code": 200,
            "body": {"data": [{"embedding": [0.1, 0.2]}]},
        },
    }
    parsed = parse_openai_batch_output_line(json.dumps(payload))
    assert parsed == ("c1", [0.1, 0.2])


def test_jsonl_batch_writer_rollover(tmp_path):
    writer = JsonlBatchWriter(output_dir=tmp_path, max_lines=1, max_bytes=10_000)
    writer.write(build_batch_request_line("c1", "hello", "model"))
    writer.write(build_batch_request_line("c2", "world", "model"))

    paths = writer.close()
    assert len(paths) == 2

    for path in paths:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
