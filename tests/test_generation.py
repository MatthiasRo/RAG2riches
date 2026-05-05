"""Tests for generation module."""

import pytest

import rag2riches.generation as generation
from rag2riches.generation import MockGenerator
from rag2riches.types import Chunk


class TestMockGenerator:
    """Test MockGenerator behavior."""

    def test_generate_requires_query(self):
        generator = MockGenerator()
        with pytest.raises(ValueError, match="query_text must be non-empty"):
            generator.generate("", [])

    def test_generate_includes_chunk_count(self):
        generator = MockGenerator()
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="alpha text",
                metadata={},
                chunk_index=0,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d2",
                text="beta text",
                metadata={},
                chunk_index=0,
            ),
        ]
        result = generator.generate("test query", chunks)
        assert "Chunks: 2" in result

    def test_generate_includes_instruction(self):
        generator = MockGenerator()
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                text="alpha text",
                metadata={},
                chunk_index=0,
            ),
        ]
        result = generator.generate("test query", chunks, additional_instructions="Be brief")
        assert "Instructions: Be brief" in result


def test_litellm_generator_builds_prompt(monkeypatch):
    captured = {}

    def fake_completion(**params):
        captured["params"] = params
        return {"choices": [{"message": {"content": "ok"}}]}

    class FakeLiteLLM:
        completion = staticmethod(fake_completion)

        @staticmethod
        def get_llm_provider(model):
            return "mock"

    monkeypatch.setattr(generation, "_import_litellm", lambda: FakeLiteLLM)

    generator = generation.LiteLLMGenerator(model="mock-model")
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            text="alpha text",
            metadata={},
            chunk_index=0,
        )
    ]

    result = generator.generate(
        "test query",
        chunks,
        additional_instructions="Be brief",
    )

    assert result == "ok"
    user_message = captured["params"]["messages"][1]["content"]
    assert "Additional instructions: Be brief" in user_message
    assert generator.last_metadata is not None
    assert generator.last_metadata["prompt_version"] == generation.PROMPT_VERSION

