"""Tests for embedder module."""

import pytest

import rag2riches.embeddings as embeddings
from rag2riches.embeddings import MockEmbedder


class TestMockEmbedder:
    """Test MockEmbedder behavior."""

    def test_embed_query_returns_correct_dim(self):
        embedder = MockEmbedder(dim=8)
        vec = embedder.embed_query("hello")
        assert len(vec) == 8

    def test_embed_texts_length_matches(self):
        embedder = MockEmbedder(dim=6)
        texts = ["a", "b", "c"]
        vecs = embedder.embed_texts(texts)
        assert len(vecs) == len(texts)
        assert all(len(v) == 6 for v in vecs)

    def test_deterministic_embeddings(self):
        embedder = MockEmbedder(dim=10)
        v1 = embedder.embed_query("same text")
        v2 = embedder.embed_query("same text")
        assert v1 == v2

    def test_different_texts_different_embeddings(self):
        embedder = MockEmbedder(dim=10)
        v1 = embedder.embed_query("text one")
        v2 = embedder.embed_query("text two")
        assert v1 != v2

    def test_embed_records_length_mismatch_raises(self):
        embedder = MockEmbedder(dim=4)
        with pytest.raises(ValueError, match="same length"):
            embedder.embed_records(["c1"], ["text1", "text2"])


def test_litellm_embedder_uses_cache(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_embedding(**params):
        calls["count"] += 1
        inputs = params["input"]
        return {
            "data": [
                {"embedding": [float(idx)] * 3, "index": idx}
                for idx, _ in enumerate(inputs)
            ]
        }

    class FakeLiteLLM:
        embedding = staticmethod(fake_embedding)

    monkeypatch.setattr(embeddings, "_import_litellm", lambda: FakeLiteLLM)

    cache_path = tmp_path / "embeddings.jsonl"
    embedder = embeddings.LiteLLMEmbedder(
        model="mock-embed",
        batch_size=10,
        cache_path=cache_path,
    )

    first = embedder.embed_texts(["alpha", "beta"])
    second = embedder.embed_texts(["alpha", "beta"])

    assert calls["count"] == 1
    assert first == second

