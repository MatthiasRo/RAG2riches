# Changelog

All notable changes to RAG2riches will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation updates for alpha release

## [0.1.0a1] - 2026-05-04

### Initial Alpha Release

First alpha release with end-to-end comparative RAG features:

- ✅ Core data types: Document, Chunk, EmbeddingRecord, Cell, QuerySpec, ResponseRecord
- ✅ CSV, TXT, and PDF ingestion with metadata handling
- ✅ Text cleaning with whitespace normalization
- ✅ Character-based chunking with configurable overlap
- ✅ Metadata propagation and cell construction utilities
- ✅ Export to CSV and JSON/JSONL
- ✅ Embedder interface with MockEmbedder and LiteLLMEmbedder (batching, caching, retries)
- ✅ VectorStore interface with InMemoryVectorStore and LanceDBVectorStore
- ✅ Retriever with metadata pre-filtering
- ✅ Generator interface with MockGenerator and LiteLLMGenerator (grounded prompts)
- ✅ ComparativeRunner with checkpointing and resume
- ✅ High-level RAG2richesPipeline API
- ✅ Streamlit UI for interactive comparative runs
- ✅ Comprehensive tests and documentation
