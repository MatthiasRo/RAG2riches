"""
Text chunking utilities for RAG2riches.

Provides character-level chunking with overlap and allows custom chunking functions.
"""

from typing import Callable, Optional
from uuid import uuid4

from loguru import logger

from .types import Chunk, Document


def chunk_text(
    text: str,
    chunk_size: int = 750,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split text into chunks by character count with overlap.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks

    Returns:
        List of chunk strings
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        # Move start position for next chunk
        start = end - chunk_overlap
        if start <= 0 and end < len(text):
            start = chunk_size - chunk_overlap

    return chunks


def chunks_from_documents(
    documents: list[Document],
    chunk_size: int = 750,
    chunk_overlap: int = 100,
    chunk_fn: Optional[Callable[[str], list[str]]] = None,
) -> list[Chunk]:
    """Create chunks from a list of documents.

    Chunks inherit all document-level metadata and add chunk-specific metadata
    (chunk_index, start_char, end_char).

    Args:
        documents: List of Document objects
        chunk_size: Characters per chunk (ignored if chunk_fn provided)
        chunk_overlap: Characters to overlap (ignored if chunk_fn provided)
        chunk_fn: Custom chunking function that takes text and returns list of chunk strings.
                 If None, uses default character-based chunking.

    Returns:
        List of Chunk objects
    """
    if chunk_fn is None:
        chunk_fn = lambda text: chunk_text(text, chunk_size, chunk_overlap)

    logger.info(f"Chunking {len(documents)} documents")
    all_chunks = []
    total_chunks = 0

    for doc in documents:
        # Get chunks
        chunk_texts = chunk_fn(doc.text)

        # Track character positions
        char_pos = 0
        for chunk_idx, chunk_text_value in enumerate(chunk_texts):
            # Find where this chunk starts in the original text
            start_char = doc.text.find(chunk_text_value, char_pos)
            if start_char == -1:
                # Fallback if exact match not found (shouldn't happen in normal cases)
                start_char = char_pos
            end_char = start_char + len(chunk_text_value)

            # Create chunk with inherited metadata
            chunk = Chunk(
                chunk_id=str(uuid4()),
                document_id=doc.document_id,
                text=chunk_text_value,
                metadata=doc.metadata.copy(),  # Inherit document metadata
                chunk_index=chunk_idx,
                start_char=start_char,
                end_char=end_char,
            )
            all_chunks.append(chunk)
            char_pos = end_char - chunk_overlap if chunk_idx < len(chunk_texts) - 1 else end_char
            total_chunks += 1

    logger.info(f"Created {total_chunks} chunks from {len(documents)} documents")
    return all_chunks

