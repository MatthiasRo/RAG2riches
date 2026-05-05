"""
Text cleaning utilities for RAG2riches.

Provides default cleaning function and allows custom cleaning pipelines.
"""

import re
from typing import Callable, Optional

from loguru import logger

from .types import Document


def default_clean_text(text: str) -> str:
    """Apply default text cleaning to a document.

    Default cleaning includes:
    - Normalize whitespace (multiple spaces -> single space)
    - Remove extra newlines (multiple newlines -> single newline)
    - Strip leading/trailing whitespace

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of spaces and tabs, but keep newlines intact
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse multiple newlines to a single newline
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove stray spaces around newline boundaries
    text = re.sub(r" *\n *", "\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_documents(
    documents: list[Document],
    clean_fn: Optional[Callable[[str], str]] = None,
    preserve_raw: bool = True,
) -> list[Document]:
    """Clean a list of documents.

    Args:
        documents: List of Document objects
        clean_fn: Custom cleaning function. If None, uses default_clean_text.
        preserve_raw: If True, store original text in document.raw

    Returns:
        List of Documents with cleaned text
    """
    if clean_fn is None:
        clean_fn = default_clean_text

    logger.info(f"Cleaning {len(documents)} documents")
    cleaned = []

    for doc in documents:
        if preserve_raw and not doc.raw:
            doc.raw = doc.text

        cleaned_text = clean_fn(doc.text)
        doc.text = cleaned_text
        cleaned.append(doc)

    logger.info(f"Cleaned {len(cleaned)} documents")
    return cleaned

