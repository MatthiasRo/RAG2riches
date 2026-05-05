"""
Document ingestion from various file formats.

Supports CSV, TXT, and PDF formats. Each ingester returns a list of Document objects.
"""

from pathlib import Path
from typing import Optional
from uuid import uuid4

import pandas as pd
from loguru import logger

from .types import Document


class CSVIngester:
    """Ingest documents from a CSV file."""

    def __init__(
        self,
        text_column: str,
        metadata_columns: Optional[list[str]] = None,
        document_id_column: Optional[str] = None,
        auto_generate_ids: bool = True,
    ):
        """Initialize CSV ingester.

        Args:
            text_column: Name of the column containing document text
            metadata_columns: List of columns to include as metadata
            document_id_column: Column to use as document_id (optional)
            auto_generate_ids: If True, generate IDs for rows without ID column
        """
        self.text_column = text_column
        self.metadata_columns = metadata_columns or []
        self.document_id_column = document_id_column
        self.auto_generate_ids = auto_generate_ids

    def ingest(self, path: str | Path) -> list[Document]:
        """Ingest documents from CSV file.

        Args:
            path: Path to CSV file

        Returns:
            List of Document objects
        """
        path = Path(path)
        logger.info(f"Ingesting from CSV: {path}")

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path.name}")

        if self.text_column not in df.columns:
            raise ValueError(
                f"Text column '{self.text_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        documents = []
        for idx, row in df.iterrows():
            # Get document ID
            if self.document_id_column and self.document_id_column in df.columns:
                doc_id = str(row[self.document_id_column])
            elif self.auto_generate_ids:
                doc_id = f"{path.stem}_{idx}"
            else:
                doc_id = str(uuid4())

            # Extract text
            text = str(row[self.text_column]).strip()
            if not text:
                logger.warning(f"Empty text in row {idx}, skipping")
                continue

            # Extract metadata
            metadata = {}
            for col in self.metadata_columns:
                if col in df.columns:
                    metadata[col] = row[col]

            # Create document
            doc = Document(
                document_id=doc_id,
                source_path=f"{path.name}:row_{idx}",
                text=text,
                metadata=metadata,
            )
            documents.append(doc)

        logger.info(f"Ingested {len(documents)} documents from {path.name}")
        return documents


class TXTIngester:
    """Ingest documents from text files in a directory."""

    def __init__(self, auto_generate_ids: bool = True):
        """Initialize TXT ingester.

        Args:
            auto_generate_ids: If True, use filename as document_id
        """
        self.auto_generate_ids = auto_generate_ids

    def ingest(self, directory: str | Path) -> list[Document]:
        """Ingest documents from all .txt files in a directory.

        Args:
            directory: Path to directory containing .txt files

        Returns:
            List of Document objects
        """
        directory = Path(directory)
        logger.info(f"Ingesting from directory: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        txt_files = list(directory.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} .txt files")

        documents = []
        for file_path in sorted(txt_files):
            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if not text:
                    logger.warning(f"Empty file: {file_path.name}, skipping")
                    continue

                doc_id = file_path.stem if self.auto_generate_ids else str(uuid4())

                doc = Document(
                    document_id=doc_id,
                    source_path=str(file_path),
                    text=text,
                    metadata={"filename": file_path.name, "source": "txt"},
                )
                documents.append(doc)
                logger.debug(f"Ingested: {file_path.name}")

            except Exception as e:
                logger.error(f"Error reading {file_path.name}: {e}")
                continue

        logger.info(f"Ingested {len(documents)} documents from directory")
        return documents


class PDFIngester:
    """Ingest documents from PDF files."""

    def __init__(self, auto_generate_ids: bool = True):
        """Initialize PDF ingester.

        Args:
            auto_generate_ids: If True, use filename as document_id
        """
        self.auto_generate_ids = auto_generate_ids

    def ingest(self, path: str | Path) -> list[Document]:
        """Ingest document from a PDF file.

        Note: This implementation extracts the entire PDF as a single document.
        Future versions may support page-level extraction.

        Args:
            path: Path to PDF file

        Returns:
            List of Document objects (typically one per PDF, but could split later)
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF ingestion. Install with: pip install pypdf"
            )

        path = Path(path)
        logger.info(f"Ingesting from PDF: {path}")

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        try:
            reader = PdfReader(path)
            num_pages = len(reader.pages)
            logger.info(f"PDF has {num_pages} pages")

            # Extract all text
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {e}")

            full_text = "\n".join(text_parts).strip()

            if not full_text:
                logger.warning(f"No text extracted from {path.name}")
                return []

            doc_id = path.stem if self.auto_generate_ids else str(uuid4())

            doc = Document(
                document_id=doc_id,
                source_path=str(path),
                text=full_text,
                metadata={"num_pages": num_pages, "filename": path.name, "source": "pdf"},
            )

            logger.info(f"Ingested PDF with {len(full_text)} characters")
            return [doc]

        except Exception as e:
            logger.error(f"Error reading PDF {path.name}: {e}")
            return []


def ingest_documents(
    path: str | Path,
    format: str = "csv",
    text_column: Optional[str] = None,
    metadata_columns: Optional[list[str]] = None,
    document_id_column: Optional[str] = None,
) -> list[Document]:
    """Convenience function to ingest documents in any supported format.

    Args:
        path: Path to input file or directory
        format: One of "csv", "txt", "pdf"
        text_column: For CSV: column containing text
        metadata_columns: Columns to include as metadata
        document_id_column: Column to use as document IDs

    Returns:
        List of Document objects

    Raises:
        ValueError: If format is not supported
        FileNotFoundError: If path does not exist
    """
    if format == "csv":
        if not text_column:
            raise ValueError("text_column is required for CSV format")
        ingester = CSVIngester(
            text_column=text_column,
            metadata_columns=metadata_columns,
            document_id_column=document_id_column,
        )
        return ingester.ingest(path)

    elif format == "txt":
        ingester = TXTIngester()
        return ingester.ingest(path)

    elif format == "pdf":
        ingester = PDFIngester()
        return ingester.ingest(path)

    else:
        raise ValueError(f"Unsupported format: {format}. Choose from: csv, txt, pdf")
