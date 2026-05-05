"""
Logging utilities for RAG2riches.

Provides simple logging setup and progress tracking.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger

# Configure loguru
loguru_logger.remove()  # Remove default handler
loguru_logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> None:
    """Set up logging for RAG2riches.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to write logs to file
    """
    loguru_logger.remove()
    loguru_logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        loguru_logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            level=level,
        )


def get_logger(name: str):
    """Get a logger instance.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Logger instance
    """
    return loguru_logger.bind(name=name)

