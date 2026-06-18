"""Pytest configuration for the RAG2riches test suite."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _has_api_credentials() -> bool:
    credential_env_vars = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "VOYAGE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "LITELLM_API_KEY",
    )
    return any(os.getenv(name) for name in credential_env_vars)


def pytest_sessionstart(session) -> None:  # pragma: no cover - pytest hook
    if not _has_api_credentials():
        warnings.warn(
            "No API credentials were found in the environment. "
            "Mock-based tests will still run, but provider-backed paths cannot be fully validated.",
            UserWarning,
            stacklevel=2,
        )