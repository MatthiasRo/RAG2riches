"""
Generation module for RAG2riches.

Defines a Generator interface, a deterministic MockGenerator, and a
LiteLLM-backed generator with grounded prompting and retries.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from loguru import logger

from .prompts import DEFAULT_SYSTEM_PROMPT, PROMPT_VERSION, build_user_prompt
from .types import Chunk


class Generator(ABC):
    """Abstract generator interface."""

    @abstractmethod
    def generate(
        self,
        query_text: str,
        retrieved_chunks: list[Chunk],
        additional_instructions: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from retrieved context.

        Args:
            query_text: The user query.
            retrieved_chunks: Chunks used as context.
            additional_instructions: Extra instructions appended to the prompt.
            model: Optional model name.
            **kwargs: Provider-specific parameters.

        Returns:
            Generated response text.
        """


class MockGenerator(Generator):
    """Deterministic generator for tests and local runs."""

    def __init__(self, model_name: str = "mock-generator"):
        self.model_name = model_name

    def generate(
        self,
        query_text: str,
        retrieved_chunks: list[Chunk],
        additional_instructions: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not query_text:
            raise ValueError("query_text must be non-empty")

        # Build a simple, deterministic response
        context_count = len(retrieved_chunks)
        snippet = ""
        if retrieved_chunks:
            snippet = retrieved_chunks[0].text[:120].replace("\n", " ")

        instruction_note = ""
        if additional_instructions:
            instruction_note = f" Instructions: {additional_instructions.strip()}"

        return (
            f"[MOCK RESPONSE] Query: {query_text}. "
            f"Chunks: {context_count}. "
            f"Snippet: {snippet}.{instruction_note}"
        )


class LiteLLMGenerator(Generator):
    """LiteLLM-backed generator with grounded prompting and retries."""

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 20.0,
        request_timeout: float | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        default_params: dict[str, Any] | None = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.model = model
        self.model_name = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.request_timeout = request_timeout
        self.api_base = api_base
        self.api_key = api_key
        self.organization = organization
        self.default_params = default_params or {}
        self.last_metadata: dict[str, Any] | None = None

    def generate(
        self,
        query_text: str,
        retrieved_chunks: list[Chunk],
        additional_instructions: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not query_text:
            raise ValueError("query_text must be non-empty")

        litellm = _import_litellm()
        user_prompt = build_user_prompt(
            query_text=query_text,
            retrieved_chunks=retrieved_chunks,
            additional_instructions=additional_instructions,
        )

        params: dict[str, Any] = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key
        if self.organization:
            params["organization"] = self.organization
        if self.request_timeout is not None:
            params["request_timeout"] = self.request_timeout
        params.update(self.default_params)
        params.update(kwargs)

        response = _with_retries(
            lambda: litellm.completion(**params),
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
            backoff_max=self.backoff_max,
        )

        content = _extract_content(response)
        self.last_metadata = {
            "model": params["model"],
            "prompt_version": PROMPT_VERSION,
            "system_prompt": self.system_prompt,
            "generation_params": {k: v for k, v in params.items() if k not in {"messages"}},
        }
        try:
            provider = litellm.get_llm_provider(params["model"])
            self.last_metadata["provider"] = provider
        except Exception:
            self.last_metadata["provider"] = "unknown"

        logger.debug(f"Generated response with {params['model']}")
        return content


def _import_litellm():
    try:
        import litellm
    except ImportError as exc:
        raise ImportError(
            "LiteLLM is required for LiteLLMGenerator. "
            "Install with: pip install rag2riches[llm]"
        ) from exc
    return litellm
def _extract_content(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices", [])
    else:
        choices = getattr(response, "choices", [])

    if not choices:
        raise ValueError("No choices returned by generator")

    choice = choices[0]
    if isinstance(choice, dict):
        message = choice.get("message")
        if message and isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        if choice.get("text"):
            return str(choice["text"])
    else:
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content:
                return str(content)
        text = getattr(choice, "text", None)
        if text:
            return str(text)

    raise ValueError("Could not extract content from generator response")


def _with_retries(
    func: Callable[[], Any],
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
):
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - external failures vary
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_for = min(backoff_max, backoff_base * (2**attempt))
            time.sleep(sleep_for)
    if last_exc:
        raise last_exc
    raise RuntimeError("Generation request failed")

