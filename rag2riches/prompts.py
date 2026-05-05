"""
Prompt templates for RAG2riches.

Provides default system and user prompt templates for grounded generation.
"""

from __future__ import annotations

from .types import Chunk

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful research assistant. Use only the provided context to answer. "
    "If the answer cannot be inferred from the context, reply with: I don't know."
)

PROMPT_VERSION = "v1"

DEFAULT_USER_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\nQuestion: {question}{instructions_block}"
)


def format_context(retrieved_chunks: list[Chunk]) -> str:
    """Format retrieved chunks into a context block."""
    if not retrieved_chunks:
        return "[no retrieved context]"

    blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        blocks.append(f"[Chunk {idx}]\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(
    query_text: str,
    retrieved_chunks: list[Chunk],
    additional_instructions: str | None = None,
    template: str | None = None,
) -> str:
    """Build the user prompt from query, context, and optional instructions."""
    context = format_context(retrieved_chunks)
    instructions_block = ""
    if additional_instructions:
        instructions_block = f"\n\nAdditional instructions: {additional_instructions.strip()}"

    prompt_template = template or DEFAULT_USER_PROMPT_TEMPLATE
    return prompt_template.format(
        context=context,
        question=query_text,
        instructions_block=instructions_block,
    )

