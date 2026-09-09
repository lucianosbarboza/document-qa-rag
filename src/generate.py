"""
Calls Claude to answer a question using only the retrieved chunks, and
requires it to cite which excerpt each claim came from. This is what
turns "an LLM that might make things up" into "a Q&A system whose
answers you can actually verify against the source PDFs."
"""
import os

from anthropic import Anthropic

from .chunking import Chunk

_client: Anthropic | None = None

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a careful assistant that answers questions using ONLY the \
provided context excerpts.

Rules:
- If the answer isn't contained in the context, say you don't know. Never guess \
or use outside knowledge.
- After every claim, cite the excerpt number(s) it came from, like [1] or [2][3].
- Keep answers concise and address the question directly."""


def get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def build_context(chunks: list[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {c.source}, page {c.page})\n{c.text}")
    return "\n\n".join(parts)


def answer_question(question: str, chunks: list[Chunk]) -> str:
    context = build_context(chunks)
    message = get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context excerpts:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return message.content[0].text
