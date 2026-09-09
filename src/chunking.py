"""
Splits document text into overlapping chunks for retrieval.

Why chunk at all? Embedding models and LLM context windows work best over
small, focused pieces of text rather than whole documents -- a 40-page PDF
embedded as one vector would blur together dozens of unrelated topics.
Overlap between consecutive chunks avoids losing context for sentences
that happen to fall right on a chunk boundary.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str   # filename the chunk came from
    page: int     # 1-indexed page number (for citations)
    chunk_id: int  # position within the whole corpus


def chunk_text(
    text: str,
    source: str,
    page: int,
    chunk_id_start: int,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    """Split `text` into overlapping chunks of ~chunk_size characters.

    A character-based (not token-based) size is used to keep this
    dependency-free and easy to reason about; ~1000 characters is roughly
    150-200 words, a reasonable unit for retrieval.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = chunk_id_start

    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, page=page, chunk_id=chunk_id))
            chunk_id += 1
        if end >= len(text):
            break
        start = end - overlap  # step back so chunks overlap

    return chunks
