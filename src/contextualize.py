"""
Contextual chunking (Anthropic's "Contextual Retrieval" technique):
before indexing, each chunk gets a short LLM-generated blurb prepended
that situates it within the full document.

Why bother? Plain chunking throws away surrounding context. A chunk
that says "The success rate was 87%" is nearly impossible for BM25 or
an embedding model to match to a query like "implant survival rate",
because it never says *what* had an 87% success rate. Prepending "This
chunk is from the results section discussing implant survival --" fixes
that, without changing what's shown to the user: the chunk's original
`text` (and citations) are untouched, only the retrieval-time text
(`Chunk.indexed_text`) changes.

Cost note: this is one Claude call per chunk. The full document is sent
as a cached prompt prefix (`cache_control`), so you pay full price for
it once per source file and a small fraction of that for every
following chunk that reuses the cache -- still real cost and latency
though, which is why ingest.py makes this opt-in (`--contextual`).
"""
from .chunking import Chunk
from .generate import get_client, MODEL

SYSTEM_PROMPT = "You situate a chunk of text within its full source document, briefly and precisely."

# Anthropic's own Contextual Retrieval writeup recommends this prompt shape.
_USER_TEMPLATE = """\
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Give a short, succinct context (1-2 sentences) to situate this chunk \
within the overall document, for the purpose of improving search \
retrieval of the chunk. Answer only with the context, nothing else."""


def _context_for_chunk(document_text: str, chunk_text: str) -> str:
    message = get_client().messages.create(
        model=MODEL,
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>\n{document_text}\n</document>",
                        # Cached so re-sending the (large, unchanging) full
                        # document for every chunk of the same file is cheap.
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": _USER_TEMPLATE.format(chunk_text=chunk_text),
                    },
                ],
            }
        ],
    )
    return next(block.text for block in message.content if block.type == "text").strip()


def contextualize_chunks(chunks: list[Chunk], document_text_by_source: dict[str, str]) -> None:
    """Fills in `chunk.context` for every chunk, in place."""
    for i, chunk in enumerate(chunks, start=1):
        document_text = document_text_by_source[chunk.source]
        chunk.context = _context_for_chunk(document_text, chunk.text)
        print(f"  [{i}/{len(chunks)}] contextualized chunk {chunk.chunk_id} ({chunk.source})")
