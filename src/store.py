"""
Persistent vector store, backed by Chroma -- an embedded vector
database (no server to run; it writes to a plain directory on disk),
which keeps the "works offline, nothing to host" property we picked
local embeddings for in the first place.

This replaces the original pickle file, but keeps the same seam the
README always pointed at: `save_index` writes an index, `load_index`
reads it back, and nothing outside this module needs to know how.
What *did* change is what `load_index` hands back -- not a raw numpy
embeddings array the caller had to search manually, but a `Collection`
that can run its own approximate-nearest-neighbor search. See
src/retrieval.py for how that changes the hybrid search.
"""
from pathlib import Path

import chromadb

from .chunking import Chunk

DB_PATH = Path(__file__).resolve().parent.parent / "index" / "chroma"
COLLECTION_NAME = "chunks"


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(DB_PATH))


def save_index(chunks: list[Chunk], embeddings) -> None:
    """(Re)builds the collection from scratch with the given chunks."""
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)  # start clean on re-ingest
    except Exception:
        pass  # first run: nothing to delete yet

    # hnsw:space="cosine" matches src/embeddings.py, which already
    # L2-normalizes its vectors -- cosine and dot-product agree there,
    # but naming the space explicitly keeps the intent readable.
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[str(c.chunk_id) for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c.indexed_text for c in chunks],
        metadatas=[
            {"text": c.text, "source": c.source, "page": c.page, "context": c.context}
            for c in chunks
        ],
    )


def load_index() -> tuple[list[Chunk], "chromadb.Collection"]:
    """Returns (chunks, collection). `chunks` is needed in full to build
    BM25 -- its IDF statistics are a global corpus property, so there's
    no way around loading every chunk's text for that half of hybrid
    search, vector DB or not. `collection` is what the dense half
    queries against, instead of a numpy array held in memory."""
    not_found = FileNotFoundError(
        f"No index found at {DB_PATH}. Add PDFs to data/ and run `python ingest.py` first."
    )
    if not DB_PATH.exists():
        raise not_found
    try:
        collection = _client().get_collection(COLLECTION_NAME)
    except Exception:
        raise not_found
    if collection.count() == 0:
        raise not_found

    result = collection.get(include=["metadatas"])
    chunks = [
        Chunk(
            text=meta["text"],
            source=meta["source"],
            page=meta["page"],
            chunk_id=int(chunk_id),
            context=meta.get("context", ""),
        )
        for chunk_id, meta in zip(result["ids"], result["metadatas"])
    ]
    chunks.sort(key=lambda c: c.chunk_id)  # Chroma doesn't guarantee get() order
    return chunks, collection
