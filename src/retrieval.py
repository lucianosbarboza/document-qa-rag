"""
Hybrid retrieval: combines BM25 (sparse/keyword) and dense (embedding)
scores into a single ranking.

Why hybrid? BM25 nails exact matches -- names, IDs, jargon ("invoice
#4471") -- that embeddings tend to blur together. Dense embeddings catch
semantic matches -- "cancellation policy" ~ "how do I get a refund" --
that pure keyword search misses entirely. Combining both is a common,
effective pattern in real-world retrieval systems.
"""
import numpy as np

from .bm25 import BM25
from .chunking import Chunk
from .embeddings import embed


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1] so BM25 scores (unbounded) and cosine
    similarities (already in [-1, 1]) can be combined fairly."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, alpha: float = 0.5):
        """
        alpha: weight given to the dense score vs. the BM25 score.
               1.0 = pure dense search, 0.0 = pure BM25. 0.5 is a
               reasonable starting point -- tune it and see how results
               shift for keyword-heavy vs. conceptual questions.
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.alpha = alpha
        # indexed_text == text unless contextual chunking (src/contextualize.py)
        # filled in chunk.context -- this stays a no-op for plain chunks.
        self.bm25 = BM25([c.indexed_text for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        bm25_scores = np.array(self.bm25.score(query))

        query_embedding = embed([query])[0]
        dense_scores = self.embeddings @ query_embedding  # cosine sim (vectors are normalized)

        combined = self.alpha * _normalize(dense_scores) + (1 - self.alpha) * _normalize(bm25_scores)

        top_indices = np.argsort(combined)[::-1][:top_k]
        return [(self.chunks[i], float(combined[i])) for i in top_indices]
