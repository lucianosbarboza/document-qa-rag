"""
Hybrid retrieval: combines BM25 (sparse/keyword) and dense (embedding)
scores into a single ranking.

Why hybrid? BM25 nails exact matches -- names, IDs, jargon ("invoice
#4471") -- that embeddings tend to blur together. Dense embeddings catch
semantic matches -- "cancellation policy" ~ "how do I get a refund" --
that pure keyword search misses entirely. Combining both is a common,
effective pattern in real-world retrieval systems.

Dense search is delegated to Chroma (src/store.py) instead of a manual
`embeddings @ query_embedding` matrix multiply -- that's the actual job
of a vector database: approximate-nearest-neighbor search without
holding every embedding in Python memory to answer one query. BM25
still needs the whole corpus in memory regardless (its IDF statistics
are a global property of the corpus), so that half is unchanged.
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
    def __init__(self, chunks: list[Chunk], collection, alpha: float = 0.5):
        """
        alpha: weight given to the dense score vs. the BM25 score.
               1.0 = pure dense search, 0.0 = pure BM25. 0.5 is a
               reasonable starting point -- tune it and see how results
               shift for keyword-heavy vs. conceptual questions.
        """
        self.chunks_by_id = {str(c.chunk_id): c for c in chunks}
        self.collection = collection
        self.alpha = alpha
        self.bm25 = BM25([c.indexed_text for c in chunks])
        self._bm25_ids = [str(c.chunk_id) for c in chunks]  # same order BM25 scored in

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        # Over-fetch beyond top_k from each method: this is the standard
        # hybrid-search pattern -- BM25 and Chroma each nominate a pool
        # of candidates, and only that union gets re-scored together.
        # (Scoring the *entire* corpus this way, like a single in-memory
        # array let us do before, stops scaling once the vector DB is
        # doing its actual job of not loading everything into memory.)
        pool_size = max(top_k * 4, 20)

        bm25_scores = dict(zip(self._bm25_ids, self.bm25.score(query)))
        bm25_candidates = sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:pool_size]

        query_embedding = embed([query])[0]
        dense_result = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=pool_size
        )
        dense_ids = dense_result["ids"][0]
        # Chroma's cosine space returns a distance (0 = identical); flip
        # it back to a similarity so higher is better, like BM25's score.
        dense_scores = {i: 1.0 - d for i, d in zip(dense_ids, dense_result["distances"][0])}

        candidate_ids = list(set(bm25_candidates) | set(dense_ids))
        bm25_arr = np.array([bm25_scores.get(i, 0.0) for i in candidate_ids])
        dense_arr = np.array([dense_scores.get(i, 0.0) for i in candidate_ids])

        combined = self.alpha * _normalize(dense_arr) + (1 - self.alpha) * _normalize(bm25_arr)
        order = np.argsort(combined)[::-1][:top_k]
        return [(self.chunks_by_id[candidate_ids[i]], float(combined[i])) for i in order]
