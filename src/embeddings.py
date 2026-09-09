"""
Wraps a local sentence-transformers model to produce dense (semantic)
embeddings.

We use a local, open model instead of a paid API because Anthropic's API
does not currently offer an embeddings endpoint -- Claude is used later,
purely for the generation step. A nice side effect: indexing your
documents costs nothing and works offline.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for learning/most Q&A
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Downloads (~80MB) the first time it's used, then caches locally.
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into normalized vectors (so a dot product
    between two embeddings equals their cosine similarity)."""
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
