"""
A minimal on-disk index: a pickle file holding the chunks (with their
source/page metadata) plus their embeddings as a numpy array.

This is intentionally simple -- good enough for a learning project or a
handful of PDFs. Once you outgrow it, swap this module for a real vector
database (Chroma, Qdrant, pgvector, ...) without touching the rest of
the pipeline: `save_index`/`load_index` is the only seam that needs to
change.
"""
import pickle
from pathlib import Path

import numpy as np

from .chunking import Chunk

INDEX_PATH = Path(__file__).resolve().parent.parent / "index" / "index.pkl"


def save_index(chunks: list[Chunk], embeddings: np.ndarray) -> None:
    INDEX_PATH.parent.mkdir(exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)


def load_index() -> tuple[list[Chunk], np.ndarray]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Add PDFs to data/ and run "
            "`python ingest.py` first."
        )
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"]
