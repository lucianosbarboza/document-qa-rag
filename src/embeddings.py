"""
Produces dense (semantic) embeddings for chunks and queries, using a
local, open model: `all-MiniLM-L6-v2`.

We embed locally instead of calling a paid API because Anthropic's API
does not currently offer an embeddings endpoint -- Claude is used later,
purely for the generation step. A nice side effect: indexing your
documents costs nothing and works offline.

**Why the ONNX runtime and not sentence-transformers?** This module
used to load the model through `sentence-transformers`, which pulls in
PyTorch (~550MB installed, plus a large resident footprint just to
import it). That is fine on a laptop, but it makes the app impossible
to host on a small free-tier instance: the first deploy to Render died
with `Out of memory (used over 512Mi)` before uvicorn could even bind
its port, because importing this module imported torch.

Chroma (already a dependency -- see src/store.py) ships the *same*
`all-MiniLM-L6-v2` weights exported to ONNX and runs them on
`onnxruntime`, which Chroma already installs and which is ~45MB
rather than ~670MB. Same model, same 384-dimension normalized output,
a fraction of the memory.

Measured, rather than assumed: embedding all 68 chunks of the test
corpus both ways gives a cosine similarity of 1.0000 between the two
models' vectors (mean *and* minimum), and the retriever returns the
identical top-5 chunks for all 8 questions in eval/dataset.json. The
two paths are numerically equivalent for retrieval purposes. They
aren't guaranteed bit-identical though (different kernels round
differently), so rebuild the index with `python ingest.py` after
switching rather than querying an old index with a new model.
"""
import numpy as np
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

_model: ONNXMiniLM_L6_V2 | None = None


def get_model() -> ONNXMiniLM_L6_V2:
    global _model
    if _model is None:
        # Downloads (~80MB) the first time it's used, then caches under
        # ~/.cache/chroma/onnx_models/. On a deploy, `python ingest.py`
        # runs as a build step, so that download happens at build time
        # and the cache is already warm when the server starts.
        _model = ONNXMiniLM_L6_V2()
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into normalized vectors (so a dot product
    between two embeddings equals their cosine similarity)."""
    model = get_model()
    return np.asarray(model(texts), dtype=np.float32)
