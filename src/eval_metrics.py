"""
RAGAS-style evaluation metrics, implemented from scratch (no `ragas`
dependency, consistent with the rest of this project).

RAGAS itself works by using an LLM as a judge to score answers along
several dimensions, plus embedding similarity for one metric. That's
exactly what we do here with Claude (already used for generation) and
the local embedding model (already used for retrieval) -- no new
dependencies needed.

Four metrics, mirroring RAGAS's core set:

- faithfulness: of the claims made in the answer, what fraction are
  actually supported by the retrieved context? Catches hallucination.
- answer_relevancy: does the answer actually address the question
  asked? Measured indirectly: generate questions the answer *would*
  answer, then compare them to the real question via embeddings.
- context_precision: of the retrieved chunks, what fraction are
  actually relevant to answering the question? (needs a reference
  answer to judge relevance against)
- context_recall: does the retrieved context contain everything needed
  to reconstruct the reference answer? Catches retrieval gaps.

All four return a float in [0, 1] (higher is better).
"""
import json
import re

from .chunking import Chunk
from .embeddings import embed
from .generate import get_client, MODEL


def _call_json(system: str, user: str):
    """Call Claude and parse its reply as JSON, tolerating stray code
    fences or prose around the JSON object/array."""
    message = get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system + "\n\nRespond with ONLY valid JSON. No prose, no code fences.",
        messages=[{"role": "user", "content": user}],
    )
    raw = next(block.text for block in message.content if block.type == "text").strip()
    match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
    return json.loads(match.group(0) if match else raw)


def _build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))


def _as_bool(item) -> bool:
    """Coerce a judge's per-item verdict to a bool, tolerating a bare
    true/false, a {"relevant": true}-style dict, or a "true"/"yes" string."""
    if isinstance(item, bool):
        return item
    if isinstance(item, dict):
        return any(v is True for v in item.values())
    if isinstance(item, str):
        return item.strip().lower() in {"true", "yes"}
    return False


def _fraction_true(items: list, key: str) -> float:
    """Fraction of items judged true on `key`. Tolerates the judge
    occasionally not following the requested {"...": ..., key: bool}
    schema (e.g. returning bare strings) by counting those as false
    instead of crashing the whole eval run."""
    if not items:
        return 0.0
    true_count = sum(1 for item in items if isinstance(item, dict) and item.get(key))
    return true_count / len(items)


def faithfulness(answer: str, chunks: list[Chunk]) -> float:
    """Fraction of the answer's claims that are supported by the context."""
    context = _build_context(chunks)
    result = _call_json(
        system="You extract atomic factual claims from an answer and judge "
        "whether each is supported by the given context excerpts.",
        user=(
            f"Context excerpts:\n{context}\n\n"
            f"Answer to check:\n{answer}\n\n"
            'Break the answer into atomic factual claims, then judge each one. '
            'Return a JSON array of objects, each with a "claim" string field and a '
            '"supported" boolean field -- never a bare string. Example: '
            '[{"claim": "the sky is blue", "supported": true}, '
            '{"claim": "the sky is green", "supported": false}]. '
            "If the answer makes no checkable claims (e.g. \"I don't know\"), return []."
        ),
    )
    if not result:
        return 1.0  # nothing to hallucinate
    return _fraction_true(result, "supported")


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """How well the answer addresses the question, via round-trip
    question generation + embedding similarity (no reference needed)."""
    generated = _call_json(
        system="You generate the questions that a given answer would be a "
        "good response to.",
        user=(
            f"Answer:\n{answer}\n\n"
            f'Generate exactly {n} distinct questions this answer would directly answer. '
            'Return a JSON array of strings, e.g. ["question 1", "question 2", ...].'
        ),
    )
    if not generated:
        return 0.0
    vectors = embed([question] + list(generated))
    question_vec, generated_vecs = vectors[0], vectors[1:]
    similarities = generated_vecs @ question_vec  # cosine sim (embeddings are normalized)
    return float(similarities.mean())


def context_precision(question: str, reference_answer: str, chunks: list[Chunk]) -> float:
    """Precision@k over the retrieved chunks, weighted by rank (like
    RAGAS): relevant chunks near the top count more than ones near the
    bottom."""
    context = _build_context(chunks)
    relevance = _call_json(
        system="You judge whether retrieved excerpts are relevant to "
        "answering a question, given a reference answer.",
        user=(
            f"Question: {question}\n"
            f"Reference answer: {reference_answer}\n\n"
            f"Retrieved excerpts:\n{context}\n\n"
            'For each excerpt, is it relevant to producing the reference answer? '
            'Return a JSON array of booleans in excerpt order, e.g. [true, false, ...].'
        ),
    )
    if not relevance:
        return 0.0
    verdicts = [_as_bool(r) for r in relevance]
    precisions_at_k = []
    relevant_so_far = 0
    for k, is_relevant in enumerate(verdicts, start=1):
        if is_relevant:
            relevant_so_far += 1
            precisions_at_k.append(relevant_so_far / k)
    if not precisions_at_k:
        return 0.0
    return sum(precisions_at_k) / sum(verdicts)


def context_recall(reference_answer: str, chunks: list[Chunk]) -> float:
    """Fraction of the reference answer's statements that can be
    attributed to the retrieved context. Low recall means the retriever
    missed information needed to answer well."""
    context = _build_context(chunks)
    statements = _call_json(
        system="You break a reference answer into individual statements "
        "and judge whether each can be attributed to the given context.",
        user=(
            f"Context excerpts:\n{context}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            'Break the reference answer into individual statements, then judge each. '
            'Return a JSON array of objects, each with a "statement" string field and an '
            '"attributed" boolean field -- never a bare string. Example: '
            '[{"statement": "the sky is blue", "attributed": true}, '
            '{"statement": "the sky is green", "attributed": false}].'
        ),
    )
    if not statements:
        return 0.0
    return _fraction_true(statements, "attributed")
