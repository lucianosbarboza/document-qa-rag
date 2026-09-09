"""
A from-scratch implementation of BM25 (the Okapi variant) -- a classic
sparse (keyword-based) retrieval scoring function, still a core piece of
most production hybrid-search systems.

Intuition: a chunk scores high for a query when it contains the query's
terms (term frequency), those terms are rare across the whole corpus so
they're distinctive (inverse document frequency), and the chunk isn't
artificially winning just for being long (length normalization).
"""
import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Good enough for BM25; a real system
    might add stemming/stopword removal."""
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        # k1 controls how quickly term-frequency saturates (diminishing
        # returns for a term appearing many times); b controls how much
        # document length is penalized.
        self.k1 = k1
        self.b = b

        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.n_docs = len(documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs else 0
        )
        self.doc_term_freqs = [Counter(doc) for doc in self.tokenized_docs]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        doc_freq = Counter()
        for doc in self.tokenized_docs:
            for term in set(doc):
                doc_freq[term] += 1

        idf = {}
        for term, freq in doc_freq.items():
            # +1 inside the log keeps idf non-negative even for terms
            # that appear in more than half the corpus.
            idf[term] = math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
        return idf

    def score(self, query: str) -> list[float]:
        """Return a BM25 score for every document against `query`."""
        query_terms = tokenize(query)
        scores = [0.0] * self.n_docs

        for i in range(self.n_docs):
            doc_len = self.doc_lengths[i]
            term_freqs = self.doc_term_freqs[i]
            for term in query_terms:
                f = term_freqs.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_length
                )
                scores[i] += idf * numerator / denominator

        return scores
