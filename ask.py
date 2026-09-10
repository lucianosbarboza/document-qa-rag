"""
Command-line Q&A loop: retrieves the most relevant chunks for your
question via hybrid (BM25 + dense) search, then asks Claude to answer
using only those chunks, with citations back to source/page.

Usage:
    python ask.py
"""
import sys

from dotenv import load_dotenv

from src.generate import answer_question
from src.retrieval import HybridRetriever
from src.store import load_index

# Windows consoles often default to a legacy codepage (e.g. cp1252)
# instead of UTF-8, which mangles characters like em dashes or accents
# coming back from the model (they show up as "?" or "�"). Force UTF-8
# on stdout/stdin regardless of how the terminal was launched.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8")

load_dotenv()


def main() -> None:
    chunks, collection = load_index()
    retriever = HybridRetriever(chunks, collection)

    print(f"Loaded index with {len(chunks)} chunks. Ask a question (or 'quit').\n")

    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        results = retriever.search(question, top_k=5)
        top_chunks = [c for c, _ in results]

        print("\nRetrieved from:")
        for c, score in results:
            print(f"  [{c.chunk_id}] {c.source} p.{c.page} (score={score:.3f})")

        answer = answer_question(question, top_chunks)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
