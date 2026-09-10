"""
Command-line Q&A: retrieves the most relevant chunks for your question via
hybrid (BM25 + dense) search, then asks Claude to answer using only those
chunks, with citations back to source/page.

Usage:
    python ask.py                  interactive loop (ask multiple questions)
    python ask.py "your question"  one-shot mode: answer just this question,
                                    print the answer alone, then exit -- meant
                                    for other programs to call (e.g. as a tool
                                    from an agent project), not just humans.
"""
import argparse
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


def ask(retriever: HybridRetriever, question: str, top_k: int = 5, verbose: bool = True) -> str:
    """Retrieve + generate for a single question. Shared by both interactive
    and one-shot mode, and importable by other projects that just want the
    answer (see clinic-agent's search_implant_literature tool)."""
    results = retriever.search(question, top_k=top_k)
    top_chunks = [c for c, _ in results]

    if verbose:
        print("\nRetrieved from:")
        for c, score in results:
            print(f"  [{c.chunk_id}] {c.source} p.{c.page} (score={score:.3f})")

    return answer_question(question, top_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", default=None, help="ask one question and exit (omit for interactive mode)")
    args = parser.parse_args()

    chunks, collection = load_index()
    retriever = HybridRetriever(chunks, collection)

    if args.question:
        print(ask(retriever, args.question, verbose=False))
        return

    print(f"Loaded index with {len(chunks)} chunks. Ask a question (or 'quit').\n")
    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        print(f"\n{ask(retriever, question)}\n")


if __name__ == "__main__":
    main()
