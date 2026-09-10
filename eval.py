"""
RAGAS-style evaluation harness: runs the full retrieve-then-generate
pipeline against a fixed set of Q&A pairs (eval/dataset.json) and scores
each answer on four axes -- faithfulness, answer relevancy, context
precision, and context recall -- using Claude as an LLM judge.

This replaces "eyeballing" answer quality with a repeatable measurement,
so you can tell whether a change (a different chunk size, a different
alpha, a re-ranker, ...) actually made the system better or worse.

Usage:
    python eval.py
    python eval.py --dataset eval/my_other_dataset.json --top-k 3

Note: this makes several Claude API calls per question (one for
generation, plus one judge call per metric), so it costs more and runs
slower than a single ask.py query -- expect ~4-5 calls per row.
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.eval_metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from src.generate import answer_question
from src.retrieval import HybridRetriever
from src.store import load_index

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

DEFAULT_DATASET = Path(__file__).resolve().parent / "eval" / "dataset.json"


def evaluate_row(retriever: HybridRetriever, question: str, reference_answer: str, top_k: int) -> dict:
    results = retriever.search(question, top_k=top_k)
    chunks = [c for c, _ in results]
    answer = answer_question(question, chunks)

    return {
        "question": question,
        "answer": answer,
        "reference_answer": reference_answer,
        "faithfulness": round(faithfulness(answer, chunks), 3),
        "answer_relevancy": round(answer_relevancy(question, answer), 3),
        "context_precision": round(context_precision(question, reference_answer, chunks), 3),
        "context_recall": round(context_recall(reference_answer, chunks), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None, help="optional path to save results as JSON")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    chunks, collection = load_index()
    retriever = HybridRetriever(chunks, collection)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    rows = []
    print(f"Evaluating {len(dataset)} questions from {args.dataset}...\n")

    for i, item in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] {item['question']}")
        row = evaluate_row(retriever, item["question"], item["reference_answer"], args.top_k)
        rows.append(row)
        print("  " + "  ".join(f"{m}={row[m]:.2f}" for m in metric_names) + "\n")

    print("=" * 60)
    print("Averages across all questions:")
    for m in metric_names:
        avg = sum(r[m] for r in rows) / len(rows)
        print(f"  {m:>18}: {avg:.3f}")

    if args.out:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved detailed results to {args.out}")


if __name__ == "__main__":
    main()
