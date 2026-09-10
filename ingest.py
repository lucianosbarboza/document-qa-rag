"""
Ingestion pipeline: reads every PDF in data/, splits each into
overlapping chunks, embeds them, and saves the resulting index to disk.

Run this once, and again whenever you add or change PDFs:
    python ingest.py

Add --contextual to enable contextual chunking (see
src/contextualize.py): each chunk gets a short Claude-generated blurb
situating it within its document before being indexed, which tends to
improve retrieval for chunks that read ambiguously on their own. It
costs one extra Claude call per chunk, so it's opt-in:
    python ingest.py --contextual
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

from src.chunking import chunk_text, Chunk
from src.embeddings import embed
from src.store import save_index

DATA_DIR = Path(__file__).resolve().parent / "data"

load_dotenv()  # only needed for --contextual, but harmless otherwise


def load_pdf_chunks(pdf_path: Path) -> tuple[list[Chunk], str]:
    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []
    chunk_id = 0
    pages_text = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages_text.append(text)
        page_chunks = chunk_text(
            text, source=pdf_path.name, page=page_num, chunk_id_start=chunk_id
        )
        chunks.extend(page_chunks)
        chunk_id += len(page_chunks)

    return chunks, "\n\n".join(pages_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contextual",
        action="store_true",
        help="prepend an LLM-generated context blurb to each chunk before indexing",
    )
    args = parser.parse_args()

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}. Add some and re-run.")
        return

    all_chunks: list[Chunk] = []
    document_text_by_source: dict[str, str] = {}
    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name}...")
        pdf_chunks, full_text = load_pdf_chunks(pdf_path)
        print(f"  -> {len(pdf_chunks)} chunks")
        all_chunks.extend(pdf_chunks)
        document_text_by_source[pdf_path.name] = full_text

    if not all_chunks:
        print("No extractable text found in the PDFs (are they scanned images?).")
        return

    if args.contextual:
        from src.contextualize import contextualize_chunks

        print(f"\nContextualizing {len(all_chunks)} chunks (1 Claude call each)...")
        contextualize_chunks(all_chunks, document_text_by_source)

    print(f"\nEmbedding {len(all_chunks)} chunks (first run downloads the model)...")
    embeddings = embed([c.indexed_text for c in all_chunks])

    save_index(all_chunks, embeddings)
    print(f"Done. Index saved to index/chroma/ ({len(all_chunks)} chunks).")


if __name__ == "__main__":
    main()
