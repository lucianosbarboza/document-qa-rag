"""
Ingestion pipeline: reads every PDF in data/, splits each into
overlapping chunks, embeds them, and saves the resulting index to disk.

Run this once, and again whenever you add or change PDFs:
    python ingest.py
"""
from pathlib import Path

from pypdf import PdfReader

from src.chunking import chunk_text, Chunk
from src.embeddings import embed
from src.store import save_index

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_pdf_chunks(pdf_path: Path) -> list[Chunk]:
    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []
    chunk_id = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_chunks = chunk_text(
            text, source=pdf_path.name, page=page_num, chunk_id_start=chunk_id
        )
        chunks.extend(page_chunks)
        chunk_id += len(page_chunks)

    return chunks


def main() -> None:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}. Add some and re-run.")
        return

    all_chunks: list[Chunk] = []
    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name}...")
        pdf_chunks = load_pdf_chunks(pdf_path)
        print(f"  -> {len(pdf_chunks)} chunks")
        all_chunks.extend(pdf_chunks)

    if not all_chunks:
        print("No extractable text found in the PDFs (are they scanned images?).")
        return

    print(f"\nEmbedding {len(all_chunks)} chunks (first run downloads the model)...")
    embeddings = embed([c.text for c in all_chunks])

    save_index(all_chunks, embeddings)
    print(f"Done. Index saved to index/index.pkl ({len(all_chunks)} chunks).")


if __name__ == "__main__":
    main()
