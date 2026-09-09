# Document Q&A RAG

A from-scratch Retrieval-Augmented Generation (RAG) pipeline that answers
questions about your own PDFs, with citations back to the exact
source/page. Built as a beginner-level portfolio project covering the
full RAG stack: chunking, hybrid retrieval (BM25 + dense embeddings),
and grounded generation.

Based on the "Document Q&A RAG" project from
[ai-engineer-portfolio-projects](https://github.com/landedjobs/ai-engineer-portfolio-projects).

## How it works

```
PDF files (data/)
      |
      v
 extract text per page (pypdf)
      |
      v
 split into overlapping chunks (src/chunking.py)
      |
      v
 embed each chunk (src/embeddings.py, local sentence-transformers model)
      |
      v
 save chunks + embeddings to disk (src/store.py)  <-- python ingest.py


Your question
      |
      v
 BM25 keyword score (src/bm25.py)  ---\
                                        +--> combined hybrid ranking (src/retrieval.py)
 dense embedding cosine score       ---/
      |
      v
 top-k chunks --> Claude, forced to cite excerpt numbers (src/generate.py)
      |
      v
 answer with citations              <-- python ask.py
```

**Why hybrid retrieval?** BM25 (keyword matching) nails exact terms --
names, IDs, jargon -- that embeddings can blur together. Dense embeddings
catch semantic matches ("cancellation policy" ~ "how do I get a refund")
that pure keyword search misses. Combining both, weighted, is what most
production RAG systems actually do.

**Why local embeddings?** Anthropic's API doesn't currently expose an
embeddings endpoint, so this project uses a small open model
(`sentence-transformers/all-MiniLM-L6-v2`) running on your machine.
Claude is used only for the generation step. This also means indexing
is free and works offline.

## Setup

```bash
cd document-qa-rag
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

copy .env.example .env       # then edit .env and add your ANTHROPIC_API_KEY
```

## Usage

1. Drop one or more PDFs into `data/`.
2. Build the index:
   ```bash
   python ingest.py
   ```
3. Ask questions:
   ```bash
   python ask.py
   ```
   Type a question, see which chunks were retrieved (with source/page
   and score), and get a Claude-generated answer that cites them, e.g.
   `The warranty period is 12 months [2].`
   Type `quit` to exit.

Re-run `ingest.py` whenever you add or change PDFs in `data/`.

> **Windows + `cmd.exe` note:** if answers show `?` or `�` instead of
> special characters (em dashes, accents), your console is using a
> legacy codepage. `ask.py` already forces UTF-8 output, but if it still
> looks wrong, run `chcp 65001` first or use Windows Terminal/PowerShell,
> which handle UTF-8 by default.

## Project layout

```
document-qa-rag/
├── data/            # put your PDFs here (gitignored)
├── index/           # generated index (gitignored)
├── src/
│   ├── chunking.py    # splits page text into overlapping chunks
│   ├── bm25.py         # BM25 keyword scoring, implemented from scratch
│   ├── embeddings.py   # local dense embeddings (sentence-transformers)
│   ├── retrieval.py    # combines BM25 + dense scores into one ranking
│   ├── store.py        # saves/loads the on-disk index
│   └── generate.py     # calls Claude with retrieved context + citation rules
├── ingest.py        # build the index from data/*.pdf
└── ask.py           # interactive Q&A CLI
```

## What this teaches (mapped to the pipeline)

- **Chunking strategy** and why overlap matters
- **Sparse retrieval**: BM25 scoring implemented by hand (term frequency,
  IDF, length normalization)
- **Dense retrieval**: semantic embeddings + cosine similarity
- **Hybrid search**: normalizing and blending two different score scales
- **Grounded generation**: prompting an LLM to answer *only* from
  provided context and cite its sources, to reduce hallucination

## Ideas to extend (next difficulty tier)

- Swap the pickle-based store for a real vector DB (Chroma, Qdrant)
- Add re-ranking of the top-k results with a cross-encoder
- Try contextual chunking (prepend a short LLM-generated summary of the
  surrounding document to each chunk before embedding)
- Add an eval harness (e.g. a small set of Q&A pairs + RAGAS) to measure
  answer quality instead of eyeballing it
- Support Slack/Notion as additional sources (as in the original project
  brief), not just PDF
