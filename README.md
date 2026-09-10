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
 save chunks + embeddings to Chroma (src/store.py)  <-- python ingest.py


Your question
      |
      v
 BM25 keyword score, in memory (src/bm25.py)      ---\
                                                        +--> combined hybrid ranking (src/retrieval.py)
 dense nearest-neighbor search via Chroma           ---/
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

**Why Chroma?** The index started as a pickle file (fine for a handful
of PDFs, but everything had to fit in memory and every query re-scored
every chunk by hand). Chroma is an embedded vector database -- no
server to run, it just writes to `index/chroma/` -- so it keeps the
"nothing to host, works offline" property the local embedding model was
chosen for, while doing what a vector DB is actually for: real
approximate-nearest-neighbor search, so the dense half of retrieval
doesn't need every embedding held in Python memory to answer one query.
BM25 still needs the whole corpus in memory regardless -- its IDF
statistics are a global property of the corpus, vector DB or not -- so
`src/bm25.py` is unchanged. Hybrid search now works as a two-stage
fusion: BM25 and Chroma each nominate their own top candidates, and only
that combined pool gets re-scored together (`src/retrieval.py`) -- the
standard pattern for combining a keyword and a vector index, rather
than the original's brute-force score-everything approach.

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

## Contextual chunking (optional)

Plain chunking throws away surrounding context: a chunk that says "The
success rate was 87%" is nearly impossible for BM25 or an embedding
model to match to a query like "implant survival rate", since it never
says *what* had an 87% success rate.

Contextual chunking (the technique Anthropic calls [Contextual
Retrieval](https://www.anthropic.com/news/contextual-retrieval)) asks
Claude to write a short 1-2 sentence blurb situating each chunk within
its full document, and prepends that blurb only to the copy of the
chunk that gets indexed (BM25 + embeddings) -- the original chunk text
and citations shown to the user are untouched (`src/contextualize.py`,
`Chunk.indexed_text`). The whole document is sent as a cached prompt
prefix, so the cost is one Claude call per chunk, most of it billed at
the (much cheaper) cache-read rate rather than full price each time.

It's opt-in because of that extra cost/latency:

```bash
python ingest.py --contextual
```

**Measured result on this repo's test corpus:** running `eval.py`
before and after contextual chunking (`eval/last_run.json` vs.
`eval/last_run_contextual.json`) showed no clear improvement here --
context_precision and context_recall were actually slightly lower,
answer_relevancy slightly higher, all within the noise of an 8-question,
single-document eval set. That tracks: contextual chunking mainly helps
when chunks are ambiguous *relative to other chunks* -- across many
documents, or many sections covering similar topics. A single 23-page
paper on one subject doesn't have much of that ambiguity to fix. Worth
re-measuring if you point this at a larger, more heterogeneous corpus.

## Evaluation

Instead of eyeballing answer quality, `eval.py` runs the full pipeline
against a fixed set of Q&A pairs (`eval/dataset.json`) and scores each
answer with four RAGAS-style metrics, implemented from scratch using
Claude as an LLM judge (plus the local embedding model for one of them):

- **faithfulness** -- of the claims in the answer, what fraction are
  actually supported by the retrieved context? (catches hallucination)
- **answer_relevancy** -- does the answer address the question asked?
  (generates candidate questions from the answer, compares them to the
  real question via embedding similarity)
- **context_precision** -- of the retrieved chunks, what fraction are
  actually relevant? (rank-weighted, like average precision)
- **context_recall** -- does the retrieved context contain everything
  needed to reconstruct the reference answer? (catches retrieval gaps)

```bash
python eval.py                                    # run eval/dataset.json
python eval.py --dataset eval/my_dataset.json --top-k 3 --out eval/results.json
```

Each question needs a `reference_answer` in the dataset (used to judge
context precision/recall) -- see `eval/dataset.json` for the format.
`eval/last_run.json` is a sample run against this repo's test PDF.

This turns "did I make the retriever better?" from a guess into a
number you can compare across chunk sizes, alpha values, or a future
re-ranker.

## Project layout

```
document-qa-rag/
├── data/            # put your PDFs here (gitignored)
├── index/           # generated index (gitignored)
├── eval/
│   ├── dataset.json            # Q&A pairs with reference answers, for eval.py
│   ├── last_run.json           # sample eval.py output (plain chunking)
│   └── last_run_contextual.json # sample eval.py output (--contextual)
├── src/
│   ├── chunking.py       # splits page text into overlapping chunks
│   ├── contextualize.py  # optional: LLM-generated per-chunk context (ingest.py --contextual)
│   ├── bm25.py            # BM25 keyword scoring, implemented from scratch
│   ├── embeddings.py      # local dense embeddings (sentence-transformers)
│   ├── retrieval.py       # fuses BM25 candidates + Chroma's dense search into one ranking
│   ├── store.py           # saves/loads the Chroma-backed index
│   ├── generate.py        # calls Claude with retrieved context + citation rules
│   └── eval_metrics.py    # RAGAS-style eval metrics (LLM-as-judge, from scratch)
├── ingest.py        # build the index from data/*.pdf
├── ask.py           # interactive Q&A CLI
└── eval.py          # scores answer quality against eval/dataset.json
```

## What this teaches (mapped to the pipeline)

- **Chunking strategy** and why overlap matters
- **Sparse retrieval**: BM25 scoring implemented by hand (term frequency,
  IDF, length normalization)
- **Dense retrieval**: semantic embeddings + a real vector database
  (Chroma) doing approximate-nearest-neighbor search
- **Hybrid search**: fusing two independent candidate pools (BM25's and
  the vector DB's) and normalizing/blending their scores
- **Grounded generation**: prompting an LLM to answer *only* from
  provided context and cite its sources, to reduce hallucination

## Ideas to extend (next difficulty tier)

- ~~Swap the pickle-based store for a real vector DB (Chroma, Qdrant)~~ --
  done: Chroma, embedded (no server), see `src/store.py`
- Add re-ranking of the top-k results with a cross-encoder
- ~~Try contextual chunking (prepend a short LLM-generated summary of the
  surrounding document to each chunk before embedding)~~ -- done, see
  [Contextual chunking](#contextual-chunking-optional) above (though it
  didn't measurably help on this small single-document corpus)
- ~~Add an eval harness (e.g. a small set of Q&A pairs + RAGAS) to
  measure answer quality instead of eyeballing it~~ -- done, see
  [Evaluation](#evaluation) above
- Support Slack/Notion as additional sources (as in the original project
  brief), not just PDF
