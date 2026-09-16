"""
FastAPI backend for document-qa-rag: exposes the same hybrid retrieval +
Claude generation pipeline as ask.py, over HTTP, plus a static front end
(static/) so it can be demoed in a browser instead of a terminal.

Run locally:
    uvicorn api:app --reload

Deploy: see Procfile / render.yaml -- both run
`python ingest.py` (rebuild the index) as a build step, then start uvicorn.
"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.generate import answer_question
from src.retrieval import HybridRetriever
from src.store import load_index

load_dotenv()

app = FastAPI(title="Document Q&A RAG")

_retriever: HybridRetriever | None = None


@app.on_event("startup")
def _load_retriever() -> None:
    global _retriever
    chunks, collection = load_index()
    _retriever = HybridRetriever(chunks, collection)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    results = _retriever.search(question, top_k=req.top_k)
    answer = answer_question(question, [c for c, _ in results])
    sources = [Source(source=c.source, page=c.page, score=score) for c, score in results]
    return AskResponse(answer=answer, sources=sources)


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
