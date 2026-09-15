"""Local FastAPI server for the PaperLens research-paper RAG application."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import time
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import DocumentIndex, RAGError, answer_question, build_document_index, retrieve


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SESSION_TTL_SECONDS = 4 * 60 * 60

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="PaperLens", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@dataclass
class ActiveDocument:
    index: DocumentIndex
    created_at: float


class QuestionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2_000)
    result_count: int = Field(default=5, ge=3, le=6)


documents: dict[str, ActiveDocument] = {}


def _api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="The server has no Gemini API key. Add GEMINI_API_KEY to the local .env file.",
        )
    return key


def _remove_expired_documents() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    for session_id, document in list(documents.items()):
        if document.created_at < cutoff:
            documents.pop(session_id, None)


def _rag_error(error: RAGError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(error))


@app.get("/", include_in_schema=False)
async def homepage() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, bool]:
    return {"ok": True, "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip())}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str | int]:
    """Index one document in memory and return an opaque browser session id."""
    _remove_expired_documents()
    key = _api_key()
    filename = file.filename or "untitled-document"
    data = await file.read()
    try:
        index = await asyncio.to_thread(build_document_index, data, filename, key)
    except RAGError as error:
        raise _rag_error(error) from error
    finally:
        await file.close()

    session_id = uuid4().hex
    documents[session_id] = ActiveDocument(index=index, created_at=time.time())
    return {
        "session_id": session_id,
        "filename": index.filename,
        "chunk_count": len(index.chunks),
        "retrieval_mode": index.retrieval_mode,
    }


@app.post("/api/query")
async def query_document(request: QuestionRequest) -> dict[str, str | list[dict[str, str | float]]]:
    """Retrieve source passages and answer a question about the active document."""
    _remove_expired_documents()
    document = documents.get(request.session_id)
    if document is None:
        raise HTTPException(status_code=404, detail="This document session expired. Upload the paper again.")

    try:
        key = _api_key()
        retrieved = await asyncio.to_thread(
            retrieve, document.index, request.question, key, request.result_count
        )
        answer = await asyncio.to_thread(answer_question, request.question, retrieved, key)
    except RAGError as error:
        raise _rag_error(error) from error

    sources = [
        {
            "label": f"S{number}",
            "location": item.source.location,
            "score": round(item.score, 2),
            "text": item.source.text,
        }
        for number, item in enumerate(retrieved, start=1)
    ]
    return {"answer": answer, "sources": sources}


@app.delete("/api/session/{session_id}")
async def clear_document(session_id: str) -> dict[str, bool]:
    documents.pop(session_id, None)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8501, reload=True)
