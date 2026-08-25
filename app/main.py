from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .chunking import chunk_text
from .config import settings
from .embeddings import provider
from .store import KnowledgeBase
from .generation import grounded_answer


def extract_text(filename: str, payload: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            from io import BytesIO
            return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)
        except Exception as error:
            raise HTTPException(422, "The PDF could not be read.") from error
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(415, "Only UTF-8 text, Markdown, and PDF files are supported.") from error


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.kb = KnowledgeBase(settings.data_dir)
    app.state.embedder = provider(settings.openai_api_key, settings.embedding_model)
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health(request: Request):
    return {"status": "ok", "documents": len(request.app.state.kb.documents), "provider": "openai" if settings.openai_api_key else "local"}


@app.get("/api/documents")
def documents(request: Request):
    return request.app.state.kb.list_documents()


@app.post("/api/documents", status_code=201)
async def upload_document(request: Request, file: Annotated[UploadFile, File(...)]):
    if not file.filename:
        raise HTTPException(422, "A filename is required.")
    allowed = (".txt", ".md", ".markdown", ".pdf")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(415, "Supported formats: TXT, Markdown, PDF.")
    payload = await file.read(settings.max_upload_bytes + 1)
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(413, "File exceeds the configured upload limit.")
    text = extract_text(file.filename, payload)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(422, "The document has no extractable text.")
    vectors = request.app.state.embedder.embed(chunks)
    return request.app.state.kb.add(file.filename, file.content_type, chunks, vectors)


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request):
    if not request.app.state.kb.delete(document_id):
        raise HTTPException(404, "Document not found.")


@app.post("/api/query")
def query(payload: QueryRequest, request: Request):
    kb = request.app.state.kb
    if not kb.chunks:
        raise HTTPException(409, "Upload at least one document before asking a question.")
    matches = kb.search(request.app.state.embedder.embed([payload.question])[0], payload.top_k)
    sources = [{"document_id": match["document_id"], "document": match["document"]["name"], "chunk_id": match["id"], "score": round(match["score"], 4), "text": match["text"]} for match in matches]
    if settings.openai_api_key:
        try:
            answer = grounded_answer(settings.openai_api_key, settings.chat_model, payload.question, sources)
            mode = "generated"
        except Exception as error:
            raise HTTPException(502, "The language model request failed. Check your OpenAI configuration.") from error
    else:
        # The offline mode deliberately avoids pretending to generate an answer.
        answer = "\n\n".join(f"From {item['document']}: {item['text']}" for item in sources)
        mode = "extractive"
    return {"answer": answer, "sources": sources, "mode": mode}
