# mini-rag

A small, runnable retrieval-augmented generation (RAG) application. Upload TXT, Markdown, or PDF files, retrieve the most relevant passages, and ask grounded questions through a clean web interface or JSON API.

It works out of the box **without an API key** using deterministic local retrieval and extractive answers. Add an OpenAI key to use semantic embeddings and generate an answer constrained to the retrieved sources.

## Features

- Upload UTF-8 `.txt`, `.md`, `.markdown`, and `.pdf` documents
- Sensible text chunking with overlap
- Persistent, local JSON knowledge base in `data/`
- Local offline embedding fallback; optional OpenAI embeddings and chat generation
- Source passages and relevance scores returned with every answer
- FastAPI API, interactive Swagger docs, and a lightweight browser UI
- File size limits and format validation

> This is a compact single-user demo. Do not expose it publicly with untrusted users or use it as a multi-tenant production knowledge store without adding authentication, authorization, durable database/vector storage, rate limits, and observability.

## Requirements

- Python 3.10+
- An OpenAI API key is optional

## Quick start

```bash
# Clone the repository, then from its root:
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Optionally add OPENAI_API_KEY to .env
```

The application reads `.env` automatically when it starts. Start the application:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000). Interactive API documentation is at [http://localhost:8000/docs](http://localhost:8000/docs).

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service status and selected provider |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/documents` | Upload a document as multipart field `file` |
| `DELETE` | `/api/documents/{id}` | Remove a document and its chunks |
| `POST` | `/api/query` | Retrieve and answer a question |

Example query:

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does the document say about refunds?", "top_k":4}'
```

Responses include `answer`, `mode`, and `sources`. In offline mode, `answer` is the top retrieved passages rather than an LLM-generated summary. This makes the behavior transparent and keeps the project usable without external services.

## Configuration

Copy `.env.example` and adjust values as needed:

- `OPENAI_API_KEY`: enables OpenAI embeddings and grounded generated responses
- `OPENAI_EMBEDDING_MODEL`: defaults to `text-embedding-3-small`
- `OPENAI_CHAT_MODEL`: defaults to `gpt-4.1-mini`
- `DATA_DIR`: local persistent storage directory (default `data`)
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: chunking controls
- `MAX_UPLOAD_BYTES`: upload limit, default 5 MiB

Changing embedding providers/models after indexing documents requires deleting and re-uploading documents, since vectors from different models are incompatible.

## Development

```bash
python -m pytest
```

The local `data/` directory is intentionally ignored by Git so private uploaded files and embeddings are never committed.

## Project layout

```text
app/
  main.py          FastAPI routes and upload/query workflow
  chunking.py      Text normalization and chunking
  embeddings.py    Local and OpenAI embedding providers
  generation.py    Grounded OpenAI response generation
  store.py         Persistent local knowledge-base store
  static/          Browser interface
tests/             Unit tests
```
