from __future__ import annotations
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4
from .embeddings import cosine_similarity


class KnowledgeBase:
    def __init__(self, directory: Path):
        self.path = directory / "knowledge_base.json"
        self._lock = RLock()
        directory.mkdir(parents=True, exist_ok=True)
        self.documents: dict[str, dict] = {}
        self.chunks: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.documents = payload.get("documents", {})
            self.chunks = payload.get("chunks", [])

    def _save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"documents": self.documents, "chunks": self.chunks}), encoding="utf-8")
        temporary.replace(self.path)

    def add(self, name: str, content_type: str | None, chunks: list[str], vectors: list[list[float]]) -> dict:
        document_id = str(uuid4())
        record = {"id": document_id, "name": name, "content_type": content_type or "application/octet-stream", "chunk_count": len(chunks)}
        with self._lock:
            self.documents[document_id] = record
            self.chunks.extend({"id": str(uuid4()), "document_id": document_id, "text": text, "vector": vector} for text, vector in zip(chunks, vectors))
            self._save()
        return record

    def list_documents(self) -> list[dict]:
        with self._lock:
            return list(self.documents.values())

    def delete(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self.documents:
                return False
            del self.documents[document_id]
            self.chunks = [chunk for chunk in self.chunks if chunk["document_id"] != document_id]
            self._save()
        return True

    def search(self, query_vector: list[float], limit: int) -> list[dict]:
        with self._lock:
            ranked = sorted(self.chunks, key=lambda item: cosine_similarity(query_vector, item["vector"]), reverse=True)
            return [{**chunk, "score": cosine_similarity(query_vector, chunk["vector"]), "document": self.documents[chunk["document_id"]]} for chunk in ranked[:limit]]
