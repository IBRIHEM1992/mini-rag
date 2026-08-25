from app.chunking import chunk_text, normalize_text
from app.embeddings import LocalEmbeddingProvider, cosine_similarity
from app.store import KnowledgeBase


def test_normalizes_and_chunks_with_overlap():
    chunks = chunk_text("alpha   beta gamma delta epsilon zeta", size=15, overlap=4)
    assert len(chunks) > 1
    assert all("  " not in chunk for chunk in chunks)
    assert normalize_text("one\n\n two") == "one two"


def test_local_embeddings_rank_related_text():
    embedder = LocalEmbeddingProvider()
    query, relevant, unrelated = embedder.embed(["cats purr", "cats purr softly", "volcano lava"])
    assert cosine_similarity(query, relevant) > cosine_similarity(query, unrelated)


def test_knowledge_base_persists_and_deletes(tmp_path):
    db = KnowledgeBase(tmp_path)
    record = db.add("notes.txt", "text/plain", ["Useful fact"], [[1.0, 0.0]])
    assert db.search([1.0, 0.0], 1)[0]["document"]["name"] == "notes.txt"
    assert KnowledgeBase(tmp_path).list_documents()[0]["id"] == record["id"]
    assert db.delete(record["id"])
    assert not db.list_documents()
