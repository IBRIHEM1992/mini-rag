"""Embedding providers. The local provider keeps the demo useful offline."""
from __future__ import annotations
import hashlib
import math
import re
from functools import lru_cache

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


class LocalEmbeddingProvider:
    """A deterministic hashing-vector baseline; not a substitute for semantic embeddings."""
    dimensions = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _TOKEN.findall(text.lower()):
            index = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") % self.dimensions
            vector[index] += 1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


@lru_cache(maxsize=1)
def provider(api_key: str | None, model: str):
    return OpenAIEmbeddingProvider(api_key, model) if api_key else LocalEmbeddingProvider()


def cosine_similarity(first: list[float], second: list[float]) -> float:
    # Vectors are normally normalised, but calculate the general form for safety.
    numerator = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(a * a for a in first))
    second_length = math.sqrt(sum(b * b for b in second))
    return numerator / (first_length * second_length) if first_length and second_length else 0.0
