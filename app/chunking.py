"""Text cleanup and overlap-aware chunking."""
import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Avoid splitting a word when a suitable breakpoint is nearby.
            breakpoint = text.rfind(" ", start + size // 2, end)
            if breakpoint > start:
                end = breakpoint
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
