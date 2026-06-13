from hashlib import sha256

from .contracts import Chunk, Document


def split_document(document: Document, size: int = 1000, overlap: int = 120) -> list[Chunk]:
    if not 64 <= size <= 4000 or not 0 <= overlap < size:
        raise ValueError("chunk size must be 64..4000 and overlap smaller than size")
    # 청크 크기를 바꿨을 때 색인이 생략되지 않도록 분할 설정도 버전에 포함한다.
    version = sha256(f"{document.content_hash}:{size}:{overlap}:v1".encode()).hexdigest()
    text = document.text
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(
            Chunk(document.key, document.source_id, version, len(chunks), text[start:end])
        )
        if end == len(text):
            break
        start = end - overlap
    return chunks
