from datetime import UTC, datetime
from math import isfinite

from .chunking import split_document
from .contracts import EmbeddingSpace, Inference, PublicationLedger, Source, VectorIndex


def now() -> str:
    return datetime.now(UTC).isoformat()


def require_gpu(space: EmbeddingSpace):
    if space.device != "cuda" and not (
        space.device.startswith("cuda:") and space.device[5:].isdigit()
    ):
        raise ValueError("inference must report an actual CUDA device; CPU fallback is disabled")
    if space.dimension < 1 or not space.model or not space.revision:
        raise ValueError("inference must identify its model, immutable revision and dimension")


def validate_vectors(vectors: list[list[float]], count: int, dimension: int):
    if len(vectors) != count or any(
        len(vector) != dimension
        or not all(isfinite(x) for x in vector)
        or not any(x != 0 for x in vector)
        for vector in vectors
    ):
        raise ValueError("embedding response has invalid count, dimension or values")


class KnowledgeBase:
    def __init__(
        self,
        ledger: PublicationLedger,
        inference: Inference,
        index: VectorIndex,
        sources: list[Source],
        chunk_size=1000,
        overlap=120,
        rerank=True,
    ):
        self.ledger, self.inference, self.index = ledger, inference, index
        self.sources = {source.id: source for source in sources}
        if len(self.sources) != len(sources):
            raise ValueError("source IDs must be unique")
        self.chunk_size, self.overlap, self.rerank_enabled = chunk_size, overlap, rerank

    def sync(self) -> dict:
        results = {}
        with self.ledger.writer():
            space = self.inference.space()
            require_gpu(space)
            self.ledger.check_space(space.fingerprint, initialize=True)
            self.index.setup(space.dimension)
            for source in self.sources.values():
                self.ledger.started(source.id, now())
                try:
                    snapshot = source.scan(self.ledger.checkpoint(source.id))
                    if not snapshot.complete:
                        raise ValueError(
                            "incomplete source snapshot; publication was not attempted"
                        )
                    keys = [doc.key for doc in snapshot.documents]
                    if len(set(keys)) != len(keys) or any(
                        doc.source_id != source.id or not doc.id for doc in snapshot.documents
                    ):
                        raise ValueError(
                            "source returned duplicate or incorrectly scoped documents"
                        )
                    keep = set()
                    changed = 0
                    for document in snapshot.documents:
                        chunks = split_document(document, self.chunk_size, self.overlap)
                        if not chunks:
                            continue
                        keep.add(document.key)
                        current = self.ledger.get(document.key)
                        if current and current[1][0].version == chunks[0].version:
                            continue
                        vectors = []
                        for start in range(0, len(chunks), 16):
                            batch = chunks[start : start + 16]
                            embedded = self.inference.embed([c.text for c in batch], "document")
                            validate_vectors(embedded, len(batch), space.dimension)
                            vectors.extend(embedded)
                        # 벡터 저장이 끝나도 장부에 게시하기 전까지는 검색 결과에 노출하지 않는다.
                        self.ledger.prepare_version(document.key, chunks[0].version)
                        self.index.upsert(chunks, vectors)
                        self.ledger.activate(document, chunks)
                        changed += 1
                    self.ledger.retire_missing(source.id, keep)
                    for key, version in self.ledger.retired():
                        self.index.delete_version(key, version)
                        self.ledger.cleaned(key, version)
                    self.ledger.finished(source.id, snapshot.checkpoint, len(keep), now())
                    results[source.id] = {
                        "status": "ok",
                        "documents": len(keep),
                        "changed": changed,
                    }
                except Exception as error:
                    self.ledger.failed(source.id, type(error).__name__)
                    results[source.id] = {"status": "error", "error": type(error).__name__}
        return results
