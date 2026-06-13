import json
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Protocol


@dataclass(frozen=True)
class Document:
    id: str
    source_id: str
    text: str
    title: str = ""
    url: str = ""
    path: str = ""
    revision: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return sha256(json.dumps([self.source_id, self.id]).encode()).hexdigest()

    @property
    def content_hash(self) -> str:
        raw = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
        return sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    documents: list[Document]
    checkpoint: str
    complete: bool = True


@dataclass(frozen=True)
class Chunk:
    document_key: str
    source_id: str
    version: str
    index: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class EmbeddingSpace:
    model: str
    revision: str
    dimension: int
    device: str

    @property
    def fingerprint(self) -> str:
        raw = json.dumps([self.model, self.revision, self.dimension, "normalized-dense-v1"])
        return sha256(raw.encode()).hexdigest()


class Source(Protocol):
    id: str

    def scan(self, checkpoint: str | None) -> Snapshot: ...


class Inference(Protocol):
    def space(self) -> EmbeddingSpace: ...

    def embed(self, texts: list[str], purpose: str) -> list[list[float]]: ...

    def rerank(self, query: str, texts: list[str]) -> list[float]: ...


class VectorIndex(Protocol):
    def setup(self, dimension: int) -> None: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], limit: int, source_id: str | None) -> list[Hit]: ...

    def delete_version(self, document_key: str, version: str) -> None: ...


class PublicationLedger(Protocol):
    def writer(self) -> AbstractContextManager: ...
    def check_space(self, fingerprint: str, *, initialize: bool = False) -> None: ...
    def get(self, key: str) -> tuple[Document, list[Chunk]] | None: ...
    def activate(self, document: Document, chunks: list[Chunk]) -> None: ...
    def prepare_version(self, key: str, version: str) -> None: ...
    def retire_missing(self, source_id: str, keep: set[str]) -> None: ...
    def retired(self) -> list[tuple[str, str]]: ...
    def cleaned(self, key: str, version: str) -> None: ...
    def checkpoint(self, source_id: str) -> str | None: ...
    def started(self, source_id: str, now: str) -> None: ...
    def finished(self, source_id: str, checkpoint: str, count: int, now: str) -> None: ...
    def failed(self, source_id: str, kind: str) -> None: ...
    def status(self) -> dict: ...
