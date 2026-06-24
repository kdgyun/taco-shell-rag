from dataclasses import replace

import pytest
from qdrant_client import QdrantClient

from taco_shell_rag.adapters.catalog import Catalog
from taco_shell_rag.adapters.qdrant import QdrantIndex
from taco_shell_rag.core.contracts import EmbeddingSpace
from taco_shell_rag.core.service import KnowledgeBase


class FakeInference:
    def __init__(self):
        self.info = EmbeddingSpace("test-only", "fixed-test-revision", 3, "cuda:0")
        self.fail = False
        self.embedded = 0

    def space(self):
        return self.info

    def embed(self, texts, purpose):
        if self.fail:
            raise RuntimeError("test failure")
        self.embedded += len(texts)
        return [[1.0, float("refund" in text.lower()), 0.2] for text in texts]

    def rerank(self, query, texts):
        return [float(query.lower() in text.lower()) for text in texts]

    def cpu(self):
        self.info = replace(self.info, device="cpu")


@pytest.fixture
def kb(tmp_path):
    inference = FakeInference()
    client = QdrantClient(":memory:")
    service = KnowledgeBase(
        Catalog(tmp_path / "state.sqlite3"),
        inference,
        QdrantIndex(client, "test"),
        [],
        chunk_size=64,
        overlap=8,
    )
    yield service
    client.close()
