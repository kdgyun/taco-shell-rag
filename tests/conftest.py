import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from taco_shell_rag.core.contracts import (
    EmbeddingResult,
    EmbeddingSpace,
    RerankerInfo,
    RerankResult,
)
from taco_shell_rag.indexer.catalog import Catalog
from taco_shell_rag.indexer.pipeline import KnowledgeBase
from taco_shell_rag.indexer.vectorstore import QdrantIndex


@pytest.fixture(scope="session")
def go_binaries(tmp_path_factory):
    if not shutil.which("go"):
        if os.environ.get("RUN_GO_GPU_TESTS") == "1":
            pytest.fail("Go toolchain is required for the requested GPU MCP verification")
        pytest.skip("Go toolchain required for cross-language MCP integration")
    directory = tmp_path_factory.mktemp("go-mcp-binaries")
    binaries = {}
    for name in ["http", "stdio"]:
        binary = directory / f"mcp-{name}"
        subprocess.run(
            ["go", "build", "-mod=readonly", "-o", str(binary), f"./cmd/mcp-{name}"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            timeout=180,
        )
        binaries[name] = str(binary)
    return binaries


class FakeEmbedder:
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
        vectors = [[1.0, float("refund" in text.lower()), 0.2] for text in texts]
        return EmbeddingResult(vectors, self.info)

    def cpu(self):
        self.info = replace(self.info, device="cpu")


class FakeReranker:
    def __init__(self):
        self.info = RerankerInfo("test-reranker", "fixed-test-revision", "cuda:0", "fake-test")
        self.fail = False
        self.calls = 0

    def describe(self):
        if self.fail:
            raise ConnectionError("fake reranker unavailable")
        return self.info

    def rerank(self, query, texts):
        self.calls += 1
        scores = [float(query.lower() in text.lower()) for text in texts]
        return RerankResult(scores, self.describe())


@pytest.fixture
def kb(tmp_path):
    embedder = FakeEmbedder()
    client = QdrantClient(":memory:")
    service = KnowledgeBase(
        Catalog(tmp_path / "state.sqlite3"),
        embedder,
        QdrantIndex(client, "test"),
        [],
        chunk_size=64,
        overlap=8,
        reranker=FakeReranker(),
    )
    yield service
    client.close()
