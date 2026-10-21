import os
from dataclasses import asdict
from math import isclose, isfinite, sqrt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp import Client
from qdrant_client import QdrantClient

from taco_shell_rag.gpu import GpuModels, create_app
from taco_shell_rag.indexer.catalog import Catalog
from taco_shell_rag.indexer.pipeline import (
    KnowledgeBase,
    require_gpu,
    require_reranker,
    validate_vectors,
)
from taco_shell_rag.indexer.sources import FolderSource
from taco_shell_rag.indexer.vectorstore import QdrantIndex
from tests.mcp_fixture import create_server

# 기본 테스트에서는 가중치를 받거나 GPU를 사용하지 않는다.
# Do not download weights or use a GPU unless the integration run is explicitly enabled.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="requires RUN_GPU_TESTS=1"),
]


@pytest.fixture(scope="module")
def gpu_models():
    return GpuModels()


@pytest.fixture
def gpu_kb(gpu_models, tmp_path):
    if gpu_models.encoder is None:
        pytest.skip("embedding role is disabled")
    index = QdrantClient(":memory:")
    try:
        yield KnowledgeBase(
            Catalog(tmp_path / "catalog.sqlite3"),
            gpu_models.encoder,
            QdrantIndex(index, "gpu_test"),
            [FolderSource("handbook", Path("examples/knowledge"))],
            rerank=gpu_models.reranker is not None,
            reranker=gpu_models.reranker,
        )
    finally:
        index.close()


def test_embedding_vectors_match_declared_space_and_are_normalized(gpu_models):
    if gpu_models.encoder is None:
        pytest.skip("embedding role is disabled")
    space = gpu_models.encoder.space()
    require_gpu(space)
    for purpose, texts in [
        ("query", ["환불은 언제까지 가능한가요?", "When can I request a refund?"]),
        ("document", ["미사용 구독은 구매 후 7일 이내에 환불할 수 있습니다."]),
    ]:
        result = gpu_models.embed(texts, purpose)
        assert result.space == space
        vectors = result.vectors
        validate_vectors(vectors, len(texts), space.dimension)
        for vector in vectors:
            assert isclose(sqrt(sum(value * value for value in vector)), 1.0, abs_tol=0.01)


def test_reranker_returns_finite_scores_and_execution_identity(gpu_models):
    if gpu_models.reranker is None:
        pytest.skip("reranker role is disabled")
    texts = [
        "미사용 구독은 구매 후 7일 이내에 환불할 수 있습니다.",
        "연락처는 계정 설정에서 변경할 수 있습니다.",
    ]
    result = gpu_models.rerank("환불은 언제까지 가능한가요?", texts)
    require_reranker(result.info)
    assert result.info == gpu_models.reranker.describe()
    assert len(result.scores) == len(texts)
    assert all(isfinite(score) for score in result.scores)


def test_inference_http_contract_reports_the_loaded_roles(gpu_models):
    with TestClient(create_app(lambda: gpu_models)) as client:
        health = client.get("/healthz").json()
        assert health["ready"]
        embedded = client.post("/embed", json={"texts": ["refund policy"], "purpose": "query"})
        if gpu_models.info is None:
            assert health["embedding_space"] is None
            assert embedded.status_code == 503
        else:
            assert health["embedding_space"] == asdict(gpu_models.info)
            assert embedded.status_code == 200
            assert embedded.json()["embedding_space"] == asdict(gpu_models.info)
            validate_vectors(embedded.json()["vectors"], 1, gpu_models.info.dimension)
        ranked = client.post("/rerank", json={"query": "refund", "texts": ["refund policy"]})
        if gpu_models.reranker_info is None:
            assert health["reranker"] is None
            assert ranked.status_code == 503
        else:
            assert health["reranker"] == asdict(gpu_models.reranker_info)
            assert ranked.status_code == 200
            result = ranked.json()
            assert result["reranker"] == asdict(gpu_models.reranker_info)
            assert len(result["scores"]) == 1 and isfinite(result["scores"][0])


async def test_gpu_indexing_search_and_exact_document_over_mcp(gpu_models, gpu_kb):
    first = gpu_kb.sync()["handbook"]
    assert first["status"] == "ok" and first["changed"] == 1
    assert gpu_kb.sync()["handbook"]["changed"] == 0
    async with Client(create_server(gpu_kb)) as client:
        result = await client.call_tool("search", {"query": "refund"})
        assert not result.is_error
        data = result.structured_content
        assert data["embedding_space"] == asdict(gpu_models.info)
        assert data["rerank_enabled"] == (gpu_models.reranker is not None)
        if gpu_models.reranker_info is not None:
            assert data["reranker"] == asdict(gpu_models.reranker_info)
        chunk = data["results"][0]["chunk"]
        document = await client.call_tool(
            "get_document",
            {"document_key": chunk["document_key"], "version": chunk["version"]},
        )
        assert not document.is_error
        assert "seven days" in str(document.content)
