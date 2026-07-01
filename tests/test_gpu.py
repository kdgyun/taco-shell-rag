from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from taco_shell_rag.core.contracts import EmbeddingSpace
from taco_shell_rag.gpu import create_app, require_cuda


def test_runtime_explicitly_refuses_missing_cuda():
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))


def test_inference_api_contract_and_bounds_without_loading_gpu_models():
    class Engine:
        info = EmbeddingSpace("fake-test-model", "test-revision", 2, "cuda:0")
        reranker_info = {"model": "fake-reranker", "revision": "test-revision"}

        def embed(self, texts, purpose):
            assert purpose == "query"
            return [[1.0, 0.5] for _ in texts]

        def rerank(self, query, texts):
            return [0.2 for _ in texts]

    with TestClient(create_app(Engine)) as client:
        health = client.get("/healthz").json()
        assert health["ready"] and health["embedding_space"]["model"] == "fake-test-model"
        embedded = client.post("/embed", json={"texts": ["policy"], "purpose": "query"})
        assert embedded.json() == {"vectors": [[1.0, 0.5]]}
        assert client.post("/embed", json={"texts": [], "purpose": "query"}).status_code == 422
        assert (
            client.post("/embed", json={"texts": ["a" * 4001], "purpose": "query"}).status_code
            == 422
        )
        assert client.post("/rerank", json={"query": "policy", "texts": ["evidence"]}).json() == {
            "scores": [0.2]
        }
        assert client.post("/rerank", json={"query": "", "texts": ["a"]}).status_code == 422
