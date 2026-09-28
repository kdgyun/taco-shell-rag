import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from taco_shell_rag.core.contracts import (
    EmbeddingResult,
    EmbeddingSpace,
    RerankerInfo,
    RerankResult,
)
from taco_shell_rag.gpu import GpuModels, check_loaded_model, create_app, require_cuda
from taco_shell_rag.model_config import (
    EmbeddingOptions,
    ModelConfigurationError,
    ModelSettings,
    RerankerOptions,
)


def test_runtime_explicitly_refuses_missing_cuda():
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))


class FakeCudaDevice:
    type = "cuda"

    def __init__(self, value="cuda:0"):
        self.value = value

    def __str__(self):
        return self.value


@pytest.fixture
def fake_libraries(monkeypatch):
    calls, loads = [], []
    encoder = SimpleNamespace(
        device=FakeCudaDevice(),
        dtype="float32",
        max_seq_length=8192,
        get_embedding_dimension=lambda: 3,
    )
    reranker = SimpleNamespace(
        device=FakeCudaDevice(),
        dtype="float32",
        max_seq_length=512,
        num_labels=1,
        prompts={"query": "native instruction"},
        default_prompt_name="query",
    )

    def encode(purpose, texts, **kwargs):
        calls.append((purpose, texts, kwargs))
        return SimpleNamespace(tolist=lambda: [[1.0, 0.0, 0.5] for _ in texts])

    def predict(pairs, **kwargs):
        calls.append(("rerank", pairs, kwargs))
        return SimpleNamespace(tolist=lambda: [0.2 for _ in pairs])

    def load(kind, model, **kwargs):
        loads.append((kind, model, kwargs))
        target = encoder if kind == "embedding" else reranker
        target.device = FakeCudaDevice(kwargs["device"])
        target.dtype = kwargs["model_kwargs"]["dtype"]
        if kind == "reranker":
            target.max_seq_length = kwargs["processor_kwargs"]["model_max_length"]
        return target

    encoder.encode_query = lambda texts, **kw: encode("query", texts, **kw)
    encoder.encode_document = lambda texts, **kw: encode("document", texts, **kw)
    reranker.predict = predict
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            nn=SimpleNamespace(Identity=lambda: None),
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda name, **kw: load("embedding", name, **kw),
            CrossEncoder=lambda name, **kw: load("reranker", name, **kw),
        ),
    )
    return SimpleNamespace(encoder=encoder, reranker=reranker, calls=calls, loads=loads)


def settings():
    return ModelSettings(
        embedding=EmbeddingOptions(preset="bge-m3"),
        reranker=RerankerOptions(preset="qwen3-reranker-4b", precision="float16"),
    )


def test_model_setup_uses_current_embedding_dimension_api_with_fake_models(fake_libraries):
    engine = GpuModels(settings(), "all")
    assert engine.info.dimension == 3
    assert engine.reranker_info.model == "Qwen/Qwen3-Reranker-4B"
    assert engine.reranker_info.precision == "float16"
    assert all(load[2]["trust_remote_code"] is False for load in fake_libraries.loads)
    assert all(load[2]["backend"] == "torch" for load in fake_libraries.loads)


@pytest.mark.parametrize("role", ["embedding", "reranker"])
def test_role_only_loads_the_selected_model(fake_libraries, role):
    engine = GpuModels(settings(), role)
    assert [load[0] for load in fake_libraries.loads] == [role]
    assert (engine.info is not None) == (role == "embedding")
    assert (engine.reranker_info is not None) == (role == "reranker")


def test_query_document_and_reranker_options_are_forwarded_independently(fake_libraries):
    config = ModelSettings(
        embedding=EmbeddingOptions(
            preset="multilingual-e5-large",
            batch_size=2,
            max_length=256,
            device="cuda:1",
        ),
        reranker=RerankerOptions(
            preset="qwen3-reranker-8b", prompt="custom instruction", batch_size=1
        ),
    )
    engine = GpuModels(config, "all")
    engine.embed(["question"], "query")
    engine.embed(["document"], "document")
    result = engine.rerank("question", ["document"])
    assert fake_libraries.calls[0][2]["prompt"] == "query: "
    assert fake_libraries.calls[1][2]["prompt"] == "passage: "
    assert fake_libraries.calls[0][2]["batch_size"] == 2
    assert fake_libraries.calls[0][2]["normalize_embeddings"] is True
    assert fake_libraries.calls[2][2]["prompt"] == "custom instruction"
    assert fake_libraries.calls[2][2]["batch_size"] == 1
    assert engine.info.device == "cuda:1"
    assert engine.info.encoding["max_length"] == 256
    assert result.info.model == "Qwen/Qwen3-Reranker-8B"
    assert result.scores == [0.2]


def test_empty_prompt_disables_native_prompt_without_becoming_none(fake_libraries):
    config = ModelSettings(embedding=EmbeddingOptions(preset="bge-m3", query_prompt=""))
    GpuModels(config, "embedding").embed(["question"], "query")
    assert fake_libraries.calls[0][2]["prompt"] == ""


def test_batch_size_and_gpu_placement_do_not_change_embedding_space(fake_libraries):
    first = GpuModels(
        ModelSettings(embedding=EmbeddingOptions(preset="bge-m3", batch_size=1)), "embedding"
    ).info
    second = GpuModels(
        ModelSettings(embedding=EmbeddingOptions(preset="bge-m3", batch_size=16, device="cuda:1")),
        "embedding",
    ).info
    assert first.device != second.device
    assert first.fingerprint == second.fingerprint


def test_legacy_model_environment_fails_before_loading_weights(fake_libraries, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "old-configuration")
    with pytest.raises(ModelConfigurationError, match="models.toml"):
        GpuModels(role="embedding")
    assert fake_libraries.loads == []


def test_multi_label_reranker_is_rejected_before_serving(fake_libraries):
    fake_libraries.reranker.num_labels = 2
    with pytest.raises(ValueError, match="one scalar"):
        GpuModels(settings(), "reranker")


def test_runtime_refuses_ignored_precision_or_cpu_model():
    with pytest.raises(RuntimeError, match="precision"):
        check_loaded_model(SimpleNamespace(device=FakeCudaDevice(), dtype="float16"), "float32")
    with pytest.raises(RuntimeError, match="CUDA"):
        check_loaded_model(SimpleNamespace(device=SimpleNamespace(type="cpu")), "float32")


def test_inference_api_contract_and_bounds_without_loading_gpu_models():
    class Engine:
        info = EmbeddingSpace("fake-test-model", "test-revision", 2, "cuda:0")
        reranker_info = RerankerInfo("fake-reranker", "test-revision", "cuda:0", "fake-test")

        def embed(self, texts, purpose):
            assert purpose == "query"
            return EmbeddingResult([[1.0, 0.5] for _ in texts], self.info)

        def rerank(self, query, texts):
            return RerankResult([0.2 for _ in texts], self.reranker_info)

    with TestClient(create_app(Engine)) as client:
        health = client.get("/healthz").json()
        assert health["ready"] and health["embedding_space"]["model"] == "fake-test-model"
        embedded = client.post("/embed", json={"texts": ["policy"], "purpose": "query"})
        assert embedded.json() == {
            "vectors": [[1.0, 0.5]],
            "embedding_space": asdict(Engine.info),
        }
        assert client.post("/embed", json={"texts": [], "purpose": "query"}).status_code == 422
        assert (
            client.post("/embed", json={"texts": ["a" * 4001], "purpose": "query"}).status_code
            == 422
        )
        ranked = client.post("/rerank", json={"query": "policy", "texts": ["evidence"]}).json()
        assert ranked == {"scores": [0.2], "reranker": asdict(Engine.reranker_info)}
        assert client.post("/rerank", json={"query": "", "texts": ["a"]}).status_code == 422


def test_disabled_role_fails_without_loading_or_falling_back(fake_libraries):
    config = ModelSettings(embedding=EmbeddingOptions(preset="bge-m3"))
    with TestClient(create_app(lambda: GpuModels(config, "embedding"))) as client:
        assert client.get("/healthz").json()["reranker"] is None
        assert client.post("/rerank", json={"query": "q", "texts": ["text"]}).status_code == 503
    assert [load[0] for load in fake_libraries.loads] == ["embedding"]


def test_embedding_http_metadata_comes_from_execution_not_health_snapshot():
    actual = EmbeddingSpace("served-encoder", "served-revision", 2, "cuda:0")

    class Engine:
        info = EmbeddingSpace("health-encoder", "health-revision", 2, "cuda:0")
        reranker_info = None

        def embed(self, texts, purpose):
            return EmbeddingResult([[1.0, 0.5] for _ in texts], actual)

    with TestClient(create_app(Engine)) as client:
        response = client.post("/embed", json={"texts": ["text"], "purpose": "query"})
        assert response.json()["embedding_space"] == asdict(actual)
