import tomllib
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from taco_shell_rag import config
from taco_shell_rag.agent.models import GenerationConfig
from taco_shell_rag.model_config import load_model_settings


@pytest.mark.parametrize("rerank", [True, False])
def test_factory_wires_separate_endpoints_and_credentials(tmp_path, monkeypatch, rerank):
    client = QdrantClient(":memory:")
    monkeypatch.setattr(config, "QdrantClient", lambda *_, **__: client)
    monkeypatch.setenv("TEST_EMBEDDING_TOKEN", "embedding-only-secret")
    monkeypatch.setenv("TEST_RERANKER_TOKEN", "reranker-only-secret")
    settings = config.Settings(
        embedding_url="https://embedding.test",
        reranker_url="https://reranker.test",
        embedding_token_env="TEST_EMBEDDING_TOKEN",
        reranker_token_env="TEST_RERANKER_TOKEN" if rerank else "TEST_UNUSED_TOKEN",
        rerank=rerank,
        sources=[{"id": "notes", "type": "folder", "path": "notes"}],
    )
    try:
        kb = config.build_knowledge_base(settings, tmp_path)
        assert kb.embedder.url == "https://embedding.test"
        assert kb.embedder.headers == {"Authorization": "Bearer embedding-only-secret"}
        if rerank:
            assert kb.reranker.url == "https://reranker.test"
            assert kb.reranker.headers == {"Authorization": "Bearer reranker-only-secret"}
        else:
            assert kb.reranker is None
    finally:
        client.close()


def test_legacy_combined_endpoint_setting_remains_accepted():
    settings = config.Settings(
        inference_url="http://legacy.test:8000",
        sources=[{"id": "notes", "type": "folder", "path": "notes"}],
    )
    assert settings.embedding_url == "http://legacy.test:8000"
    assert settings.reranker_url is None


def test_missing_configured_credential_fails_before_external_connection(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_MISSING_QDRANT_TOKEN", raising=False)
    settings = config.Settings(
        qdrant_token_env="TEST_MISSING_QDRANT_TOKEN",
        sources=[{"id": "notes", "type": "folder", "path": "notes"}],
    )
    with pytest.raises(ValueError, match="credential environment variable is unset"):
        config.build_knowledge_base(settings, tmp_path)


@pytest.mark.parametrize("path", sorted(Path("examples/inference").glob("*.toml")))
def test_example_inference_profiles_are_valid_without_loading_weights(path):
    load_model_settings(path)


@pytest.mark.parametrize("path", sorted(Path("examples/models").glob("*.toml")))
def test_example_generation_profiles_are_valid_without_calling_models(path):
    with path.open("rb") as stream:
        GenerationConfig.model_validate(tomllib.load(stream))
