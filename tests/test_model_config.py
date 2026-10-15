from pathlib import Path

import pytest
from pydantic import ValidationError

from taco_shell_rag.model_config import (
    PRESETS,
    EmbeddingOptions,
    ModelConfigurationError,
    ModelSettings,
    RerankerOptions,
    load_model_settings,
)


@pytest.mark.parametrize("name", PRESETS["embedding"])
def test_embedding_presets_resolve_without_loading_models(name):
    options = EmbeddingOptions(preset=name)
    assert options.model == PRESETS["embedding"][name]["model"]
    assert len(options.revision) == 40


@pytest.mark.parametrize("name", PRESETS["reranker"])
def test_reranker_presets_resolve_without_loading_models(name):
    options = RerankerOptions(preset=name)
    assert options.model == PRESETS["reranker"][name]["model"]
    assert len(options.revision) == 40


def test_custom_models_are_not_restricted_to_presets():
    options = EmbeddingOptions(model="my-org/my-native-encoder", revision="c" * 40)
    assert options.preset is None
    assert options.model == "my-org/my-native-encoder"


@pytest.mark.parametrize(
    "values",
    [
        {"preset": "qwen3-reranker-4b"},
        {"preset": "unknown"},
        {"preset": "bge-m3", "revision": "a" * 40},
        {"preset": "bge-m3", "model": "other"},
        {"model": "custom", "revision": "main"},
        {"preset": "bge-m3", "device": "cpu"},
        {"preset": "bge-m3", "backend": "onnx"},
        {"preset": "bge-m3", "precision": "int8"},
        {"preset": "bge-m3", "batch_size": 0},
    ],
)
def test_ambiguous_incompatible_or_unpinned_settings_are_rejected(values):
    with pytest.raises(ValidationError):
        EmbeddingOptions(**values)


def test_default_profile_and_role_specific_profiles():
    config = load_model_settings(Path("models.toml"))
    assert config.embedding.model == "BAAI/bge-m3"
    assert config.reranker.model == "BAAI/bge-reranker-v2-m3"
    assert ModelSettings(embedding=config.embedding).reranker is None
    assert ModelSettings(reranker=config.reranker).embedding is None
    with pytest.raises(ValidationError):
        ModelSettings()


def test_invalid_model_settings_do_not_echo_input_values(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text('[embedding]\npreset="bge-m3"\nprecision="private-input-value"\n')
    with pytest.raises(ModelConfigurationError) as error:
        load_model_settings(path)
    assert "private-input-value" not in str(error.value)
