import pytest

from taco_shell_rag.agent.models import GenerationConfig, HttpModel


def test_anthropic_requires_an_api_key_not_ambient_cli_oauth(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "not-an-api-key")
    config = GenerationConfig(provider="anthropic", model="test", base_url="https://api.example/v1")
    with pytest.raises(ValueError, match="API key"):
        HttpModel(config)
