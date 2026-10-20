import json
from types import SimpleNamespace

import pytest
from qdrant_client import models

from taco_shell_rag import cli
from taco_shell_rag.agent.models import GenerationConfig
from taco_shell_rag.core.contracts import EmbeddingSpaceMismatch


def prepare_doctor(kb, monkeypatch):
    monkeypatch.setattr("sys.argv", ["taco", "doctor"])
    monkeypatch.setattr(cli, "load_settings", lambda _: object())
    monkeypatch.setattr(cli, "build_knowledge_base", lambda *_: kb)


def test_doctor_checks_qdrant_without_initializing_collection(kb, monkeypatch, capsys):
    prepare_doctor(kb, monkeypatch)
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ready"]
    assert not result["vector_index"]["exists"]
    assert not kb.index.client.collection_exists("test")


def test_doctor_fails_when_qdrant_is_unreachable_without_leaking_error(kb, monkeypatch, capsys):
    prepare_doctor(kb, monkeypatch)

    def unavailable(*_):
        raise ConnectionError("private-token-in-external-response")

    monkeypatch.setattr(kb.index.client, "collection_exists", unavailable)
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "ConnectionError" in captured.err
    assert "private-token" not in captured.err


def test_permission_failure_explains_mount_access_without_leaking_path(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["taco", "doctor"])

    def unreadable(_):
        raise PermissionError("/private/config-with-secret-token.toml")

    monkeypatch.setattr(cli, "load_settings", unreadable)
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "PermissionError" in captured.err
    assert "UID 10001" in captured.err
    assert "docs/install.md" in captured.err
    assert "secret-token" not in captured.err


def test_doctor_rejects_incompatible_existing_collection(kb, monkeypatch, capsys):
    prepare_doctor(kb, monkeypatch)
    kb.index.client.create_collection(
        "test", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
    )
    assert cli.main() == 1
    assert not capsys.readouterr().out


def test_embedding_migration_error_has_an_actionable_safe_hint(kb, monkeypatch, capsys):
    prepare_doctor(kb, monkeypatch)

    def incompatible(*_):
        raise EmbeddingSpaceMismatch("private-catalog-path")

    monkeypatch.setattr(kb.ledger, "check_space", incompatible)
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "new collection and catalog" in captured.err
    assert "private-catalog-path" not in captured.err


def test_remote_claude_cli_does_not_build_local_retrieval(monkeypatch, capsys):
    settings = SimpleNamespace(generation=GenerationConfig(provider="claude-cli", model="test"))
    monkeypatch.setattr("sys.argv", ["taco", "ask", "refund?"])
    monkeypatch.setattr(cli, "load_settings", lambda _: settings)

    def unexpected(*_):
        raise AssertionError("remote MCP must not initialize a local backend")

    monkeypatch.setattr(cli, "build_knowledge_base", unexpected)
    monkeypatch.setattr(
        "taco_shell_rag.agent.claude_cli.run_claude",
        lambda *_: {"status": "completed", "answer": "remote answer"},
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["answer"] == "remote answer"


def test_watch_survives_backend_outage_and_retries(monkeypatch, capsys):
    attempts = 0

    def sync():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("private response body")
        return {"notes": {"status": "ok", "changed": 0}}

    def sleep(_):
        if attempts == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("sys.argv", ["taco", "sync", "--watch", "--interval", "10"])
    monkeypatch.setattr(cli, "load_settings", lambda _: object())
    monkeypatch.setattr(cli, "build_knowledge_base", lambda *_: SimpleNamespace(sync=sync))
    monkeypatch.setattr(cli.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        cli.main()
    output = capsys.readouterr().out
    events = [json.loads(line) for line in output.splitlines()]
    assert events[0]["sync"] == {"status": "error", "error": "ConnectionError"}
    assert events[1]["notes"]["status"] == "ok"
    assert "private response" not in output


def test_cli_lists_models_without_backend_or_source_access(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["taco", "models"])

    def unexpected(*_):
        raise AssertionError("model discovery must stay offline")

    monkeypatch.setattr(cli, "load_settings", unexpected)
    assert cli.main() == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["presets"]) == {"embedding", "reranker"}
    assert "qwen3-reranker-8b" in data["presets"]["reranker"]


def test_cli_shows_resolved_model_options_without_loading_gpu(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["taco", "models", "--model-config", "models.toml"])
    assert cli.main() == 0
    data = json.loads(capsys.readouterr().out)
    assert data["embedding"]["model"] == "BAAI/bge-m3"
    assert len(data["reranker"]["revision"]) == 40


def test_missing_go_binary_has_an_actionable_hint_without_its_private_path(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["taco", "search", "refund", "--mode", "keyword"])
    monkeypatch.setenv("MCP_STDIO_BIN", "/nonexistent/private-binary-location")
    assert cli.main() == 1
    error = capsys.readouterr().err
    assert "make build" in error and "MCP_STDIO_BIN" in error
    assert "private-binary-location" not in error
