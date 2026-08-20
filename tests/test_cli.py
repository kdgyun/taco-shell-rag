import json
from types import SimpleNamespace

from qdrant_client import models

from taco_shell_rag import cli
from taco_shell_rag.agent.models import GenerationConfig


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


def test_doctor_rejects_incompatible_existing_collection(kb, monkeypatch, capsys):
    prepare_doctor(kb, monkeypatch)
    kb.index.client.create_collection(
        "test", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
    )
    assert cli.main() == 1
    assert not capsys.readouterr().out


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
