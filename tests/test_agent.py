import json

import httpx
import pytest
from mcp import Client

from taco_shell_rag.agent.models import GenerationConfig, HttpModel
from taco_shell_rag.agent.runner import run_agent
from taco_shell_rag.core.contracts import Document, Snapshot
from taco_shell_rag.mcp_server import create_server


def populate(kb):
    class Notes:
        id = "notes"

        def scan(self, checkpoint):
            return Snapshot([Document("refund", self.id, "Refund within seven days")], "one")

    kb.sources = {"notes": Notes()}
    kb.sync()


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "openai/gpt-oss-20b", "another-model"])
async def test_openai_protocol_real_mcp_agent_loop(kb, model_id):
    populate(kb)
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == model_id
        assert all("is_error" not in m for m in body["messages"])
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "read-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query":"refund"}'},
                    }
                ],
            }
            reason = "tool_calls"
        else:
            assert "Refund within seven days" in body["messages"][-1]["content"]
            message, reason = {"role": "assistant", "content": "Seven days, per notes."}, "stop"
        return httpx.Response(
            200,
            json={
                "model": "served-alias",
                "choices": [{"message": message, "finish_reason": reason}],
            },
        )

    config = GenerationConfig(
        provider="openai-compatible", model=model_id, base_url="http://generator/v1"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        model = HttpModel(config, http)
        async with Client(create_server(kb)) as client:
            result = await run_agent("refund?", model, client)
    assert result["status"] == "completed"
    assert result["served_model"] == "served-alias"
    assert result["evidence_reads"] == 1
    assert result["trace"][0]["tool"] == "search"


async def test_anthropic_native_tool_result_translation(kb, monkeypatch):
    populate(kb)
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "dummy-test-key")
    requests = []

    def respond(request):
        assert request.headers["x-api-key"] == "dummy-test-key"
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert body["tools"][0]["input_schema"]
            content = [
                {"type": "tool_use", "id": "t1", "name": "search", "input": {"query": "refund"}}
            ]
            reason = "tool_use"
        else:
            assert body["messages"][-1]["content"][0]["type"] == "tool_result"
            content, reason = [{"type": "text", "text": "Evidence-based answer"}], "end_turn"
        return httpx.Response(
            200, json={"model": "test-claude", "stop_reason": reason, "content": content}
        )

    config = GenerationConfig(
        provider="anthropic",
        model="test-claude",
        token_env="TEST_ANTHROPIC_KEY",
        base_url="https://api.example/v1",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        async with Client(create_server(kb)) as client:
            result = await run_agent("refund?", HttpModel(config, http), client)
    assert result["evidence_reads"] == 1


async def test_unknown_tool_never_executes_and_budget_is_bounded(kb):
    class Model:
        config = GenerationConfig(provider="openai-compatible", model="test")
        served_model = "test"

        async def complete(self, messages, tools):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "bad",
                        "function": {"name": "shell", "arguments": '{"command":"unsafe"}'},
                    }
                ],
            }

    async with Client(create_server(kb)) as client:
        result = await run_agent("question", Model(), client, max_steps=2)
    assert result["status"] == "budget_exhausted"
    assert result["tool_calls"] == 2
    assert all(event["status"] == "error" for event in result["trace"])


def test_anthropic_requires_an_api_key_not_ambient_cli_oauth(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "not-an-api-key")
    config = GenerationConfig(provider="anthropic", model="test", base_url="https://api.example/v1")
    with pytest.raises(ValueError, match="API key"):
        HttpModel(config)
