import os
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["openai-compatible"]
    model: str = Field(min_length=1)
    base_url: str = ""
    token_env: str = ""
    max_output_tokens: int = Field(default=2048, ge=128, le=16384)
    max_steps: int = Field(default=8, ge=1, le=20)
    timeout: int = Field(default=120, ge=5, le=600)
    cli_bin: str = "claude"
    mode: Literal["agent", "rag"] = "agent"
    mcp_url: str = "http://127.0.0.1:8080/mcp"
    mcp_token_env: str = "MCP_AUTH_TOKEN"


class HttpModel:
    def __init__(self, config: GenerationConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self.client = client or httpx.AsyncClient(
            timeout=config.timeout, follow_redirects=False, trust_env=False
        )
        if not config.base_url:
            raise ValueError("HTTP generation requires an explicit base_url")
        url = httpx.URL(config.base_url)
        if url.scheme not in {"http", "https"} or url.username or url.password or url.query:
            raise ValueError("generation base_url must not embed credentials or query parameters")
        self.token = os.environ.get(config.token_env) if config.token_env else None
        if config.token_env and not self.token:
            raise ValueError("generation credential environment variable is unset")
        self.served_model = None

    async def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        return await self._openai(messages, tools)

    async def _post(self, path, body, headers):
        response = await self.client.post(
            self.config.base_url.rstrip("/") + path, json=body, headers=headers
        )
        if not response.is_success:
            raise RuntimeError(f"generation endpoint returned HTTP {response.status_code}")
        data = response.json()
        self.served_model = data.get("model")
        return data

    async def _openai(self, messages, tools):
        messages = [
            {key: value for key, value in message.items() if key != "is_error"}
            for message in messages
        ]
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        body = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_output_tokens,
        }
        if tools:
            body.update(tools=tools, tool_choice="auto")
        data = await self._post("/chat/completions", body, headers)
        choice = data["choices"][0]
        message = choice["message"]
        if choice.get("finish_reason") not in {"stop", "tool_calls"}:
            raise RuntimeError("generation ended without a complete answer or tool request")
        return {
            key: message[key]
            for key in ["role", "content", "tool_calls", "reasoning_content"]
            if key in message
        }
