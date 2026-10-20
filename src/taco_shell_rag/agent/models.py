import json
import os
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["openai-compatible", "anthropic", "claude-cli"]
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
    required_tools: list[
        Literal["search", "keyword_search", "deep_search", "get_document", "status"]
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_required_tools(self):
        if len(self.required_tools) != len(set(self.required_tools)):
            raise ValueError("required_tools must be unique")
        if self.required_tools and (self.provider == "claude-cli" or self.mode != "agent"):
            raise ValueError("required_tools is supported only by HTTP agent mode")
        return self


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
        if config.provider == "anthropic" and not self.token:
            raise ValueError(
                "Anthropic Messages requires its API key, not a Claude CLI OAuth token"
            )
        self.served_model = None

    async def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        if self.config.provider == "anthropic":
            return await self._anthropic(messages, tools)
        return await self._openai(messages, tools)

    async def _post(self, path, body, headers):
        # 리다이렉트로 API 키와 프롬프트가 다른 호스트에 넘어가지 않게 요청마다 차단한다.
        # Block redirects per request so API keys and prompts cannot reach another host.
        response = await self.client.post(
            self.config.base_url.rstrip("/") + path,
            json=body,
            headers=headers,
            follow_redirects=False,
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

    async def _anthropic(self, messages, tools):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        translated = []
        for message in messages:
            role = message["role"]
            if role == "system":
                continue
            if role == "tool":
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                        "is_error": message.get("is_error", False),
                    }
                ]
                role = "user"
            else:
                content = (
                    [{"type": "text", "text": message["content"]}] if message.get("content") else []
                )
                for call in message.get("tool_calls", []):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": json.loads(call["function"]["arguments"]),
                        }
                    )
            if translated and translated[-1]["role"] == role:
                translated[-1]["content"].extend(content)
            else:
                translated.append({"role": role, "content": content})
        body = {
            "model": self.config.model,
            "system": system,
            "messages": translated,
            "max_tokens": self.config.max_output_tokens,
        }
        if tools:
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
        data = await self._post(
            "/messages", body, {"x-api-key": self.token, "anthropic-version": "2023-06-01"}
        )
        if data.get("stop_reason") not in {"end_turn", "tool_use"}:
            raise RuntimeError("Anthropic response is incomplete")
        calls, texts = [], []
        for block in data["content"]:
            if block["type"] == "text":
                texts.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        },
                    }
                )
        return {"role": "assistant", "content": "\n".join(texts), "tool_calls": calls}
