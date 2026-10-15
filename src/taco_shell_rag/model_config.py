import tomllib
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# 프리셋은 편의용 고정 스냅샷이다. 사용자 모델을 이 목록으로 제한하지 않는다.
# Presets are pinned conveniences, not an allowlist for custom model identities.
PRESETS = {
    "embedding": {
        "bge-m3": {
            "model": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        },
        "multilingual-e5-base": {
            "model": "intfloat/multilingual-e5-base",
            "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
            "query_prompt": "query: ",
            "document_prompt": "passage: ",
        },
        "multilingual-e5-large": {
            "model": "intfloat/multilingual-e5-large",
            "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
            "query_prompt": "query: ",
            "document_prompt": "passage: ",
        },
        "multilingual-e5-large-instruct": {
            "model": "intfloat/multilingual-e5-large-instruct",
            "revision": "274baa43b0e13e37fafa6428dbc7938e62e5c439",
            "query_prompt": "Instruct: Retrieve passages relevant to the question.\nQuery: ",
            "document_prompt": "",
        },
        "qwen3-embedding-0.6b": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        },
        "qwen3-embedding-4b": {
            "model": "Qwen/Qwen3-Embedding-4B",
            "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",
        },
        "qwen3-embedding-8b": {
            "model": "Qwen/Qwen3-Embedding-8B",
            "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        },
    },
    "reranker": {
        "bge-reranker-v2-m3": {
            "model": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        },
        "bge-reranker-large": {
            "model": "BAAI/bge-reranker-large",
            "revision": "55611d7bca2a7133960a6d3b71e083071bbfc312",
        },
        "mxbai-rerank-base-v1": {
            "model": "mixedbread-ai/mxbai-rerank-base-v1",
            "revision": "800f24c113213a187e65bde9db00c15a2bb12738",
        },
        "qwen3-reranker-0.6b": {
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        },
        "qwen3-reranker-4b": {
            "model": "Qwen/Qwen3-Reranker-4B",
            "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
        },
        "qwen3-reranker-8b": {
            "model": "Qwen/Qwen3-Reranker-8B",
            "revision": "77d193c791ed757ca307ee72715aa132723da912",
        },
    },
}


class ModelConfigurationError(ValueError):
    """입력값이나 비밀값을 포함하지 않는 설정 안내.

    Configuration guidance that does not include input values or credentials.
    """


class ModelOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: ClassVar[str]
    preset: str | None = None
    model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    device: str = Field(default="cuda:0", pattern=r"^cuda(:[0-9]+)?$")
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    batch_size: int = Field(default=8, ge=1, le=32)
    max_length: int | None = Field(default=None, ge=32, le=32768)

    @model_validator(mode="before")
    @classmethod
    def resolve_preset(cls, values):
        if not isinstance(values, dict) or not values.get("preset"):
            return values
        if "model" in values or "revision" in values:
            raise ModelConfigurationError("choose a preset OR a custom model and revision")
        try:
            preset = PRESETS[cls.role][values["preset"]]
        except (KeyError, TypeError):
            raise ModelConfigurationError("unknown role-specific preset; run taco models") from None
        return {**preset, **values}


class EmbeddingOptions(ModelOptions):
    role = "embedding"
    backend: Literal["sentence-transformers"] = "sentence-transformers"
    query_prompt: str | None = Field(default=None, max_length=2000)
    document_prompt: str | None = Field(default=None, max_length=2000)


class RerankerOptions(ModelOptions):
    role = "reranker"
    backend: Literal["cross-encoder"] = "cross-encoder"
    max_length: int = Field(default=512, ge=32, le=32768)
    prompt: str | None = Field(default=None, max_length=2000)


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    embedding: EmbeddingOptions | None = None
    reranker: RerankerOptions | None = None

    @model_validator(mode="after")
    def require_a_role(self):
        if self.embedding is None and self.reranker is None:
            raise ModelConfigurationError("configure at least one inference role")
        return self


def load_model_settings(path: Path) -> ModelSettings:
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    try:
        return ModelSettings.model_validate(values)
    except ValidationError as error:
        # Pydantic 오류 원문에는 설정 입력이 포함될 수 있다.
        # Pydantic diagnostics may include configuration input values.
        for item in error.errors(include_input=False):
            cause = item.get("ctx", {}).get("error")
            if isinstance(cause, ModelConfigurationError):
                raise cause from None
        raise ModelConfigurationError(
            "invalid model settings; check roles, backend, pinned revision and CUDA options "
            "in docs/models.md"
        ) from None


def model_catalog() -> dict:
    return {
        "presets": PRESETS,
        "generation_providers": ["claude-cli", "anthropic", "openai-compatible"],
        "generation_examples": {
            name: f"examples/models/{name}.toml"
            for name in [
                "claude-cli",
                "claude-api",
                "qwen",
                "gpt-oss",
                "llama",
                "mistral",
                "rag-only",
            ]
        },
        "limitations": "Presets are configuration examples, not GPU-fit or quality guarantees. "
        "Custom models require a compatible backend and an immutable revision.",
    }
