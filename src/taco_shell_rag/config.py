import os
import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from qdrant_client import QdrantClient

from .adapters.catalog import Catalog
from .adapters.inference import HttpEmbedder, HttpReranker
from .adapters.qdrant import QdrantIndex
from .adapters.sources import FolderSource, RestSource
from .agent.models import GenerationConfig
from .core.service import KnowledgeBase


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    type: Literal["folder", "rest"]
    path: str = ""
    url: str = ""
    token_env: str = ""
    allow_http: bool = False
    fields: dict[str, str] | None = None

    @model_validator(mode="after")
    def required_options(self):
        if self.type == "folder" and not self.path:
            raise ValueError("folder source requires path")
        if self.type == "rest" and not self.url:
            raise ValueError("REST source requires url")
        if self.token_env and not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.token_env):
            raise ValueError("token_env must name an environment variable")
        return self


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    embedding_url: str = Field(
        default="http://inference:8000",
        validation_alias=AliasChoices("embedding_url", "inference_url"),
    )
    reranker_url: str | None = None
    embedding_token_env: str = Field(default="", pattern=r"^([A-Z][A-Z0-9_]*)?$")
    reranker_token_env: str = Field(default="", pattern=r"^([A-Z][A-Z0-9_]*)?$")
    qdrant_url: str = "http://qdrant:6333"
    qdrant_token_env: str = ""
    collection: str = Field(default="taco", pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    catalog: str = "data/catalog.sqlite3"
    chunk_size: int = Field(default=1000, ge=64, le=4000)
    overlap: int = Field(default=120, ge=0)
    rerank: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    token_env: str = "MCP_AUTH_TOKEN"
    allowed_hosts: list[str] | None = None
    allowed_origins: list[str] | None = None
    sources: list[SourceConfig] = Field(min_length=1)
    generation: GenerationConfig | None = None

    @model_validator(mode="after")
    def validate_overlap_and_identity(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError("source IDs must be unique")
        return self


def load_settings(path: Path) -> Settings:
    with path.open("rb") as stream:
        return Settings.model_validate(tomllib.load(stream))


def build_knowledge_base(settings: Settings, config_dir: Path) -> KnowledgeBase:
    sources = []
    for source in settings.sources:
        if source.type == "folder":
            sources.append(FolderSource(source.id, config_dir / source.path))
        else:
            sources.append(
                RestSource(
                    source.id, source.url, source.fields, source.token_env, source.allow_http
                )
            )

    def credential(name):
        value = os.environ.get(name) if name else None
        if name and not value:
            raise ValueError("configured credential environment variable is unset")
        return value

    api_key = credential(settings.qdrant_token_env)
    index = QdrantIndex(
        QdrantClient(settings.qdrant_url, api_key=api_key, timeout=30), settings.collection
    )
    return KnowledgeBase(
        Catalog(config_dir / settings.catalog),
        HttpEmbedder(settings.embedding_url, token=credential(settings.embedding_token_env)),
        index,
        sources,
        settings.chunk_size,
        settings.overlap,
        settings.rerank,
        HttpReranker(
            settings.reranker_url or settings.embedding_url,
            token=credential(settings.reranker_token_env),
        )
        if settings.rerank
        else None,
    )
