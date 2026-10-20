import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .embedder.model import SentenceTransformerEmbedder
from .model_config import (
    ModelConfigurationError,
    ModelSettings,
    load_model_settings,
)
from .model_runtime import check_loaded_model as check_loaded_model
from .model_runtime import require_cuda as require_cuda
from .reranker.model import CrossEncoderReranker


class GpuModels:
    def __init__(self, settings: ModelSettings | None = None, role: str | None = None):
        role = role or os.environ.get("INFERENCE_ROLE", "all")
        if role not in {"all", "embedding", "reranker"}:
            raise ModelConfigurationError("INFERENCE_ROLE must be all, embedding or reranker")
        if settings is None:
            legacy = [
                "EMBEDDING_MODEL",
                "EMBEDDING_MODEL_REVISION",
                "RERANKER_MODEL",
                "RERANKER_MODEL_REVISION",
            ]
            if any(os.environ.get(name) for name in legacy):
                raise ModelConfigurationError(
                    "move legacy model environment values to models.toml and remove the old keys; "
                    "see docs/models.md"
                )
            settings = load_model_settings(Path(os.environ.get("MODEL_CONFIG", "models.toml")))
        if role != "all" and getattr(settings, role) is None:
            raise ModelConfigurationError("the requested inference role is not configured")
        self.encoder = (
            SentenceTransformerEmbedder(settings.embedding)
            if role in {"all", "embedding"} and settings.embedding is not None
            else None
        )
        self.reranker = (
            CrossEncoderReranker(settings.reranker)
            if role in {"all", "reranker"} and settings.reranker is not None
            else None
        )
        self.info = self.encoder.space() if self.encoder else None
        self.reranker_info = self.reranker.describe() if self.reranker else None
        # 한 프로세스에서 같은 GPU를 쓰면 색인과 검색의 피크 메모리가 겹치지 않게 한다.
        # Serialize shared-process GPU work to avoid overlapping indexing/search peaks.
        self.lock = threading.Lock()

    def embed(self, texts, purpose):
        with self.lock:
            return self.encoder.embed(texts, purpose)

    def rerank(self, query, texts):
        with self.lock:
            return self.reranker.rerank(query, texts)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)
    purpose: Literal["query", "document"]


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    texts: list[str] = Field(min_length=1, max_length=160)


def create_app(engine_factory=GpuModels):
    from fastapi import FastAPI, HTTPException

    @asynccontextmanager
    async def lifespan(app):
        app.state.models = engine_factory()
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    def bounded(texts):
        if any(not text or len(text) > 4000 for text in texts):
            raise HTTPException(422, "each text must contain 1..4000 characters")

    @app.get("/healthz")
    def health():
        engine = app.state.models
        return {
            "ready": True,
            "embedding_space": asdict(engine.info) if engine.info else None,
            "embedding_fingerprint": engine.info.fingerprint if engine.info else None,
            "reranker": asdict(engine.reranker_info) if engine.reranker_info else None,
        }

    @app.post("/embed")
    def embed(request: EmbedRequest):
        bounded(request.texts)
        if app.state.models.info is None:
            raise HTTPException(503, "embedding role is not enabled on this endpoint")
        result = app.state.models.embed(request.texts, request.purpose)
        return {
            "vectors": result.vectors,
            "embedding_space": asdict(result.space),
            "embedding_fingerprint": result.space.fingerprint,
        }

    @app.post("/rerank")
    def rerank(request: RerankRequest):
        bounded(request.texts)
        if app.state.models.reranker_info is None:
            raise HTTPException(503, "reranker role is not enabled on this endpoint")
        result = app.state.models.rerank(request.query, request.texts)
        return {"scores": result.scores, "reranker": asdict(result.info)}

    return app
