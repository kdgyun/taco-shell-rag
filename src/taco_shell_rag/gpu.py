import os
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, Field

from .core.contracts import EmbeddingSpace


def require_cuda(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and container GPU access.")


class GpuModels:
    def __init__(self):
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        require_cuda(torch)
        model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        revision = os.environ.get("EMBEDDING_MODEL_REVISION", "")
        reranker = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        reranker_revision = os.environ.get("RERANKER_MODEL_REVISION", "")
        if not all(re.fullmatch(r"[a-f0-9]{40}", value) for value in [revision, reranker_revision]):
            raise ValueError("set immutable embedding and reranker revisions before loading models")
        self.encoder = SentenceTransformer(
            model, revision=revision, device="cuda", trust_remote_code=False
        )
        self.reranker = CrossEncoder(
            reranker,
            revision=reranker_revision,
            device="cuda",
            trust_remote_code=False,
            max_length=512,
            activation_fn=torch.nn.Identity(),
        )
        if self.encoder.device.type != "cuda" or self.reranker.model.device.type != "cuda":
            raise RuntimeError("a model is not on CUDA; CPU fallback is disabled")
        self.info = EmbeddingSpace(
            model,
            revision,
            self.encoder.get_sentence_embedding_dimension(),
            str(self.encoder.device),
        )
        self.reranker_info = {"model": reranker, "revision": reranker_revision}
        self.lock = threading.Lock()

    def embed(self, texts, purpose):
        with self.lock:
            method = (
                self.encoder.encode_query if purpose == "query" else self.encoder.encode_document
            )
            return method(
                texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

    def rerank(self, query, texts):
        with self.lock:
            return self.reranker.predict(
                [(query, text) for text in texts], batch_size=8, show_progress_bar=False
            ).tolist()


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
            "embedding_space": asdict(engine.info),
            "reranker": engine.reranker_info,
        }

    @app.post("/embed")
    def embed(request: EmbedRequest):
        bounded(request.texts)
        return {"vectors": app.state.models.embed(request.texts, request.purpose)}

    @app.post("/rerank")
    def rerank(request: RerankRequest):
        bounded(request.texts)
        return {"scores": app.state.models.rerank(request.query, request.texts)}

    return app
