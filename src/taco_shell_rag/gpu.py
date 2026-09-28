import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .core.contracts import EmbeddingResult, EmbeddingSpace, RerankerInfo, RerankResult
from .model_config import (
    EmbeddingOptions,
    ModelConfigurationError,
    ModelSettings,
    RerankerOptions,
    load_model_settings,
)


def require_cuda(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and container GPU access.")


def check_loaded_model(model, precision):
    if model.device.type != "cuda":
        raise RuntimeError("a model is not on CUDA; CPU fallback is disabled")
    if str(model.dtype).removeprefix("torch.") != precision:
        raise RuntimeError("the requested model precision was not applied")


class SentenceTransformerEmbedder:
    def __init__(self, options: EmbeddingOptions):
        import torch
        from sentence_transformers import SentenceTransformer

        require_cuda(torch)
        self.options = options
        self.model = SentenceTransformer(
            options.model,
            revision=options.revision,
            device=options.device,
            backend="torch",
            model_kwargs={"dtype": getattr(torch, options.precision)},
            trust_remote_code=False,
        )
        check_loaded_model(self.model, options.precision)
        if options.max_length is not None:
            if options.max_length > self.model.max_seq_length:
                raise ValueError("embedding max_length exceeds the model's supported length")
            self.model.max_seq_length = options.max_length
        self.info = EmbeddingSpace(
            options.model,
            options.revision,
            self.model.get_embedding_dimension(),
            str(self.model.device),
            {
                "backend": options.backend,
                "precision": options.precision,
                "max_length": self.model.max_seq_length,
                "query_prompt": options.query_prompt,
                "document_prompt": options.document_prompt,
            },
        )

    def space(self):
        return self.info

    def embed(self, texts, purpose):
        method = self.model.encode_query if purpose == "query" else self.model.encode_document
        prompt = self.options.query_prompt if purpose == "query" else self.options.document_prompt
        vectors = method(
            texts,
            batch_size=self.options.batch_size,
            prompt=prompt,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        return EmbeddingResult(vectors, self.info)


class CrossEncoderReranker:
    def __init__(self, options: RerankerOptions):
        import torch
        from sentence_transformers import CrossEncoder

        require_cuda(torch)
        self.options = options
        self.model = CrossEncoder(
            options.model,
            revision=options.revision,
            device=options.device,
            backend="torch",
            model_kwargs={"dtype": getattr(torch, options.precision)},
            processor_kwargs={"model_max_length": options.max_length},
            trust_remote_code=False,
            activation_fn=torch.nn.Identity(),
        )
        check_loaded_model(self.model, options.precision)
        if self.model.num_labels != 1:
            raise ValueError("reranking requires one scalar per query/document pair")
        if self.model.max_seq_length != options.max_length:
            raise ValueError("the requested reranker max_length was not applied")
        prompt = options.prompt
        if prompt is None:
            prompt = self.model.prompts.get(self.model.default_prompt_name)
        self.info = RerankerInfo(
            options.model,
            options.revision,
            str(self.model.device),
            options.backend,
            options.precision,
            self.model.max_seq_length,
            options.batch_size,
            prompt,
        )

    def describe(self):
        return self.info

    def rerank(self, query, texts):
        scores = self.model.predict(
            [(query, text) for text in texts],
            batch_size=self.options.batch_size,
            prompt=self.options.prompt,
            show_progress_bar=False,
        ).tolist()
        return RerankResult(scores, self.info)


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
            "reranker": asdict(engine.reranker_info) if engine.reranker_info else None,
        }

    @app.post("/embed")
    def embed(request: EmbedRequest):
        bounded(request.texts)
        if app.state.models.info is None:
            raise HTTPException(503, "embedding role is not enabled on this endpoint")
        result = app.state.models.embed(request.texts, request.purpose)
        return {"vectors": result.vectors, "embedding_space": asdict(result.space)}

    @app.post("/rerank")
    def rerank(request: RerankRequest):
        bounded(request.texts)
        if app.state.models.reranker_info is None:
            raise HTTPException(503, "reranker role is not enabled on this endpoint")
        result = app.state.models.rerank(request.query, request.texts)
        return {"scores": result.scores, "reranker": asdict(result.info)}

    return app
