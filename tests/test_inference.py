from dataclasses import asdict

import httpx
import pytest

from taco_shell_rag.adapters.inference import HttpEmbedder, HttpReranker
from taco_shell_rag.core.contracts import EmbeddingSpace, RerankerInfo


def test_separate_endpoints_keep_requests_credentials_and_model_identity_independent():
    space = EmbeddingSpace("test-encoder", "a" * 40, 2, "cuda:0")
    info = RerankerInfo("test-reranker", "b" * 40, "cuda:1", "cross-encoder")
    seen = []

    def respond(request):
        seen.append((request.url.host, request.url.path, request.headers["authorization"]))
        if request.url.host == "embedding.test":
            if request.url.path == "/healthz":
                return httpx.Response(200, json={"ready": True, "embedding_space": asdict(space)})
            assert request.url.path == "/embed"
            return httpx.Response(
                200, json={"vectors": [[1.0, 0.5]], "embedding_space": asdict(space)}
            )
        assert request.url.host == "reranker.test"
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"ready": True, "reranker": asdict(info)})
        assert request.url.path == "/rerank"
        return httpx.Response(200, json={"scores": [0.7], "reranker": asdict(info)})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        embedder = HttpEmbedder("https://embedding.test", client, token="embedding-test-token")
        reranker = HttpReranker("https://reranker.test", client, token="reranker-test-token")
        assert embedder.space() == space
        embedded = embedder.embed(["document"], "document")
        assert embedded.vectors == [[1.0, 0.5]] and embedded.space == space
        assert reranker.describe() == info
        result = reranker.rerank("question", ["document"])
        assert result.info == info and result.scores == [0.7]
    assert seen == [
        ("embedding.test", "/healthz", "Bearer embedding-test-token"),
        ("embedding.test", "/embed", "Bearer embedding-test-token"),
        ("reranker.test", "/healthz", "Bearer reranker-test-token"),
        ("reranker.test", "/rerank", "Bearer reranker-test-token"),
    ]


def test_reranker_scores_are_attributed_to_the_execution_response():
    info = RerankerInfo("served-model", "c" * 40, "cuda:0", "cross-encoder")

    def respond(request):
        assert request.url.path == "/rerank"
        return httpx.Response(200, json={"scores": [0.2], "reranker": asdict(info)})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = HttpReranker("https://reranker.test", client).rerank("q", ["text"])
    assert result.info.model == "served-model"


def test_embedding_vectors_are_attributed_to_the_execution_response():
    space = EmbeddingSpace("served-encoder", "d" * 40, 2, "cuda:0")

    def respond(request):
        assert request.url.path == "/embed"
        return httpx.Response(
            200, json={"vectors": [[1.0, 0.5]], "embedding_space": asdict(space)}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = HttpEmbedder("https://embedding.test", client).embed(["text"], "query")
    assert result.space == space


def test_embedding_response_without_execution_identity_is_rejected():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"vectors": [[1.0]]}))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(KeyError, match="embedding_space"):
            HttpEmbedder("https://embedding.test", client).embed(["text"], "query")


@pytest.mark.parametrize("adapter,method", [(HttpEmbedder, "space"), (HttpReranker, "describe")])
@pytest.mark.parametrize("ready", [True, False])
def test_unready_or_wrong_role_endpoints_are_rejected(adapter, method, ready):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"ready": ready}))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ValueError):
            getattr(adapter("https://inference.test", client), method)()


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@inference.test",
        "https://inference.test?token=secret",
        "https://inference.test#secret",
        "file:///private/model",
    ],
)
def test_inference_url_rejects_embedded_credentials_and_non_http_schemes(url):
    with pytest.raises(ValueError, match="inference URL"):
        HttpEmbedder(url)
