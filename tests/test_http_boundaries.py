import httpx
import pytest

from taco_shell_rag.adapters.inference import HttpEmbedder, HttpReranker
from taco_shell_rag.agent.models import GenerationConfig, HttpModel
from taco_shell_rag.indexer.sources import RestSource


@pytest.mark.parametrize("status", [302, 307])
def test_rest_redirect_policy_overrides_shared_client_defaults(status):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "attacker.invalid":
            return httpx.Response(200, json={"items": [], "next_cursor": None})
        return httpx.Response(status, headers={"location": "https://attacker.invalid/data"})

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        source = RestSource("api", "https://example.com/docs", client=client)
        with pytest.raises(ValueError, match=f"HTTP {status}"):
            source.scan(None)
    assert [request.url.host for request in requests] == ["example.com"]


def test_redirect_cannot_publish_an_empty_replacement_snapshot(kb):
    redirected = False
    hosts = []

    def respond(request):
        hosts.append(request.url.host)
        if request.url.host == "attacker.invalid":
            return httpx.Response(200, json={"items": [], "next_cursor": None})
        if redirected:
            return httpx.Response(307, headers={"location": "https://attacker.invalid/data"})
        return httpx.Response(
            200, json={"items": [{"id": "policy", "text": "published policy"}], "next_cursor": None}
        )

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        source = RestSource("api", "https://example.com/docs", client=client)
        kb.sources = {"api": source}
        assert kb.sync()["api"]["status"] == "ok"
        checkpoint = kb.ledger.checkpoint("api")
        redirected = True
        assert kb.sync()["api"]["status"] == "error"
        assert kb.ledger.checkpoint("api") == checkpoint
        assert kb.ledger.status()["documents"] == 1
    assert hosts == ["example.com", "example.com"]


@pytest.mark.parametrize("provider", ["openai-compatible", "anthropic"])
@pytest.mark.parametrize("status", [302, 307])
async def test_model_redirects_never_forward_shared_client_credentials(
    provider, status, monkeypatch
):
    monkeypatch.setenv("TEST_MODEL_KEY", "dummy-api-key")
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "attacker.invalid":
            return httpx.Response(
                200,
                json={
                    "model": "untrusted-endpoint",
                    "choices": [
                        {"finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}
                    ],
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "x"}],
                },
            )
        return httpx.Response(status, headers={"location": "https://attacker.invalid/generate"})

    config = GenerationConfig(
        provider=provider,
        model="test",
        base_url="https://example.com/v1",
        token_env="TEST_MODEL_KEY",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True
    ) as client:
        model = HttpModel(config, client)
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            await model.complete([{"role": "user", "content": "question"}], [])
    assert [request.url.host for request in requests] == ["example.com"]
    if provider == "anthropic":
        assert requests[0].headers["x-api-key"] == "dummy-api-key"
    else:
        assert requests[0].headers["Authorization"] == "Bearer dummy-api-key"


@pytest.mark.parametrize(
    "adapter,operation,args",
    [
        (HttpEmbedder, "space", ()),
        (HttpEmbedder, "embed", (["text"], "document")),
        (HttpReranker, "describe", ()),
        (HttpReranker, "rerank", ("query", ["text"])),
    ],
)
@pytest.mark.parametrize("status", [302, 307])
def test_inference_redirect_policy_overrides_shared_client_defaults(
    adapter, operation, args, status
):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "attacker.invalid":
            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "embedding_space": {
                        "model": "fake",
                        "revision": "test",
                        "dimension": 3,
                        "device": "cuda:0",
                    },
                    "vectors": [[1.0, 0.0, 0.0]],
                    "scores": [0.2],
                },
            )
        return httpx.Response(status, headers={"location": "https://attacker.invalid/infer"})

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        inference = adapter("https://example.com", client)
        with pytest.raises(httpx.HTTPStatusError, match=str(status)):
            getattr(inference, operation)(*args)
    assert [request.url.host for request in requests] == ["example.com"]
