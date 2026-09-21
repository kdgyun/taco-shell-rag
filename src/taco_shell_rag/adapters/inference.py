import httpx

from taco_shell_rag.core.contracts import EmbeddingSpace


class HttpInference:
    def __init__(self, url: str, client: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.client = client or httpx.Client(timeout=90, follow_redirects=False, trust_env=False)

    def space(self) -> EmbeddingSpace:
        # 재사용 클라이언트의 기본값과 별개로 각 요청에서 리다이렉트를 금지한다.
        # Disable redirects on every request, independent of a reused client's defaults.
        response = self.client.get(f"{self.url}/healthz", follow_redirects=False)
        response.raise_for_status()
        data = response.json()
        if data.get("ready") is not True:
            raise ValueError("GPU inference is not ready")
        return EmbeddingSpace(**data["embedding_space"])

    def embed(self, texts: list[str], purpose: str) -> list[list[float]]:
        response = self.client.post(
            f"{self.url}/embed", json={"texts": texts, "purpose": purpose}, follow_redirects=False
        )
        response.raise_for_status()
        return response.json()["vectors"]

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        response = self.client.post(
            f"{self.url}/rerank", json={"query": query, "texts": texts}, follow_redirects=False
        )
        response.raise_for_status()
        return response.json()["scores"]
