import httpx

from taco_shell_rag.core.contracts import EmbeddingSpace


class HttpInference:
    def __init__(self, url: str, client: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.client = client or httpx.Client(timeout=90, follow_redirects=False, trust_env=False)

    def space(self) -> EmbeddingSpace:
        response = self.client.get(f"{self.url}/healthz")
        response.raise_for_status()
        data = response.json()
        if data.get("ready") is not True:
            raise ValueError("GPU inference is not ready")
        return EmbeddingSpace(**data["embedding_space"])

    def embed(self, texts: list[str], purpose: str) -> list[list[float]]:
        response = self.client.post(f"{self.url}/embed", json={"texts": texts, "purpose": purpose})
        response.raise_for_status()
        return response.json()["vectors"]

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        response = self.client.post(f"{self.url}/rerank", json={"query": query, "texts": texts})
        response.raise_for_status()
        return response.json()["scores"]
