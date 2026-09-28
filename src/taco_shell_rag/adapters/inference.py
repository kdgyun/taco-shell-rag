import httpx

from taco_shell_rag.core.contracts import (
    EmbeddingResult,
    EmbeddingSpace,
    RerankerInfo,
    RerankResult,
)


class HttpEndpoint:
    def __init__(self, url: str, client: httpx.Client | None = None, token: str | None = None):
        parsed = httpx.URL(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("inference URL must not embed credentials, queries or fragments")
        self.url = url.rstrip("/")
        self.client = client or httpx.Client(timeout=90, follow_redirects=False, trust_env=False)
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def health(self) -> dict:
        # 재사용 클라이언트의 기본값과 별개로 각 요청에서 리다이렉트를 금지한다.
        # Disable redirects on every request, independent of a reused client's defaults.
        response = self.client.get(
            f"{self.url}/healthz", headers=self.headers, follow_redirects=False
        )
        response.raise_for_status()
        data = response.json()
        if data.get("ready") is not True:
            raise ValueError("GPU inference is not ready")
        return data

    def post(self, path: str, body: dict) -> dict:
        response = self.client.post(
            f"{self.url}/{path}", json=body, headers=self.headers, follow_redirects=False
        )
        response.raise_for_status()
        return response.json()


class HttpEmbedder(HttpEndpoint):
    def space(self) -> EmbeddingSpace:
        data = self.health().get("embedding_space")
        if not data:
            raise ValueError("configured embedding endpoint does not serve an embedding model")
        return EmbeddingSpace(**data)

    def embed(self, texts: list[str], purpose: str) -> EmbeddingResult:
        data = self.post("embed", {"texts": texts, "purpose": purpose})
        return EmbeddingResult(data["vectors"], EmbeddingSpace(**data["embedding_space"]))


class HttpReranker(HttpEndpoint):
    def describe(self) -> RerankerInfo:
        data = self.health().get("reranker")
        if not data:
            raise ValueError("configured reranker endpoint does not serve a reranker")
        return RerankerInfo(**data)

    def rerank(self, query: str, texts: list[str]) -> RerankResult:
        data = self.post("rerank", {"query": query, "texts": texts})
        # 실행 응답의 모델을 기록한다. 별도 health 조회의 모델로 추측하지 않는다.
        # Attribute scores to the execution response, not a separate health snapshot.
        return RerankResult(data["scores"], RerankerInfo(**data["reranker"]))
