from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from taco_shell_rag.core.contracts import Chunk, Hit


class QdrantIndex:
    def __init__(self, client: QdrantClient, collection: str):
        self.client, self.collection = client, collection

    def setup(self, dimension: int):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        self.inspect(dimension)
        self.client.create_payload_index(
            self.collection, "source_id", models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            self.collection, "document_key", models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            self.collection, "version", models.PayloadSchemaType.KEYWORD
        )

    def inspect(self, dimension: int) -> dict:
        # 점검만으로 빈 컬렉션을 만들지 않는다. 최초 색인은 sync가 담당한다.
        # Inspection never creates a collection; the first sync owns initialization.
        if not self.client.collection_exists(self.collection):
            return {"collection": self.collection, "exists": False, "points": 0}
        info = self.client.get_collection(self.collection)
        configured = info.config.params.vectors
        if not isinstance(configured, models.VectorParams) or configured.size != dimension:
            raise ValueError("collection vector schema differs from the configured embedding space")
        if configured.distance != models.Distance.COSINE:
            raise ValueError("collection must use cosine distance")
        return {"collection": self.collection, "exists": True, "points": info.points_count}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]):
        for start in range(0, len(chunks), 64):
            points = [
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"taco:{c.document_key}:{c.version}:{c.index}")),
                    vector=vector,
                    payload=asdict(c),
                )
                for c, vector in zip(
                    chunks[start : start + 64], vectors[start : start + 64], strict=True
                )
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int, source_id: str | None):
        if not self.client.collection_exists(self.collection):
            return []
        selector = (
            models.Filter(
                must=[
                    models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))
                ]
            )
            if source_id
            else None
        )
        rows = self.client.query_points(
            self.collection, query=vector, query_filter=selector, limit=limit, with_payload=True
        ).points
        return [Hit(Chunk(**row.payload), row.score) for row in rows]

    def delete_version(self, document_key: str, version: str):
        selector = models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in {"document_key": document_key, "version": version}.items()
            ]
        )
        self.client.delete(self.collection, models.FilterSelector(filter=selector), wait=True)
