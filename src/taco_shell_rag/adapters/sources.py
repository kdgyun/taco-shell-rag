import json
import os
from hashlib import sha256
from pathlib import Path

import httpx

from taco_shell_rag.core.contracts import Document, Snapshot


def inventory(documents: list[Document]) -> Snapshot:
    hashes = sorted((doc.key, doc.content_hash) for doc in documents)
    return Snapshot(documents, sha256(json.dumps(hashes).encode()).hexdigest())


class FolderSource:
    def __init__(self, source_id: str, root: Path, max_bytes=2_000_000, max_total=20_000_000):
        self.id, self.root = source_id, root.resolve()
        self.max_bytes, self.max_total = max_bytes, max_total

    def scan(self, checkpoint: str | None) -> Snapshot:
        if not self.root.is_dir():
            raise ValueError("source directory does not exist")
        documents, total = [], 0
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.is_symlink() or any(
                parent.is_symlink()
                for parent in path.parents
                if parent != self.root and self.root in parent.parents
            ):
                continue
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            # stat 이후 파일이 커질 수도 있으므로 실제로 읽는 바이트 수도 제한한다.
            # Bound the actual read: the file may grow after a stat call.
            with path.open("rb") as stream:
                raw = stream.read(self.max_bytes + 1)
            total += len(raw)
            if len(raw) > self.max_bytes or total > self.max_total or len(documents) >= 1000:
                raise ValueError("folder snapshot exceeds its safety bound; narrow the source")
            text = raw.decode("utf-8")
            documents.append(
                Document(
                    str(relative),
                    self.id,
                    text,
                    title=path.stem,
                    path=str(relative),
                    revision=sha256(raw).hexdigest(),
                )
            )
        return inventory(documents)


class RestSource:
    """지정된 GET 주소에서 커서로 페이지를 읽는다. 응답에 담긴 URL은 따라가지 않는다.

    A fixed GET endpoint with cursor pagination; never follows API-provided URLs.
    """

    def __init__(
        self,
        source_id: str,
        url: str,
        fields: dict[str, str] | None = None,
        token_env: str = "",
        allow_http=False,
        client: httpx.Client | None = None,
    ):
        parsed = httpx.URL(url)
        if parsed.scheme != "https" and not (allow_http and parsed.scheme == "http"):
            raise ValueError("REST sources require HTTPS unless allow_http is explicitly set")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("keep credentials, query parameters and fragments out of source URLs")
        self.id, self.url = source_id, url
        self.fields = fields or {"id": "id", "text": "text", "title": "title", "url": "url"}
        if not {"id", "text"} <= self.fields.keys() or not set(self.fields) <= {
            "id",
            "text",
            "title",
            "url",
            "path",
            "revision",
            "updated_at",
        }:
            raise ValueError("REST field mapping must include id/text and only document fields")
        token = os.environ.get(token_env) if token_env else None
        if token_env and not token:
            raise ValueError("REST source credential environment variable is unset")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = client or httpx.Client(timeout=30, follow_redirects=False, trust_env=False)

    def scan(self, checkpoint: str | None) -> Snapshot:
        cursor, seen, documents, total = None, set(), [], 0
        for _ in range(100):
            # 주입된 클라이언트가 리다이렉트를 허용해도 지정한 원본 주소를 벗어나지 않는다.
            # Stay on the configured source even if an injected client enables redirects.
            with self.client.stream(
                "GET",
                self.url,
                params={"cursor": cursor} if cursor else {},
                headers=self.headers,
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise ValueError(f"REST source returned HTTP {response.status_code}")
                raw = bytearray()
                for block in response.iter_bytes():
                    raw.extend(block)
                    if len(raw) > 4_000_000:
                        raise ValueError("REST response exceeds its byte bound")
            total += len(raw)
            if total > 20_000_000:
                raise ValueError("REST snapshot exceeds its total byte bound")
            page = json.loads(raw)
            if not isinstance(page, dict) or not isinstance(page.get("items"), list):
                raise ValueError("REST response must contain an items array")
            for item in page["items"]:
                if not isinstance(item, dict):
                    raise ValueError("REST item must be an object")
                mapped = {name: item.get(field, "") for name, field in self.fields.items()}
                if not all(isinstance(value, str) for value in mapped.values()) or not mapped["id"]:
                    raise ValueError("REST document fields must be strings with a nonempty id")
                documents.append(Document(source_id=self.id, **mapped))
                if len(documents) > 1000:
                    raise ValueError("REST snapshot exceeds its document bound")
            cursor = page.get("next_cursor")
            if cursor is None:
                return inventory(documents)
            if not isinstance(cursor, str) or not cursor or len(cursor) > 1024 or cursor in seen:
                raise ValueError("REST pagination cursor is invalid or cyclic")
            seen.add(cursor)
        raise ValueError("REST pagination exceeded its page bound")
