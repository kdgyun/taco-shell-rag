import json
from hashlib import sha256
from pathlib import Path

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
