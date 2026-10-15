import fcntl
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from taco_shell_rag.core.contracts import Chunk, Document, EmbeddingSpaceMismatch


class Catalog:
    """검색에 노출할 문서 버전을 관리한다. 벡터 저장만으로는 게시되지 않는다.

    Authoritative publication ledger; vector writes alone do not publish a version.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    key TEXT PRIMARY KEY, source TEXT NOT NULL, version TEXT NOT NULL,
                    document TEXT NOT NULL, chunks TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retired (
                    key TEXT NOT NULL, version TEXT NOT NULL, PRIMARY KEY (key, version)
                );
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, checkpoint TEXT, status TEXT NOT NULL,
                    attempted_at TEXT, completed_at TEXT, documents INTEGER DEFAULT 0,
                    error TEXT
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def writer(self):
        with self.path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another sync owns the writer lock") from error
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def check_space(self, fingerprint: str, *, initialize: bool = False):
        with self.connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key='embedding_space'").fetchone()
            if row and row[0] != fingerprint:
                raise EmbeddingSpaceMismatch(
                    "embedding space changed; use a new collection and catalog"
                )
            if not row and initialize:
                db.execute("INSERT INTO settings VALUES ('embedding_space', ?)", (fingerprint,))

    def get(self, key: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        return Document(**json.loads(row["document"])), [
            Chunk(**c) for c in json.loads(row["chunks"])
        ]

    def activate(self, document: Document, chunks: list[Chunk]):
        with self.connect() as db:
            old = db.execute(
                "SELECT version FROM documents WHERE key=?", (document.key,)
            ).fetchone()
            if old and old[0] != chunks[0].version:
                db.execute("INSERT OR IGNORE INTO retired VALUES (?,?)", (document.key, old[0]))
            db.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?)",
                (
                    document.key,
                    document.source_id,
                    chunks[0].version,
                    json.dumps(asdict(document)),
                    json.dumps([asdict(c) for c in chunks]),
                ),
            )
            db.execute(
                "DELETE FROM retired WHERE key=? AND version=?", (document.key, chunks[0].version)
            )

    def prepare_version(self, key: str, version: str):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO retired VALUES (?,?)", (key, version))

    def retire_missing(self, source_id: str, keep: set[str]):
        with self.connect() as db:
            for row in db.execute("SELECT key,version FROM documents WHERE source=?", (source_id,)):
                if row["key"] not in keep:
                    db.execute("INSERT OR IGNORE INTO retired VALUES (?,?)", tuple(row))
                    db.execute("DELETE FROM documents WHERE key=?", (row["key"],))

    def retired(self):
        with self.connect() as db:
            return [tuple(row) for row in db.execute("SELECT key,version FROM retired")]

    def cleaned(self, key: str, version: str):
        with self.connect() as db:
            db.execute("DELETE FROM retired WHERE key=? AND version=?", (key, version))

    def checkpoint(self, source_id: str):
        with self.connect() as db:
            row = db.execute("SELECT checkpoint FROM sources WHERE id=?", (source_id,)).fetchone()
            return row[0] if row else None

    def started(self, source_id: str, now: str):
        with self.connect() as db:
            db.execute(
                """INSERT INTO sources (id,status,attempted_at) VALUES (?,'running',?)
                ON CONFLICT(id) DO UPDATE SET status='running',attempted_at=excluded.attempted_at,
                error=NULL""",
                (source_id, now),
            )

    def finished(self, source_id: str, checkpoint: str, count: int, now: str):
        with self.connect() as db:
            db.execute(
                """UPDATE sources SET checkpoint=?,documents=?,completed_at=?,
                status='ok',error=NULL WHERE id=?""",
                (checkpoint, count, now, source_id),
            )

    def failed(self, source_id: str, kind: str):
        with self.connect() as db:
            # 외부 예외 메시지에는 토큰이나 응답 본문이 섞일 수 있어 그대로 저장하지 않는다.
            # External exceptions may contain secrets or response bodies; do not persist them.
            db.execute("UPDATE sources SET status='error',error=? WHERE id=?", (kind, source_id))

    def status(self):
        with self.connect() as db:
            return {
                "sources": [dict(row) for row in db.execute("SELECT * FROM sources ORDER BY id")],
                "documents": db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "pending_cleanup": db.execute("SELECT COUNT(*) FROM retired").fetchone()[0],
            }
